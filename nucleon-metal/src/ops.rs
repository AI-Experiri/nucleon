//! Safe Rust wrappers, one per kernel: slices in, Vec out. Each method
//! creates buffers, encodes one dispatch, waits, reads back.
//!
//! Study-grade on purpose: per-call buffer creation and a blocking wait per
//! op make every dispatch visible and measurable. A real backend keeps
//! weights resident and batches encodes — that difference IS the fusion
//! economics lesson, and the numbers land in chapter 11.

use crate::device::{Buffer, Gpu, Scalar, REDUCTION_THREADS};

/// Kernels receive dimensions as `uint`; a silent `as u32` truncation on
/// a huge shape would launch a correct-size grid with a wrong-size kernel
/// argument. Panic instead.
fn dim_u32(n: usize) -> u32 {
    n.try_into()
        .expect("dimension exceeds u32 (kernel uint) range")
}

/// Reduction kernels stride their row loop by the threadgroup size in
/// 32-bit math; a dim in the top 256 values of u32 would wrap the loop
/// index. Cap reduction dims below that (map ops only need dim_u32).
fn reduction_dim_u32(n: usize) -> u32 {
    let n32 = dim_u32(n);
    assert!(
        (n32 as u64) + (REDUCTION_THREADS as u64) <= u32::MAX as u64,
        "reduction dim {n} too close to u32::MAX (stride wrap)"
    );
    n32
}

pub struct MetalOps {
    gpu: Gpu,
}

impl MetalOps {
    pub fn new() -> Result<Self, String> {
        Ok(Self { gpu: Gpu::new()? })
    }

    pub fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        self.map2("add", a, b)
    }

    pub fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        self.map2("mul", a, b)
    }

    pub fn silu(&self, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }
        dim_u32(x.len()); // kernel indexes with 32-bit thread position
        let xb = self.gpu.buffer_from_f32(x);
        let out = self.gpu.buffer_for_output(x.len());
        unsafe { self.gpu.run("silu", &[&xb, &out], &[], (x.len(), 1), None) };
        self.gpu.read_f32(&out, x.len())
    }

    /// table is [vocab, dim] flat; returns row `id`. All preconditions are
    /// checked here: past this point a bad id is a GPU out-of-bounds read,
    /// not a Rust panic.
    pub fn embed(&self, table: &[f32], dim: usize, id: u32) -> Vec<f32> {
        assert!(dim > 0, "embed dim must be nonzero");
        assert!(
            table.len().is_multiple_of(dim),
            "table length {} is not a multiple of dim {dim}",
            table.len()
        );
        dim_u32(table.len()); // kernel flattens id*dim+i in 32-bit math
        let vocab = table.len() / dim;
        assert!((id as usize) < vocab, "token id {id} >= vocab {vocab}");
        let tb = self.gpu.buffer_from_f32(table);
        let ib = self.gpu.buffer_from_u32(&[id]);
        let out = self.gpu.buffer_for_output(dim);
        unsafe {
            self.gpu.run(
                "embed",
                &[&tb, &ib, &out],
                &[Scalar::U32(dim_u32(dim))],
                (dim, 1),
                None,
            )
        };
        self.gpu.read_f32(&out, dim)
    }

    /// w is [out_dim, in_dim] flat (checkpoint order); y = w @ x.
    pub fn matvec(&self, w: &[f32], x: &[f32]) -> Vec<f32> {
        let in_dim = x.len();
        assert!(in_dim > 0, "matvec input must be nonzero-length");
        assert_eq!(w.len() % in_dim, 0, "weight rows must match x length");
        dim_u32(w.len()); // kernel flattens row*in_dim+j in 32-bit math
        let out_dim = w.len() / in_dim;
        if out_dim == 0 {
            return Vec::new();
        }
        let wb = self.gpu.buffer_from_f32(w);
        let xb = self.gpu.buffer_from_f32(x);
        let yb = self.gpu.buffer_for_output(out_dim);
        unsafe {
            self.gpu.run(
                "matvec",
                &[&wb, &xb, &yb],
                &[Scalar::U32(dim_u32(in_dim))],
                (out_dim, 1),
                None,
            )
        };
        self.gpu.read_f32(&yb, out_dim)
    }

    /// a is [m, k] flat, b is [n, k] flat; returns a @ b^T as [m, n].
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        assert!(k > 0, "matmul inner dim must be nonzero");
        let mk = m.checked_mul(k).expect("m*k overflows");
        let nk = n.checked_mul(k).expect("n*k overflows");
        let mn = m.checked_mul(n).expect("m*n overflows");
        assert_eq!(a.len(), mk);
        assert_eq!(b.len(), nk);
        dim_u32(mk.max(nk).max(mn)); // kernels index flattened in 32-bit
        if mn == 0 {
            return Vec::new();
        }
        let ab = self.gpu.buffer_from_f32(a);
        let bb = self.gpu.buffer_from_f32(b);
        let cb = self.gpu.buffer_for_output(mn);
        unsafe {
            self.gpu.run(
                "matmul",
                &[&ab, &bb, &cb],
                &[Scalar::U32(dim_u32(k)), Scalar::U32(dim_u32(n))],
                (n, m),
                None,
            )
        };
        self.gpu.read_f32(&cb, mn)
    }

    /// x is [n_heads, head_dim] flat; rotates pairs in the NeoX half-split
    /// layout by position `pos`. Returns the rotated copy. `theta` (like
    /// `eps` elsewhere) comes from trusted model config and is not
    /// domain-checked here; the loader validates config values.
    pub fn rope(&self, x: &[f32], head_dim: usize, pos: u32, theta: f32) -> Vec<f32> {
        assert!(head_dim > 0, "rope head_dim must be nonzero");
        assert_eq!(x.len() % head_dim, 0);
        dim_u32(x.len()); // kernel flattens head*head_dim+i in 32-bit math
        assert_eq!(head_dim % 2, 0, "head_dim must be even to form pairs");
        if x.is_empty() {
            return Vec::new();
        }
        let n_pairs = x.len() / 2;
        let xb = self.gpu.buffer_from_f32(x);
        unsafe {
            self.gpu.run(
                "rope",
                &[&xb],
                &[
                    Scalar::U32(dim_u32(head_dim)),
                    Scalar::U32(pos),
                    Scalar::F32(theta),
                ],
                (n_pairs, 1),
                None,
            )
        };
        self.gpu.read_f32(&xb, x.len())
    }

    pub fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        assert_eq!(x.len(), weight.len());
        if x.is_empty() {
            return Vec::new();
        }
        reduction_dim_u32(x.len()); // validate before allocating buffers
        let xb = self.gpu.buffer_from_f32(x);
        let wb = self.gpu.buffer_from_f32(weight);
        let out = self.gpu.buffer_for_output(x.len());
        self.run_reduction(
            "rmsnorm",
            &[&xb, &wb, &out],
            &[Scalar::U32(dim_u32(x.len())), Scalar::F32(eps)],
        );
        self.gpu.read_f32(&out, x.len())
    }

    /// Non-finite inputs: -inf entries are valid masks (their probability
    /// is exactly 0). An all--inf row or any +inf entry yields NaN via
    /// inf - inf; callers guarantee at least one finite value.
    pub fn softmax(&self, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }
        reduction_dim_u32(x.len()); // validate before allocating buffers
        let xb = self.gpu.buffer_from_f32(x);
        let out = self.gpu.buffer_for_output(x.len());
        self.run_reduction("softmax", &[&xb, &out], &[Scalar::U32(dim_u32(x.len()))]);
        self.gpu.read_f32(&out, x.len())
    }

    /// Comparison policy (scan sentinel is -inf with strict >): NaN entries
    /// never win, and a literal -inf entry can never win either — inputs
    /// that are entirely NaN and/or -inf return index 0. Fine for greedy
    /// sampling, where -inf marks deliberately masked-out tokens.
    pub fn argmax(&self, x: &[f32]) -> u32 {
        assert!(!x.is_empty());
        reduction_dim_u32(x.len()); // validate before allocating buffers
        let xb = self.gpu.buffer_from_f32(x);
        let out = self.gpu.buffer_for_output(1);
        self.run_reduction("argmax", &[&xb, &out], &[Scalar::U32(dim_u32(x.len()))]);
        self.gpu.read_u32(&out, 1)[0]
    }

    /// Fused rmsnorm + matvec: y = w @ (rmsnorm(x) * norm_weight).
    /// One dispatch; the composed equivalent is rmsnorm() then matvec().
    /// Not bit-identical to the composed pair: inv_rms is applied after the
    /// dot product (linearity), which rounds differently and assumes
    /// intermediates stay finite — true for normal LLM value ranges.
    pub fn rmsnorm_matvec(&self, x: &[f32], norm_weight: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let dim = x.len();
        assert!(dim > 0, "rmsnorm_matvec input must be nonzero-length");
        assert_eq!(norm_weight.len(), dim);
        assert_eq!(w.len() % dim, 0);
        dim_u32(w.len()); // kernel flattens row*dim+j in 32-bit math
        reduction_dim_u32(dim); // fused kernel also strides sum_sq by group
        let out_dim = w.len() / dim;
        if out_dim == 0 {
            return Vec::new();
        }
        // Deliberately narrow contract (study-grade): the kernel's tree
        // reduction needs power-of-two groups, and this predicate is the
        // simple sufficient condition. Every transformer dim we target
        // (1024/2048/3072, Qwen3 shapes) satisfies it.
        assert!(
            out_dim.is_power_of_two() || out_dim.is_multiple_of(256),
            "fused kernel needs out_dim power-of-two or multiple of 256"
        );
        let xb = self.gpu.buffer_from_f32(x);
        let nb = self.gpu.buffer_from_f32(norm_weight);
        let wb = self.gpu.buffer_from_f32(w);
        let yb = self.gpu.buffer_for_output(out_dim);
        unsafe {
            self.gpu.run(
                "rmsnorm_matvec",
                &[&xb, &nb, &wb, &yb],
                &[Scalar::U32(dim_u32(dim)), Scalar::F32(eps)],
                (out_dim, 1),
                Some((out_dim.min(256), 1)),
            )
        };
        self.gpu.read_f32(&yb, out_dim)
    }

    /// Fused single-position GQA attention: q is [n_heads, head_dim] flat,
    /// k and v are [n_kv_heads, seq, head_dim] flat. One dispatch, one
    /// threadgroup per head. Study-grade limit: seq <= 4096 (score row
    /// lives in threadgroup memory).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        seq: usize,
        n_heads: usize,
        n_kv_heads: usize,
        scale: f32,
    ) -> Vec<f32> {
        assert!(head_dim > 0 && seq > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(seq <= 4096, "attention kernel caps seq at 4096");
        let q_elems = n_heads.checked_mul(head_dim).expect("q size overflows");
        let kv_elems = n_kv_heads
            .checked_mul(seq)
            .and_then(|x| x.checked_mul(head_dim))
            .expect("kv size overflows");
        assert_eq!(q.len(), q_elems);
        assert_eq!(k.len(), kv_elems);
        assert_eq!(v.len(), k.len());
        assert!(
            n_heads.is_multiple_of(n_kv_heads),
            "query heads must divide evenly over kv heads (GQA)"
        );
        // kernel flattens all three index spaces in 32-bit math, and the
        // output loop strides head_dim by the threadgroup size
        reduction_dim_u32(head_dim);
        dim_u32(q_elems);
        dim_u32(kv_elems);
        n_heads
            .checked_mul(256)
            .and_then(|g| u32::try_from(g).ok())
            .expect("attention grid exceeds u32");
        let group = n_heads / n_kv_heads;
        let qb = self.gpu.buffer_from_f32(q);
        let kb = self.gpu.buffer_from_f32(k);
        let vb = self.gpu.buffer_from_f32(v);
        let ob = self.gpu.buffer_for_output(n_heads * head_dim);
        unsafe {
            self.gpu.run(
                "attention",
                &[&qb, &kb, &vb, &ob],
                &[
                    Scalar::U32(dim_u32(head_dim)),
                    Scalar::U32(dim_u32(seq)),
                    Scalar::U32(dim_u32(group)),
                    Scalar::F32(scale),
                ],
                (n_heads * 256, 1),
                Some((256, 1)),
            )
        };
        self.gpu.read_f32(&ob, n_heads * head_dim)
    }

    /// The underlying device, for callers that batch their own dispatches
    /// (the fusion benchmark example does).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Elementwise two-input map: out[i] = op(a[i], b[i]).
    fn map2(&self, kernel: &str, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            // a zero-thread dispatch is not a valid GPU launch
            return Vec::new();
        }
        dim_u32(a.len()); // kernel indexes with 32-bit thread position
        let ab = self.gpu.buffer_from_f32(a);
        let bb = self.gpu.buffer_from_f32(b);
        let out = self.gpu.buffer_for_output(a.len());
        unsafe {
            self.gpu
                .run(kernel, &[&ab, &bb, &out], &[], (a.len(), 1), None)
        };
        self.gpu.read_f32(&out, a.len())
    }

    /// Reduction kernels run as ONE threadgroup of REDUCTION_THREADS: the
    /// whole row must share the same scratch memory and barriers.
    fn run_reduction(&self, kernel: &str, buffers: &[&Buffer], scalars: &[Scalar]) {
        unsafe {
            self.gpu.run(
                kernel,
                buffers,
                scalars,
                (REDUCTION_THREADS, 1),
                Some((REDUCTION_THREADS, 1)),
            )
        };
    }
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
