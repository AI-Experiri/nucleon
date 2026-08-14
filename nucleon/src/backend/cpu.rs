//! `CpuBackend` — the naive, readable, single-steppable reference.
//!
//! FROZEN POLICY: this backend is never optimized. Its job is to be
//! obviously correct — plain nested loops you can step through in a
//! debugger — so every other backend (Metal, later CUDA) can be judged
//! against it op-by-op. Speed lives elsewhere.

use crate::backend::ops::Backend;
use crate::tensor::Tensor;

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape(), b.shape(), "add needs matching shapes");
        let mut out = Vec::with_capacity(a.len());
        for (x, y) in a.data().iter().zip(b.data()) {
            out.push(x + y);
        }
        Tensor::new(a.shape().to_vec(), out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape(), b.shape(), "mul needs matching shapes");
        let mut out = Vec::with_capacity(a.len());
        for (x, y) in a.data().iter().zip(b.data()) {
            out.push(x * y);
        }
        Tensor::new(a.shape().to_vec(), out)
    }

    fn silu(&self, x: &Tensor) -> Tensor {
        let mut out = Vec::with_capacity(x.len());
        for &v in x.data() {
            out.push(v / (1.0 + (-v).exp()));
        }
        Tensor::new(x.shape().to_vec(), out)
    }

    fn embed(&self, table: &Tensor, id: u32) -> Tensor {
        assert_eq!(table.shape().len(), 2, "embed table must be [vocab, dim]");
        let dim = table.shape()[1];
        assert!(dim > 0, "embed dim must be nonzero");
        Tensor::new(vec![dim], table.row(id as usize).to_vec())
    }

    fn matvec(&self, w: &Tensor, x: &Tensor) -> Tensor {
        assert_eq!(w.shape().len(), 2, "weights must be [out_dim, in_dim]");
        assert_eq!(x.shape().len(), 1, "matvec input must be a 1-D vector");
        assert_eq!(
            w.shape()[1],
            x.len(),
            "weight columns must match input length"
        );
        assert!(!x.is_empty(), "matvec input must be nonzero-length");
        let out_dim = w.shape()[0];
        let mut y = Vec::with_capacity(out_dim);
        for row in 0..out_dim {
            let mut sum = 0.0;
            for (weight, input) in w.row(row).iter().zip(x.data()) {
                sum += weight * input;
            }
            y.push(sum);
        }
        Tensor::new(vec![out_dim], y)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape().len(), 2, "a must be [m, k]");
        assert_eq!(b.shape().len(), 2, "b must be [n, k]");
        assert_eq!(a.shape()[1], b.shape()[1], "inner dims must match");
        assert!(a.shape()[1] > 0, "matmul inner dim must be nonzero");
        let (m, n) = (a.shape()[0], b.shape()[0]);
        let out_len = m.checked_mul(n).expect("matmul output size overflows");
        let mut c = Vec::with_capacity(out_len);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for (x, y) in a.row(i).iter().zip(b.row(j)) {
                    sum += x * y;
                }
                c.push(sum);
            }
        }
        Tensor::new(vec![m, n], c)
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        assert_eq!(x.shape(), weight.shape(), "rmsnorm needs matching shapes");
        let mut sum_sq = 0.0;
        for &v in x.data() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
        let mut out = Vec::with_capacity(x.len());
        for (&v, &w) in x.data().iter().zip(weight.data()) {
            out.push(v * inv_rms * w);
        }
        Tensor::new(x.shape().to_vec(), out)
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        // subtract the max before exponentiating: exp(90) overflows f32,
        // and real logits reach that range
        let mut max = f32::NEG_INFINITY;
        for &v in x.data() {
            max = max.max(v);
        }
        let mut exps = Vec::with_capacity(x.len());
        let mut sum = 0.0;
        for &v in x.data() {
            let e = (v - max).exp();
            exps.push(e);
            sum += e;
        }
        for e in &mut exps {
            *e /= sum;
        }
        Tensor::new(x.shape().to_vec(), exps)
    }

    fn rope(&self, x: &Tensor, pos: u32, theta: f32) -> Tensor {
        assert_eq!(x.shape().len(), 2, "rope input must be [n_heads, head_dim]");
        let (n_heads, head_dim) = (x.shape()[0], x.shape()[1]);
        assert!(head_dim > 0, "rope head_dim must be nonzero");
        assert_eq!(head_dim % 2, 0, "head_dim must be even to form pairs");
        let half = head_dim / 2;
        let mut out = x.data().to_vec();
        for head in 0..n_heads {
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                let base = head * head_dim;
                let x0 = out[base + i];
                let x1 = out[base + i + half];
                out[base + i] = x0 * cos - x1 * sin;
                out[base + i + half] = x0 * sin + x1 * cos;
            }
        }
        Tensor::new(x.shape().to_vec(), out)
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
        assert_eq!(q.shape().len(), 2, "q must be [n_heads, head_dim]");
        assert_eq!(k.shape().len(), 3, "k must be [n_kv_heads, seq, head_dim]");
        assert_eq!(k.shape(), v.shape(), "k and v must match");
        let (n_heads, head_dim) = (q.shape()[0], q.shape()[1]);
        let (n_kv_heads, seq) = (k.shape()[0], k.shape()[1]);
        assert_eq!(k.shape()[2], head_dim, "k head_dim must match q");
        // an empty sequence or zero-sized head is a cache bug upstream;
        // fail loudly and identically on every backend
        assert!(
            head_dim > 0 && seq > 0 && n_heads > 0 && n_kv_heads > 0,
            "attention dims must be nonzero (empty cache?)"
        );
        assert!(seq <= 4096, "attention contract caps seq at 4096");
        assert_eq!(
            n_heads % n_kv_heads,
            0,
            "query heads must divide evenly over kv heads (GQA)"
        );
        let group = n_heads / n_kv_heads;

        let mut out = Vec::with_capacity(n_heads * head_dim);
        for head in 0..n_heads {
            let kv_head = head / group;
            let q_row = &q.data()[head * head_dim..(head + 1) * head_dim];

            // scores over the sequence, then a stable softmax
            let mut scores = Vec::with_capacity(seq);
            for t in 0..seq {
                let base = (kv_head * seq + t) * head_dim;
                let k_row = &k.data()[base..base + head_dim];
                let mut dot = 0.0;
                for (qi, ki) in q_row.iter().zip(k_row) {
                    dot += qi * ki;
                }
                scores.push(dot * scale);
            }
            let weights = self.softmax(&Tensor::new(vec![seq], scores));

            // weighted sum of the value rows
            let mut acc = vec![0.0; head_dim];
            for t in 0..seq {
                let base = (kv_head * seq + t) * head_dim;
                let v_row = &v.data()[base..base + head_dim];
                let w = weights.data()[t];
                for (a, vi) in acc.iter_mut().zip(v_row) {
                    *a += w * vi;
                }
            }
            out.extend_from_slice(&acc);
        }
        Tensor::new(vec![n_heads, head_dim], out)
    }

    fn argmax(&self, x: &Tensor) -> u32 {
        assert!(!x.is_empty(), "argmax needs a nonempty input");
        let mut best = f32::NEG_INFINITY;
        let mut best_idx = 0;
        for (i, &v) in x.data().iter().enumerate() {
            // strict > : ties keep the lowest index; NaN and -inf never win
            if v > best {
                best = v;
                best_idx = i;
            }
        }
        u32::try_from(best_idx).expect("argmax index exceeds u32")
    }
}

#[cfg(test)]
#[path = "cpu_tests.rs"]
mod tests;
