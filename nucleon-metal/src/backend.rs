//! MetalBackend: the GPU attached to the engine's Backend trait.
//!
//! Every op delegates to the kernel wrappers in `ops.rs`, converting
//! between `Tensor` (shape + data) and the flat slices kernels speak.
//! `CpuBackend` defines correct; the parity tests in backend_tests.rs
//! hold this implementation to it.

use nucleon::backend::Backend;
use nucleon::tensor::Tensor;

use crate::ops::MetalOps;

pub struct MetalBackend {
    ops: MetalOps,
}

impl MetalBackend {
    /// Fails with a message when there is no Metal device (headless CI)
    /// or a kernel does not compile.
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            ops: MetalOps::new()?,
        })
    }
}

impl Backend for MetalBackend {
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape(), b.shape(), "add needs matching shapes");
        Tensor::new(a.shape().to_vec(), self.ops.add(a.data(), b.data()))
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape(), b.shape(), "mul needs matching shapes");
        Tensor::new(a.shape().to_vec(), self.ops.mul(a.data(), b.data()))
    }

    fn silu(&self, x: &Tensor) -> Tensor {
        Tensor::new(x.shape().to_vec(), self.ops.silu(x.data()))
    }

    fn embed(&self, table: &Tensor, id: u32) -> Tensor {
        assert_eq!(table.shape().len(), 2, "embed table must be [vocab, dim]");
        let dim = table.shape()[1];
        Tensor::new(vec![dim], self.ops.embed(table.data(), dim, id))
    }

    fn matvec(&self, w: &Tensor, x: &Tensor) -> Tensor {
        assert_eq!(w.shape().len(), 2, "weights must be [out_dim, in_dim]");
        assert_eq!(w.shape()[1], x.len(), "weight columns must match input");
        Tensor::new(vec![w.shape()[0]], self.ops.matvec(w.data(), x.data()))
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.shape().len(), 2, "a must be [m, k]");
        assert_eq!(b.shape().len(), 2, "b must be [n, k]");
        assert_eq!(a.shape()[1], b.shape()[1], "inner dims must match");
        let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[0]);
        Tensor::new(vec![m, n], self.ops.matmul(a.data(), b.data(), m, k, n))
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        assert_eq!(x.shape(), weight.shape(), "rmsnorm needs matching shapes");
        Tensor::new(
            x.shape().to_vec(),
            self.ops.rmsnorm(x.data(), weight.data(), eps),
        )
    }

    fn softmax(&self, x: &Tensor) -> Tensor {
        Tensor::new(x.shape().to_vec(), self.ops.softmax(x.data()))
    }

    fn rope(&self, x: &Tensor, pos: u32, theta: f32) -> Tensor {
        assert_eq!(x.shape().len(), 2, "rope input must be [n_heads, head_dim]");
        let head_dim = x.shape()[1];
        Tensor::new(
            x.shape().to_vec(),
            self.ops.rope(x.data(), head_dim, pos, theta),
        )
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
        assert_eq!(q.shape().len(), 2, "q must be [n_heads, head_dim]");
        assert_eq!(k.shape().len(), 3, "k must be [n_kv_heads, seq, head_dim]");
        assert_eq!(k.shape(), v.shape(), "k and v must match");
        let (n_heads, head_dim) = (q.shape()[0], q.shape()[1]);
        let (n_kv_heads, seq) = (k.shape()[0], k.shape()[1]);
        assert_eq!(k.shape()[2], head_dim, "k head_dim must match q");
        Tensor::new(
            vec![n_heads, head_dim],
            self.ops.attention(
                q.data(),
                k.data(),
                v.data(),
                head_dim,
                seq,
                n_heads,
                n_kv_heads,
                scale,
            ),
        )
    }

    fn argmax(&self, x: &Tensor) -> u32 {
        self.ops.argmax(x.data())
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
