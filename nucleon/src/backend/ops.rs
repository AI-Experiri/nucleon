//! The `Backend` trait: the operations a transformer is made of.
//!
//! Tensor stores numbers; a Backend computes on them. The trait is the
//! seam between the model's wiring and the hardware: CpuBackend (this
//! crate) is the readable default attached at Tensor's birthplace, and
//! MetalBackend (nucleon-metal) attaches the GPU behind the same seam.
//!
//! Weight matrices follow checkpoint order [out_dim, in_dim], so matvec
//! computes y = W @ x as one dot product per output row. `attention` is
//! deliberately ONE op (not composed from matmul + softmax): backends
//! are then free to implement it fused, which the chapter-2 measurements
//! showed is where GPU performance lives.

use crate::tensor::Tensor;

/// Contract for every op: all dimensions are nonzero. Zero-sized shapes
/// are upstream bugs (an empty cache, a malformed config) and panic
/// identically on every backend, so generic code cannot pass on one
/// executor and die on another.
pub trait Backend {
    /// Elementwise a + b. Shapes must match.
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor;

    /// Elementwise a * b. Shapes must match.
    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;

    /// silu(x) = x * sigmoid(x), the activation inside SwiGLU MLPs.
    fn silu(&self, x: &Tensor) -> Tensor;

    /// Row `id` of an embedding table shaped [vocab, dim].
    fn embed(&self, table: &Tensor, id: u32) -> Tensor;

    /// y = w @ x with w shaped [out_dim, in_dim] (checkpoint order) and
    /// x a vector of in_dim. The op that dominates decode.
    fn matvec(&self, w: &Tensor, x: &Tensor) -> Tensor;

    /// c = a @ b^T with a [m, k] and b [n, k]; result [m, n].
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;

    /// rmsnorm(x) * weight, over the whole vector, eps inside the sqrt.
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor;

    /// Softmax over the whole vector, max-subtracted for stability.
    fn softmax(&self, x: &Tensor) -> Tensor;

    /// Rotate query/key pairs in the NeoX half-split layout. `x` is
    /// [n_heads, head_dim]; `pos` is the token's position in the sequence.
    fn rope(&self, x: &Tensor, pos: u32, theta: f32) -> Tensor;

    /// Single-position attention with GQA: one query per head attends
    /// over the cached sequence. q is [n_heads, head_dim]; k and v are
    /// [n_kv_heads, seq, head_dim]; heads share KV in groups of
    /// n_heads / n_kv_heads. Returns [n_heads, head_dim].
    ///
    /// One op on purpose: a backend may implement it fused (never
    /// materializing the score matrix), which composition would forbid.
    ///
    /// Contract limits (study-grade, enforced by every backend): all dims
    /// nonzero, and seq <= 4096 — the fused GPU kernel keeps the score
    /// row in threadgroup memory. Long-context support is future work
    /// and will change this contract explicitly. This method covers
    /// single-position decode; prefill and non-GQA cache layouts (MLA)
    /// arrive as separate methods when their chapters come.
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor;

    /// Index of the largest value; ties take the lowest index, NaN and
    /// -inf never win. Greedy sampling.
    fn argmax(&self, x: &Tensor) -> u32;
}
