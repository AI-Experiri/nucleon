# The Fast Kernels

> Placeholder: written when this chapter is built.

Planned content, seeded by questions from the build:

- Tiled matmul: threadgroup-memory blocking, then simdgroup_matrix ops.
- FlashAttention on Metal: online softmax (running max and sum) over
  K/V tiles in threadgroup memory — removes our attention kernel's
  seq <= 4096 cap, which exists only because it materializes the whole
  score row on-core.
- Benchmarks against llama.cpp and MLX at every step.
