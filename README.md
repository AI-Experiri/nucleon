# nucleon

A pure-Rust LLM inference engine for Apple Silicon, built from scratch as
a learning project, documented as a book while it grows.

No llama.cpp, no MLX, no FFI to any C or C++ library. The only non-Rust
files are the Metal shaders in `nucleon-metal/kernels/`.

## State

Built and tested so far:

- `nucleon/src/tensor/`: the data structure (row-major, contiguous, f32),
  split into core, access, and display files.
- `nucleon/src/backend/`: the `Backend` trait (11 ops, attention as one
  fusable method) and `CpuBackend`, the naive reference implementation.
- `nucleon-metal/`: every op as a Metal kernel (map ops, tree-reduction
  ops, a fused rmsnorm+matvec, fused single-position GQA attention), the
  objc2-metal plumbing, and `MetalBackend` implementing the same trait.
- 72 tests, including trait-level CPU-vs-GPU parity for every op.

Measured on an M3 Max: batching kernel launches into one command buffer
is worth 5.8x at decode shapes; fusing rmsnorm into matvec adds 1.15x.

## Layout

```
nucleon/            core crate (no GPU code)
  src/tensor/       Tensor: shape + data
  src/backend/      Backend trait + CpuBackend
nucleon-metal/      Apple Silicon backend
  kernels/          *.metal, the only non-Rust files
  src/              device plumbing, op wrappers, MetalBackend
docs/book/          the book (mdBook): mdbook serve docs/book --open
docs/plan.md        the 14-step build plan
docs/decisions/     ADRs
journal/            session-by-session build log, wrong turns included
scripts/quality.sh  fmt + clippy -D warnings + full test suite
```

Modules for the loader, tokenizer, cache, model families, sampler, and
generation loop are created when their chapters are built; the plan is
in `docs/plan.md`.

## Working on it

```bash
./scripts/quality.sh          # the gate every change must pass
cargo test --workspace        # tests only
mdbook serve docs/book --open # read the book with live reload
cargo run --release -p nucleon-metal --example fusion  # the benchmark
```

Metal tests need an Apple Silicon machine; without one they skip
visibly.

## Milestones

| tag | deliverable |
|-----|-------------|
| m1-cpu-hello | Qwen3-0.6B generates real tokens on CPU |
| m2-metal-parity | the same on the GPU, fast |
| m3-quant | GGUF loading and quantized inference |
| m4-deepseek | DeepSeek-V2-Lite: MLA attention and MoE |
