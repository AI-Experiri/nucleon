# ADR 005: compute via mlx-rs; nucleon-metal removed

Date: 2026-08-15. Status: accepted (user decision).
Supersedes ADR 004 (pure-Rust rule).

## Decision

nucleon's compute layer is Apple's MLX, accessed from Rust through
the community [`mlx-rs`](https://github.com/oxideai/mlx-rs) crate.
The pure-Rust rule set by ADR 004 is dropped: FFI is now allowed for
compute.

Consequences on the trunk, applied in this commit series:

1. `nucleon-metal/` is removed from the workspace entirely (device
   layer, `MetalOps`, `MetalBackend`, parity tests, fusion example).
2. `nucleon/src/backend/` is removed (the `Backend` trait, `CpuBackend`
   and its tests). Compute now lives in MLX.
3. `nucleon-metal/kernels/ops.metal` is preserved as book teaching
   reference at `docs/book/src/reference/kernels/ops.metal`.
4. A new sibling crate `nucleon-mlx/` is added as nucleon's thin
   wrapper over `mlx-rs`. All uses of MLX inside the engine flow
   through this crate so that a `mlx-rs` API break has exactly one
   file to repair. `nucleon-mlx` also exposes the pinned version
   triple at runtime (`mlx_rs_version()`, `mlx_c_version()`,
   `mlx_version()`, `version_triple()`).
5. `mlx-rs = "=0.25.3"` is exact-pinned in the workspace. That pin
   transitively picks `mlx-c 0.5.0` and Apple MLX `0.30.6` via
   `mlx-rs`'s git submodule and `mlx-c`'s CMake `FetchContent`.
   Whatever's installed via brew on the host is not used; the MLX we
   run is baked into our binary at build time (static link).
6. `Yamf.tensors` remains `HashMap<String, nucleon::tensor::Tensor>`
   for this commit. Migrating the tensor type to
   `nucleon_mlx::Array` happens with the family forward pass block,
   so this pivot does not have to rewrite the 115 passing loader
   tests in the same change.

## Why

MLX already implements the fused kernels the Metal chapter said we
would eventually need (tiled matmul, FlashAttention-style SDPA,
fused RMSNorm, fused RoPE, and lazy-graph fusion of arbitrary chains
via `compile`). Rebuilding them ourselves at production quality is
weeks of work per kernel; MLX is engineered and maintained by Apple.
The book still teaches what a kernel is and what composed-vs-fused
costs (the Metal chapter is preserved); the mature versions of the
same ideas move to Part III, taught as `mx.fast.metal_kernel` shaders
we write on top of MLX's dispatch layer.

## Version cadence

`mlx-rs`, `mlx-c`, and Apple's `mlx` are all on 0.x version lines;
semver permits API breaks on every minor release. The pin/absorb
cadence:

- We track `mlx-rs` releases and bump the exact pin in the workspace
  `Cargo.toml` when we choose to.
- On every bump, `nucleon-mlx/src/lib.rs`'s three version constants
  are updated to match the new transitive triple.
- The book's MLX chapter's version table is updated in the same
  commit so what the book claims matches what the binary loads.

## What might reopen this

If oxideai's `mlx-rs` falls significantly behind Apple's MLX
(currently, a two-minor-version gap: system MLX 0.32 vs linked MLX
0.30.6), we consider one of:

1. Contribute upstream to `mlx-rs` to close the gap.
2. Fork `mlx-rs` at a commit that tracks a newer `mlx-c`.
3. Write our own Rust bindings directly over `mlx-c` (one fewer hop
   between us and Apple, `nucleon-mlx` becomes those bindings). The
   concrete trigger will likely be `mx.fast.metal_kernel` — Apple
   MLX exposes it, `mlx-rs 0.25.3` does not.

None of these is a decision today; recording the ladder so we do not
re-argue it later.
