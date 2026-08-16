# nucleon-mlx

The compute layer. A thin nucleon-owned wrapper over the community
[`mlx-rs`](https://github.com/oxideai/mlx-rs) crate (unofficial Rust
bindings for Apple's MLX). Every use of MLX inside the engine flows
through this crate, so one file is where we notice — and repair — any
`mlx-rs` API break on a version bump.

## Pinned version triple

Baked into the binary at build time (static link). The workspace
`Cargo.toml` exact-pins `mlx-rs`; that pin transitively picks the
mlx-c and MLX versions through `mlx-rs`'s git submodule and CMake
`FetchContent`. Whatever's installed via `brew` on the host is not
used.

| layer | version |
|---|---|
| mlx-rs | 0.25.3 |
| mlx-c | 0.5.0 |
| Apple MLX (C++ core) | 0.30.6 |

Read at runtime with `nucleon_mlx::mlx_rs_version()`,
`mlx_c_version()`, `mlx_version()`, or the one-line
`version_triple()`. When we bump `mlx-rs`, we update the three
constants in `src/lib.rs` to match.

## What lives here

Today: version-inspection API, plus re-exports of the mlx-rs types
the rest of nucleon uses. As the Qwen3 forward pass lands, per-op
wrappers move here (`nucleon_mlx::rms_norm`, `matmul`, `rope`,
`scaled_dot_product_attention`, `silu`, ...), each thin enough that
if `mlx-rs`'s signature changes, only this crate updates.

The chapter for all this: `docs/book/src/03-mlx.md`.
