//! nucleon — an LLM inference engine.
//!
//! Modules exist only once their chapter is built. So far:
//! [`tensor`] (data, still nucleon-owned for the loader border) and
//! [`loader`] (the gate: GGUF in, Yamf out). Coming per docs/plan.md:
//! tokenizer, cache, families, sampler, generate, chat.
//!
//! Compute lives in the `nucleon-mlx` sibling crate (thin wrapper
//! over `mlx-rs`); the `Backend` trait + CpuBackend + hand-written
//! Metal kernels this crate used to carry were removed in ADR 005.
//!
//! Rule of the house: a module depends only on modules below it, and every
//! block is testable in isolation. See `docs/book/00-big-picture.md`.

// Layout convention: each block is a folder (chapter) whose main file bears
// the block's name, next to its sibling _tests.rs. Clippy's module_inception
// lint dislikes tensor/tensor.rs; the names are deliberate.
#![allow(clippy::module_inception)]

pub mod loader;
pub mod tensor;
