//! nucleon — an LLM inference engine.
//!
//! Modules exist only once their chapter is built. So far: [`loader`]
//! (the gate: GGUF in, Yamf out) and [`tokenizer`] (text to ids and
//! back, glue over the HF tokenizers crate). Coming per docs/plan.md:
//! cache, families, sampler, generate, chat.
//!
//! Compute lives in the `nucleon-mlx` sibling crate (thin wrapper
//! over `mlx-rs`); values crossing the loader border are
//! `nucleon_mlx::Array`, not a nucleon-owned type. See ADR 005.
//!
//! Rule of the house: a module depends only on modules below it,
//! and every block is testable in isolation. See
//! `docs/book/00-big-picture.md`.

pub mod loader;
pub mod tokenizer;
