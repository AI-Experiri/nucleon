//! nucleon — a pure-Rust LLM inference engine.
//!
//! Modules exist only once their chapter is built. So far:
//! [`tensor`] (data) and [`backend`] (the op seam, CPU + GPU impls).
//! Coming per docs/plan.md: loader, tokenizer, cache, families,
//! sampler, generate, chat.
//!
//! Rule of the house: a module depends only on modules below it, and every
//! block is testable in isolation. See `docs/book/00-big-picture.md`.

// Layout convention: each block is a folder (chapter) whose main file bears
// the block's name, next to its sibling _tests.rs. Clippy's module_inception
// lint dislikes tensor/tensor.rs; the names are deliberate.
#![allow(clippy::module_inception)]

pub mod backend;
pub mod tensor;
