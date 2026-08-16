//! GGUF v3 format adapter. Turns a GGUF file's bytes into the
//! format-agnostic `Yamf` that everything past the loader consumes.
//! Book chapter 7. Facts: `docs/research/gguf-qwen3.md`.

pub mod config;
pub mod container;
pub mod dequant;

#[cfg(test)]
pub(crate) mod builder;
