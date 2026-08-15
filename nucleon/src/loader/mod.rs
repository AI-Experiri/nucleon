//! The loader: the engine's gate. One GGUF file in, one `Yamf` out,
//! or a refusal naming exactly what could not be supported.
//! Book: docs/book/src/04-loader.md. Facts: docs/research/gguf-qwen3.md.

pub mod config;
pub mod container;
pub mod dequant;
pub mod error;
pub mod yamf;

#[cfg(test)]
pub(crate) mod gguf_builder;

pub use config::{FamilyConfig, Qwen3Config};
pub use error::LoaderError;
pub use yamf::{load, load_bytes, ChatTemplate, TokenType, TokenizerData, Yamf};
