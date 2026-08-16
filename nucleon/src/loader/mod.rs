//! The loader: the engine's gate. One weights file in, one `Yamf`
//! out, or a refusal naming exactly what could not be supported.
//! Book: docs/book/src/04-loader.md. Facts:
//! docs/research/gguf-qwen3.md.
//!
//! Format-agnostic core here (`yamf`, `error`); the actual byte
//! parsers live under `formats/`, one submodule per format. Today
//! that's `formats::gguf`; a hypothetical second format lands as a
//! sibling and this module dispatches to it.

pub mod error;
pub mod formats;
pub mod yamf;

pub use error::LoaderError;
pub use formats::gguf::config::{FamilyConfig, Qwen3Config};
pub use yamf::{load, load_bytes, ChatTemplate, TokenType, TokenizerData, Yamf};
