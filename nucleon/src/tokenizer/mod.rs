//! The tokenizer: text to ids and back — book chapter 8.
//!
//! Glue around the HF `tokenizers` crate (chapter 8.11 maps each
//! part to its crate item). Only the border types are public:
//! `Tokenizer`, `DecodeStream`, `TokenizerError`, `from_yamf`.

mod build;
mod decode;
mod encode;
mod error;

pub use build::{from_yamf, Tokenizer};
pub use decode::DecodeStream;
pub use error::TokenizerError;
