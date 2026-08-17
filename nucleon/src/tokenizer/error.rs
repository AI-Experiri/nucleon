//! `TokenizerError` — the three ways the tokenizer can fail.
//!
//! Same shape as `LoaderError`: each variant carries the reason,
//! Display is written by hand. Build covers both rejected inputs
//! (Yamf's fields are public, so from_yamf re-validates its border:
//! pre id, parallel arrays, duplicates, byte alphabet) and genuine
//! construction failures inside the HF crate; encode/decode
//! failures are runtime.

use std::fmt;

#[derive(Debug)]
pub enum TokenizerError {
    /// `from_yamf` could not assemble the HF tokenizer.
    Build { reason: String },
    /// `encode` failed inside the HF crate.
    Encode { reason: String },
    /// `decode` or a stream step failed inside the HF crate.
    Decode { reason: String },
}

impl TokenizerError {
    /// Wrap an HF-crate build failure. `impl Display` keeps the HF
    /// error type out of our signatures.
    pub(crate) fn build(e: impl fmt::Display) -> Self {
        TokenizerError::Build {
            reason: e.to_string(),
        }
    }

    pub(crate) fn encode(e: impl fmt::Display) -> Self {
        TokenizerError::Encode {
            reason: e.to_string(),
        }
    }

    pub(crate) fn decode(e: impl fmt::Display) -> Self {
        TokenizerError::Decode {
            reason: e.to_string(),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Build { reason } => {
                write!(f, "tokenizer build failed: {reason}")
            }
            TokenizerError::Encode { reason } => write!(f, "encode failed: {reason}"),
            TokenizerError::Decode { reason } => write!(f, "decode failed: {reason}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
