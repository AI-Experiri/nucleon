//! `encode` — text to ids, never adding specials.
//!
//! Book chapter 8.8: the wrapper hides the HF boolean so no caller
//! can accidentally turn special-sprinkling back on. The chat
//! template owns every special in the string (8.6).

use crate::tokenizer::build::Tokenizer;
use crate::tokenizer::error::TokenizerError;

/// Encode refuses inputs past this many bytes. fancy-regex's
/// backtrack limit (1M) trips silently at ~1M whitespace characters
/// in one segment — the tokenizers crate swallows the error and the
/// whole input becomes ONE unsplit piece, producing wrong ids with
/// no failure. 512 KiB keeps every input far under the trip point
/// (bytes >= chars) while dwarfing any real prompt: the model's
/// whole 40k-token context is about 160 KB of text.
const MAX_ENCODE_BYTES: usize = 512 * 1024;

impl Tokenizer {
    /// Encode text into ids. Added tokens (the specials already in
    /// the string) match whole; everything else goes through the
    /// qwen2 split, the byte-level alphabet, and BPE.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if text.len() > MAX_ENCODE_BYTES {
            return Err(TokenizerError::Encode {
                reason: format!(
                    "input is {} bytes; encode caps at {MAX_ENCODE_BYTES} \
                     (the pre-tokenizer regex misbehaves silently past ~1M chars)",
                    text.len()
                ),
            });
        }
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(TokenizerError::encode)?;
        Ok(encoding.get_ids().to_vec())
    }
}

#[cfg(test)]
#[path = "encode_tests.rs"]
mod tests;
