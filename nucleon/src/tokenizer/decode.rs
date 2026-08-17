//! `decode` and `stream_decoder` — ids back to text.
//!
//! Book chapter 8.9: whole-sequence decode for tests and batch
//! use; the stream decoder for generation, where a multi-byte
//! codepoint split across two token ids must not print as U+FFFD.

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;

use crate::tokenizer::build::Tokenizer;
use crate::tokenizer::error::TokenizerError;

/// Part 6: wraps the HF `DecodeStream`, which buffers bytes until
/// they form a whole UTF-8 codepoint. One per generation; borrow
/// ties it to its tokenizer. must_use: dropping the stream without
/// calling `finish` loses a tail truncated mid-codepoint.
#[must_use = "call finish() when generation ends, or a truncated tail is lost"]
pub struct DecodeStream<'t> {
    tokenizer: &'t Tokenizer,
    inner: tokenizers::tokenizer::DecodeStream<
        't,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    /// Ids fed since the last emitted chunk. The HF stream has no
    /// flush, so a generation that stops mid-codepoint would lose
    /// these silently; `finish` decodes them lossily instead.
    pending: Vec<u32>,
}

impl DecodeStream<'_> {
    /// Feed one sampled id. `None` while bytes are buffered waiting
    /// for a codepoint to complete; `Some(chunk)` when clean text
    /// is ready to print. Caveat: `None` does not always mean
    /// mid-codepoint — a literal U+FFFD in the generated text stalls
    /// one token behind (the HF stream uses it as its own sentinel);
    /// `finish` drains whatever remains either way.
    pub fn step(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        let out = self.inner.step(id).map_err(TokenizerError::decode)?;
        match out {
            Some(chunk) => {
                self.pending.clear();
                Ok(Some(chunk))
            }
            None => {
                self.pending.push(id);
                Ok(None)
            }
        }
    }

    /// Drain whatever the stream still buffers when generation ends.
    /// A codepoint truncated by a max-token stop decodes lossily
    /// (U+FFFD) — visible truncation, never silent loss. `None` when
    /// nothing was buffered.
    pub fn finish(self) -> Result<Option<String>, TokenizerError> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let tail = self.tokenizer.decode(&self.pending)?;
        Ok(if tail.is_empty() { None } else { Some(tail) })
    }
}

impl Tokenizer {
    /// Decode a whole id sequence. Specials are kept (the false
    /// below), so ChatML markers survive round-trips.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(TokenizerError::decode)
    }

    /// A fresh stream decoder for one generation. Same keep-specials
    /// flag as `decode`.
    pub fn stream_decoder(&self) -> DecodeStream<'_> {
        DecodeStream {
            tokenizer: self,
            inner: self.inner.decode_stream(false),
            pending: Vec::new(),
        }
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
