//! `from_yamf` — Yamf pieces to an in-memory HF tokenizer.
//!
//! Book chapter 8.7: every part is built from a field the loader
//! already validated — but Yamf's fields are public, so a
//! hand-built bundle can skip the gate. from_yamf therefore
//! re-validates its own border (parallel arrays, duplicate tokens
//! and duplicate merges, byte alphabet present and Normal-typed,
//! merge operands/products present in the vocab and Normal-typed,
//! pre id known) and refuses rather than build a silently-corrupting
//! tokenizer. These checks are the module's own door, not
//! redundancy. The crate items used here are the table in 8.11.

use tokenizers::models::bpe::{BpeBuilder, Vocab};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior};

use crate::loader::yamf::{TokenType, Yamf};
use crate::tokenizer::error::TokenizerError;

/// The qwen2 pre-tokenizer regex, verified against Qwen's
/// tokenizer.json and llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN2
/// (docs/research/gguf-qwen3.md section 7; book 8.5 explains each
/// alternative).
const QWEN2_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The tokenizer: parts 1-5 live inside the wrapped HF tokenizer,
/// part 7 (the stop set) is our field, part 6 (UTF-8 buffering) is
/// created per generation by `stream_decoder`.
pub struct Tokenizer {
    pub(super) inner: tokenizers::Tokenizer,
    pub(super) stops: Vec<u32>,
    /// `tokens.len()` at build time. Stored because the crate's
    /// `get_vocab_size` clones the whole vocab per call, and added
    /// tokens reuse vocab ids so the count never differs.
    vocab_len: usize,
}

impl Tokenizer {
    /// Part 7 verbatim: the stop ids the loader assembled.
    /// Comparing against them is the generation loop's job.
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stops
    }

    /// Vocab size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.vocab_len
    }
}

/// How one vocab entry becomes an added token. Split out so the
/// field choices are directly testable: only Control is special AS
/// GGUF TYPES IT (<think>/<tool_call> arrive UserDefined and stay
/// non-special, matching the reference; the fim/repo markers arrive
/// Control because llama.cpp's converter types any <|...|> shape as
/// control, so they diverge from the reference's special=false —
/// inert while every decode path keeps specials). normalized stays
/// false for both kinds, as in the reference.
fn added_token(content: &str, ty: TokenType) -> AddedToken {
    AddedToken::from(content.to_string(), matches!(ty, TokenType::Control)).normalized(false)
}

/// Build the tokenizer from the loader's border bundle.
pub fn from_yamf(y: &Yamf) -> Result<Tokenizer, TokenizerError> {
    // the loader gates this for real files, but Yamf's fields are
    // public: a hand-built bundle with a short token_types would
    // otherwise silently truncate the added-token zip below and
    // leave trailing specials unregistered (BPE-decomposed markers)
    if y.tokenizer.tokens.len() != y.tokenizer.token_types.len() {
        return Err(TokenizerError::Build {
            reason: format!(
                "{} tokens but {} token_types; the arrays must be parallel",
                y.tokenizer.tokens.len(),
                y.tokenizer.token_types.len()
            ),
        });
    }
    // part 4, vocab side: a token's id is its index in the list
    let vocab: Vocab = y
        .tokenizer
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    // same public-fields hazard as the parallel-array check above: a
    // duplicate token string would silently shadow the earlier id
    // (it vanishes from the map, decodes to "") and over-report
    // vocab_size. The loader gates this for real files.
    if vocab.len() != y.tokenizer.tokens.len() {
        return Err(TokenizerError::Build {
            reason: format!(
                "{} tokens but only {} distinct strings; duplicates shadow ids",
                y.tokenizer.tokens.len(),
                vocab.len()
            ),
        });
    }
    // and the worst silent-corruption case: BPE here has no unk
    // token, so a character missing from the vocab is DROPPED at
    // encode and its neighbors merge across the hole ("acb" with no
    // "c" encodes like "ab"). The byte-level alphabet is what makes
    // every input encodable; verify all 256 entries at this border,
    // not just in the GGUF gate. Presence is not enough: an alphabet
    // entry typed non-Normal would flow into the added-token filter
    // below and match whole ahead of BPE, corrupting every word
    // containing that byte.
    for b in 0..=255u8 {
        let ch = crate::loader::yamf::byte_level_char(b);
        let s: String = std::iter::once(ch).collect();
        let Some(&id) = vocab.get(&s) else {
            return Err(TokenizerError::Build {
                reason: format!(
                    "byte-level alphabet incomplete: byte 0x{b:02X} (char {ch:?}) has no vocab entry"
                ),
            });
        };
        if y.tokenizer.token_types[id as usize] != TokenType::Normal {
            return Err(TokenizerError::Build {
                reason: format!(
                    "byte-level alphabet entry for byte 0x{b:02X} (char {ch:?}) is not Normal-typed"
                ),
            });
        }
    }
    // merges get the same treatment. Three refusals per merge:
    // - a duplicate pair silently keeps the LAST rank inside the
    //   crate's merge map (rank corruption, wrong ids, no error);
    // - an operand or product missing from the vocab must refuse
    //   HERE: the crate's builder sizes a scratch buffer to the
    //   longest vocab key and writes the concatenation into it
    //   BEFORE checking the product exists, so a long-enough absent
    //   product is a slice-index panic, not an Err (reproduced
    //   against tokenizers 0.23.1, model.rs:264-270);
    // - an operand or product typed non-Normal becomes an added
    //   token, the piece it names can never form, and the merge
    //   silently never fires.
    let mut merge_seen = std::collections::HashSet::new();
    for (a, b) in &y.tokenizer.merges {
        if !merge_seen.insert((a.as_str(), b.as_str())) {
            let shown: String = format!("{a} {b}").chars().take(64).collect();
            return Err(TokenizerError::Build {
                reason: format!("duplicate merge \"{shown}\" would silently change its rank"),
            });
        }
        let product = format!("{a}{b}");
        for part in [a.as_str(), b.as_str(), product.as_str()] {
            let shown = || -> String { part.chars().take(64).collect() };
            let Some(&id) = vocab.get(part) else {
                return Err(TokenizerError::Build {
                    reason: format!("merge references token \"{}\" not in the vocab", shown()),
                });
            };
            if y.tokenizer.token_types[id as usize] != TokenType::Normal {
                return Err(TokenizerError::Build {
                    reason: format!("merge references non-Normal token \"{}\"", shown()),
                });
            }
        }
    }

    // part 4, merge side: file order preserved, so index = rank
    let merges: Vec<(String, String)> = y.tokenizer.merges.clone();

    let bpe = BpeBuilder::default()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(TokenizerError::build)?;
    let mut inner = tokenizers::Tokenizer::new(bpe);

    // Qwen3's reference tokenizer.json carries an NFC normalizer.
    // GGUF has no field for it, so it is pinned here like the regex:
    // without it, decomposed input (an e + combining acute, routine
    // on macOS) tokenizes to ids the model never saw in training.
    // Set before add_special_tokens below per the crate's guidance;
    // the ordering only bites for normalized(true) added tokens,
    // which added_token() never produces.
    inner
        .with_normalizer(Some(NFC))
        .map_err(TokenizerError::build)?;

    // part 2 then part 3, chained: the pre id selects the regex (the
    // loader gates the value, but the match here keeps this file
    // honest the day a second family's pre is accepted — a regex
    // nothing selects is a silent wrong-ids bug), then the
    // byte-level stage maps bytes to alphabet characters. ByteLevel's
    // own regex stays off — Split already ran.
    let pre_regex = match y.tokenizer.pre.as_str() {
        "qwen2" => QWEN2_PRE,
        other => {
            // truncate the echo: pre is caller-supplied and could be
            // arbitrarily long (same discipline as the loader's echo())
            let shown: String = other.chars().take(64).collect();
            return Err(TokenizerError::Build {
                reason: format!("pre-tokenizer id \"{shown}\"; this build supports: qwen2"),
            });
        }
    };
    let split = Split::new(
        SplitPattern::Regex(pre_regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(TokenizerError::build)?;
    let byte_level = ByteLevel::new(
        false, // add_prefix_space: the chat template owns the string
        false, // trim_offsets: offsets unused, keep them honest
        false, // use_regex: Split above already did the splitting
    );
    inner.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), byte_level.into()])));

    // part 5: the same ByteLevel type in its decoder role
    inner.with_decoder(Some(byte_level));

    // part 1: every Control/UserDefined entry becomes an added
    // token, matched whole before BPE ever sees the text. They are
    // already in the vocab, so registration reuses their ids. Only
    // Control entries are marked special (matching the reference
    // tokenizer.json, where <think>/<tool_call> are added but NOT
    // special — skip_special_tokens must never eat them); normalized
    // stays false for both kinds, as in the reference.
    let added: Vec<AddedToken> = y
        .tokenizer
        .tokens
        .iter()
        .zip(&y.tokenizer.token_types)
        .filter(|(_, ty)| {
            // llama.cpp's special-token cache is CONTROL | USER_DEFINED
            // | UNKNOWN (research doc section 7); Qwen3 has no Unknown
            // rows, but a vocab that carries an <unk> must match it
            // whole, not BPE-decompose it
            matches!(
                ty,
                TokenType::Control | TokenType::UserDefined | TokenType::Unknown
            )
        })
        .map(|(t, ty)| added_token(t, *ty))
        .collect();
    inner
        .add_special_tokens(added)
        .map_err(TokenizerError::build)?;

    Ok(Tokenizer {
        inner,
        stops: y.tokenizer.stop_token_ids.clone(),
        vocab_len: y.tokenizer.tokens.len(),
    })
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;
