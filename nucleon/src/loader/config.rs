//! Metadata keys to `Qwen3Config` — book chapter 7.6.
//!
//! Typed values read by key name, no JSON anywhere. Missing key,
//! wrong type, or a foreign `general.architecture` each refuse with
//! a named error. The result reaches the engine wrapped in
//! `FamilyConfig`, the one family-specific corner of the Yamf.

use crate::loader::container::{Container, MetaValue};
use crate::loader::error::LoaderError;

/// The typed twin of the config keys the qwen3 family needs. One
/// field per GGUF metadata key; values for Qwen3-0.6B in comments.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub num_hidden_layers: u32,       // qwen3.block_count           = 28
    pub hidden_size: u32,             // qwen3.embedding_length      = 1024
    pub intermediate_size: u32,       // qwen3.feed_forward_length   = 3072
    pub num_attention_heads: u32,     // qwen3.attention.head_count  = 16
    pub num_key_value_heads: u32,     // ...head_count_kv            = 8
    pub head_dim: u32,                // ...key_length               = 128
    pub rms_norm_eps: f32,            // ...layer_norm_rms_epsilon   = 1e-6
    pub rope_theta: f32,              // qwen3.rope.freq_base        = 1e6
    pub max_position_embeddings: u32, // qwen3.context_length        = 40960
    pub vocab_size: u32,              // tokens array length         = 151936
}

/// The family tag: which family's numbers the Yamf carries. A second
/// family is a second variant; nothing else in the Yamf moves.
#[derive(Debug, Clone, PartialEq)]
pub enum FamilyConfig {
    Qwen3(Qwen3Config),
}

pub(crate) fn get<'c>(c: &'c Container, key: &'static str) -> Result<&'c MetaValue, LoaderError> {
    c.metadata.get(key).ok_or(LoaderError::MissingKey { key })
}

pub(crate) fn get_u32(c: &Container, key: &'static str) -> Result<u32, LoaderError> {
    match get(c, key)? {
        MetaValue::U32(v) => Ok(*v),
        // the spec's standardized count/length keys may be written as
        // u64; accept them when the value fits
        MetaValue::U64(v) => u32::try_from(*v).map_err(|_| LoaderError::Structure {
            reason: format!("key \"{key}\" value {v} exceeds u32"),
        }),
        other => Err(LoaderError::WrongType {
            key: key.to_string(),
            want: "u32",
            found: other.kind(),
        }),
    }
}

pub(crate) fn get_f32(c: &Container, key: &'static str) -> Result<f32, LoaderError> {
    match get(c, key)? {
        MetaValue::F32(v) => Ok(*v),
        other => Err(LoaderError::WrongType {
            key: key.to_string(),
            want: "f32",
            found: other.kind(),
        }),
    }
}

/// For keys the spec fixes at exactly u32 (quantization_version,
/// eos_token_id): no u64 leniency.
pub(crate) fn get_u32_exact(c: &Container, key: &'static str) -> Result<u32, LoaderError> {
    match get(c, key)? {
        MetaValue::U32(v) => Ok(*v),
        other => Err(LoaderError::WrongType {
            key: key.to_string(),
            want: "u32",
            found: other.kind(),
        }),
    }
}

pub(crate) fn get_str<'c>(c: &'c Container, key: &'static str) -> Result<&'c str, LoaderError> {
    match get(c, key)? {
        MetaValue::Str(v) => Ok(v),
        other => Err(LoaderError::WrongType {
            key: key.to_string(),
            want: "string",
            found: other.kind(),
        }),
    }
}

/// Read the family config out of the metadata. The architecture is
/// checked first: everything after assumes qwen3's key namespace.
pub fn family_config(c: &Container) -> Result<FamilyConfig, LoaderError> {
    let arch = get_str(c, "general.architecture")?;
    if arch != "qwen3" {
        // truncate the echo: a hostile value should not balloon the
        // error message
        let mut found: String = arch.chars().take(64).collect();
        if arch.chars().count() > 64 {
            found.push_str("...");
        }
        return Err(LoaderError::UnsupportedArchitecture { found });
    }

    // There is no vocab_size key; the tokens array's length is it.
    let vocab_size = match get(c, "tokenizer.ggml.tokens")? {
        MetaValue::Array { items, .. } => {
            u32::try_from(items.len()).map_err(|_| LoaderError::Structure {
                reason: format!("tokens array length {} exceeds u32", items.len()),
            })?
        }
        other => {
            return Err(LoaderError::WrongType {
                key: "tokenizer.ggml.tokens".to_string(),
                want: "array",
                found: other.kind(),
            })
        }
    };

    let cfg = Qwen3Config {
        num_hidden_layers: get_u32(c, "qwen3.block_count")?,
        hidden_size: get_u32(c, "qwen3.embedding_length")?,
        intermediate_size: get_u32(c, "qwen3.feed_forward_length")?,
        num_attention_heads: get_u32(c, "qwen3.attention.head_count")?,
        num_key_value_heads: get_u32(c, "qwen3.attention.head_count_kv")?,
        head_dim: get_u32(c, "qwen3.attention.key_length")?,
        rms_norm_eps: get_f32(c, "qwen3.attention.layer_norm_rms_epsilon")?,
        rope_theta: get_f32(c, "qwen3.rope.freq_base")?,
        max_position_embeddings: get_u32(c, "qwen3.context_length")?,
        vocab_size,
    };

    // The file also declares a value-head length. Our attention op
    // computes k and v at one head_dim, so the two must agree; a
    // family where they differ is refused, not misread.
    let value_length = get_u32(c, "qwen3.attention.value_length")?;
    if value_length != cfg.head_dim {
        return Err(LoaderError::Structure {
            reason: format!(
                "value_length {value_length} != key_length {}; nucleon's attention needs them equal",
                cfg.head_dim
            ),
        });
    }

    // File-driven numbers get sanity caps so later arithmetic
    // (expected shapes, cache sizing) cannot overflow or explode.
    if cfg.num_hidden_layers == 0 || cfg.num_hidden_layers > 10_000 {
        return Err(LoaderError::Structure {
            reason: format!("block_count {} is outside 1..=10000", cfg.num_hidden_layers),
        });
    }
    for (name, v) in [
        ("embedding_length", cfg.hidden_size),
        ("feed_forward_length", cfg.intermediate_size),
        ("head_count", cfg.num_attention_heads),
        ("head_count_kv", cfg.num_key_value_heads),
        ("key_length", cfg.head_dim),
    ] {
        if v == 0 || v > 1_000_000 {
            return Err(LoaderError::Structure {
                reason: format!("qwen3 {name} {v} is outside 1..=1000000"),
            });
        }
    }
    // the float knobs must be finite and positive, or every norm and
    // every rope angle downstream is poisoned
    for (name, v) in [
        ("layer_norm_rms_epsilon", cfg.rms_norm_eps),
        ("rope.freq_base", cfg.rope_theta),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return Err(LoaderError::Structure {
                reason: format!("qwen3 {name} {v} must be finite and positive"),
            });
        }
    }
    // context gets its own, wider range: 1M-token models are real
    if cfg.max_position_embeddings == 0 || cfg.max_position_embeddings > 100_000_000 {
        return Err(LoaderError::Structure {
            reason: format!(
                "context_length {} is outside 1..=100000000",
                cfg.max_position_embeddings
            ),
        });
    }
    if cfg.num_attention_heads.checked_mul(cfg.head_dim).is_none()
        || cfg.num_key_value_heads.checked_mul(cfg.head_dim).is_none()
    {
        return Err(LoaderError::Structure {
            reason: "head_count x head_dim overflows u32".to_string(),
        });
    }
    // RoPE needs even head_dim so pairs form; refuse at the gate,
    // not with a panic on first rope() call
    if !cfg.head_dim.is_multiple_of(2) {
        return Err(LoaderError::Structure {
            reason: format!("key_length {} must be even (RoPE pairs)", cfg.head_dim),
        });
    }
    if !cfg
        .num_attention_heads
        .is_multiple_of(cfg.num_key_value_heads)
    {
        return Err(LoaderError::Structure {
            reason: format!(
                "head_count {} does not divide evenly over head_count_kv {} (GQA)",
                cfg.num_attention_heads, cfg.num_key_value_heads
            ),
        });
    }

    Ok(FamilyConfig::Qwen3(cfg))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
