//! `from_yamf` — move the loader's Yamf tensors into the family's
//! named struct fields, and validate the shapes at this border.
//!
//! Book 9.11: the HashMap lookup happens once at build, never
//! during generation. Yamf's fields are public, so a hand-built
//! bundle can skip the loader gate; from_yamf's own checks are
//! the module's door — presence, shape (per-tensor against the
//! config's derived dims), and no leftover tensors.

use nucleon_mlx::{shape_usize, Array};

use crate::families::qwen3::error::FamilyError;
use crate::loader::{FamilyConfig, Qwen3Config, Yamf};

/// One transformer block's 11 tensors, held by name.
pub(super) struct Block {
    pub(super) attn_norm: Array,   // [hidden]
    pub(super) attn_q: Array,      // [q_rows, hidden]
    pub(super) attn_k: Array,      // [kv_rows, hidden]
    pub(super) attn_v: Array,      // [kv_rows, hidden]
    pub(super) attn_q_norm: Array, // [head_dim]
    pub(super) attn_k_norm: Array, // [head_dim]
    pub(super) attn_output: Array, // [hidden, q_rows]
    pub(super) ffn_norm: Array,    // [hidden]
    pub(super) ffn_gate: Array,    // [ffn_inner, hidden]
    pub(super) ffn_up: Array,      // [ffn_inner, hidden]
    pub(super) ffn_down: Array,    // [hidden, ffn_inner]
}

/// The Qwen3 family: config + all 310 tensors in named fields.
pub struct Qwen3 {
    pub(super) cfg: Qwen3Config,
    pub(super) embed: Array,           // token_embd.weight, [vocab, hidden]
    pub(super) blocks: Vec<Block>,     // num_hidden_layers of these
    pub(super) output_norm: Array,     // [hidden]
    pub(super) lm_head: Option<Array>, // Some if untied, None if tied to embed
}

impl Qwen3 {
    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }
}

/// Build the family from the loader's border bundle.
pub fn from_yamf(y: &Yamf) -> Result<Qwen3, FamilyError> {
    let FamilyConfig::Qwen3(cfg) = &y.family;
    let cfg = cfg.clone();

    let mut tensors = y.tensors.clone();

    let hidden = cfg.hidden_size as usize;
    let head_dim = cfg.head_dim as usize;
    let ffn_inner = cfg.intermediate_size as usize;
    let vocab = cfg.vocab_size as usize;
    let q_rows = (cfg.num_attention_heads * cfg.head_dim) as usize;
    let kv_rows = (cfg.num_key_value_heads * cfg.head_dim) as usize;

    let take = |m: &mut std::collections::HashMap<String, Array>,
                name: &str,
                want: &[usize]|
     -> Result<Array, FamilyError> {
        let arr = m.remove(name).ok_or_else(|| FamilyError::Build {
            reason: format!("required tensor \"{name}\" absent from Yamf"),
        })?;
        let got = shape_usize(&arr);
        if got != want {
            return Err(FamilyError::Build {
                reason: format!("tensor \"{name}\" shape {got:?}, expected {want:?}"),
            });
        }
        Ok(arr)
    };

    let embed = take(&mut tensors, "token_embd.weight", &[vocab, hidden])?;
    let output_norm = take(&mut tensors, "output_norm.weight", &[hidden])?;

    let mut blocks = Vec::with_capacity(cfg.num_hidden_layers as usize);
    for n in 0..cfg.num_hidden_layers {
        let name = |suffix: &str| format!("blk.{n}.{suffix}.weight");
        blocks.push(Block {
            attn_norm: take(&mut tensors, &name("attn_norm"), &[hidden])?,
            attn_q: take(&mut tensors, &name("attn_q"), &[q_rows, hidden])?,
            attn_k: take(&mut tensors, &name("attn_k"), &[kv_rows, hidden])?,
            attn_v: take(&mut tensors, &name("attn_v"), &[kv_rows, hidden])?,
            attn_q_norm: take(&mut tensors, &name("attn_q_norm"), &[head_dim])?,
            attn_k_norm: take(&mut tensors, &name("attn_k_norm"), &[head_dim])?,
            attn_output: take(&mut tensors, &name("attn_output"), &[hidden, q_rows])?,
            ffn_norm: take(&mut tensors, &name("ffn_norm"), &[hidden])?,
            ffn_gate: take(&mut tensors, &name("ffn_gate"), &[ffn_inner, hidden])?,
            ffn_up: take(&mut tensors, &name("ffn_up"), &[ffn_inner, hidden])?,
            ffn_down: take(&mut tensors, &name("ffn_down"), &[hidden, ffn_inner])?,
        });
    }

    // output.weight optional: present when the size unties the head,
    // absent for Qwen3-0.6B (tied). Shape [vocab, hidden] same as
    // token_embd's — we only accept the exact expected shape here.
    let lm_head = match tensors.remove("output.weight") {
        Some(a) => {
            let got = shape_usize(&a);
            if got != [vocab, hidden] {
                return Err(FamilyError::Build {
                    reason: format!(
                        "tensor \"output.weight\" shape {got:?}, expected {:?}",
                        [vocab, hidden]
                    ),
                });
            }
            Some(a)
        }
        None => None,
    };

    if !tensors.is_empty() {
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        let shown: Vec<&str> = names.iter().take(3).map(|s| s.as_str()).collect();
        return Err(FamilyError::Build {
            reason: format!(
                "{} leftover tensors after building blocks (first: {:?})",
                tensors.len(),
                shown
            ),
        });
    }

    Ok(Qwen3 {
        cfg,
        embed,
        blocks,
        output_norm,
        lm_head,
    })
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;
