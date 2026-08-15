//! The gate and the Yamf — book chapters 7.7 and 7.8.
//!
//! `load` runs once, at startup. Exactly two things come out: a
//! complete `Yamf` (Yet Another Model Format — ironically not a
//! format: memory only, never serialized), or a refusal naming what
//! was expected, what the file held, and why that cannot be
//! supported. Nothing GGUF-shaped crosses this border.

use std::collections::HashMap;
use std::path::Path;

use crate::loader::config::{family_config, get_str, get_u32, FamilyConfig, Qwen3Config};
use crate::loader::container::{parse, Container, MetaValue};
use crate::loader::dequant::{dequantize, tensor_byte_len};
use crate::loader::error::LoaderError;
use crate::tensor::Tensor;

/// Token classes from the GGUF spec's token_type array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,      // 1
    Unknown,     // 2
    Control,     // 3
    UserDefined, // 4
    Unused,      // 5
    Byte,        // 6
}

impl TokenType {
    fn from_i32(v: i32) -> Result<Self, LoaderError> {
        Ok(match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            other => {
                return Err(LoaderError::Structure {
                    reason: format!("token_type value {other} is outside the spec's 1..=6"),
                })
            }
        })
    }
}

/// Everything the tokenizer block needs, as plain vectors. The shape
/// is byte-level BPE generic; the values are the family's. `pre`
/// names which pre-tokenizer regex to build; `stop_token_ids` is the
/// full stop set the gate assembled (the metadata under-reports it).
pub struct TokenizerData {
    pub tokens: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub token_types: Vec<TokenType>,
    pub pre: String,
    pub stop_token_ids: Vec<u32>,
}

/// The chat template, parsed and validated at the gate so a broken
/// template fails at load, not at first chat. Wraps a minijinja
/// environment owning the compiled template.
pub struct ChatTemplate {
    // consumed by the chat module's chapter (rendering); until then
    // its only job is having parsed successfully at the gate
    #[allow(dead_code)]
    env: minijinja::Environment<'static>,
    source: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, LoaderError> {
        let mut env = minijinja::Environment::new();
        env.add_template_owned("chat".to_string(), source.clone())
            .map_err(|e| LoaderError::Template {
                reason: e.to_string(),
            })?;
        Ok(Self { env, source })
    }

    /// The template's source text (rendering arrives with the chat
    /// module's chapter, through [`Self::env`]).
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// The border bundle: everything the engine needs, nothing the file
/// wanted to say beyond it.
pub struct Yamf {
    pub family: FamilyConfig,
    pub tensors: HashMap<String, Tensor>,
    pub tokenizer: TokenizerData,
    pub chat_template: ChatTemplate,
}

/// The tensor set a qwen3 checkpoint must carry: 2 global + 11 per
/// layer, shapes in OUR row-major order (ne dims reversed).
/// `output.weight` is the one optional extra: present when a size
/// unties its lm head, absent (tied) in Qwen3-0.6B.
fn expected_tensors(cfg: &Qwen3Config) -> HashMap<String, Vec<usize>> {
    let hidden = cfg.hidden_size as usize;
    let ffn = cfg.intermediate_size as usize;
    let q_rows = (cfg.num_attention_heads * cfg.head_dim) as usize;
    let kv_rows = (cfg.num_key_value_heads * cfg.head_dim) as usize;
    let head_dim = cfg.head_dim as usize;
    let vocab = cfg.vocab_size as usize;

    let mut m = HashMap::new();
    m.insert("token_embd.weight".to_string(), vec![vocab, hidden]);
    m.insert("output_norm.weight".to_string(), vec![hidden]);
    for n in 0..cfg.num_hidden_layers {
        let mut put = |suffix: &str, shape: Vec<usize>| {
            m.insert(format!("blk.{n}.{suffix}.weight"), shape);
        };
        put("attn_norm", vec![hidden]);
        put("attn_q", vec![q_rows, hidden]);
        put("attn_k", vec![kv_rows, hidden]);
        put("attn_v", vec![kv_rows, hidden]);
        put("attn_q_norm", vec![head_dim]);
        put("attn_k_norm", vec![head_dim]);
        put("attn_output", vec![hidden, q_rows]);
        put("ffn_norm", vec![hidden]);
        put("ffn_gate", vec![ffn, hidden]);
        put("ffn_up", vec![ffn, hidden]);
        put("ffn_down", vec![hidden, ffn]);
    }
    m
}

/// Load a GGUF file through the gate. See the book's chapter 7 for
/// the full contract this implements.
///
/// The file is read into memory (about 640 MB for Qwen3-0.6B, and
/// every byte is converted during dequant anyway). mmap returns in
/// Part III, when packed weights are computed from directly and its
/// unsafe contract is worth confronting.
pub fn load(path: &Path) -> Result<Yamf, LoaderError> {
    let bytes = std::fs::read(path)?;
    load_bytes(&bytes)
}

/// The whole gate over in-memory bytes; `load` adds only the file
/// read. Tests feed synthetic files through this seam.
///
/// Passes ordered cheap to expensive: metadata and template checks,
/// then the tensor contract (shapes, ranges, overlap), then dequant.
/// A refusal never costs a gigabyte of allocation first.
pub fn load_bytes(bytes: &[u8]) -> Result<Yamf, LoaderError> {
    let container = parse(bytes)?;
    let family = family_config(&container)?;
    let FamilyConfig::Qwen3(cfg) = &family;

    // The spec requires general.quantization_version whenever any
    // tensor is QUANTIZED (plain non-f32 floats like F16 are not);
    // the block layouts it names are the ones dequant.rs implements
    // (version 2 today). Unsupported types still refuse as
    // UnsupportedTensorType, not as a missing version key.
    if container
        .tensors
        .iter()
        .any(|t| t.type_id == crate::loader::dequant::GGML_Q8_0)
    {
        let qv = get_u32(&container, "general.quantization_version")?;
        if qv != 2 {
            return Err(LoaderError::Structure {
                reason: format!("quantization_version {qv}; nucleon supports 2"),
            });
        }
    }

    // ---- pass 1: cheapest checks first — tokenizer metadata and
    // the template need no allocation worth naming.
    // The gate refuses tokenizer kinds the tokenizer block cannot
    // build: byte-level BPE ("gpt2") with the qwen2 pre-tokenizer.
    let model = get_str(&container, "tokenizer.ggml.model")?;
    if model != "gpt2" {
        return Err(LoaderError::Structure {
            reason: format!("tokenizer model \"{model}\"; nucleon supports: gpt2 (byte-level BPE)"),
        });
    }
    let pre = get_str(&container, "tokenizer.ggml.pre")?.to_string();
    if pre != "qwen2" {
        return Err(LoaderError::Structure {
            reason: format!("pre-tokenizer \"{pre}\"; nucleon supports: qwen2"),
        });
    }
    let eos = get_u32(&container, "tokenizer.ggml.eos_token_id")?;
    // the BOS landmine (book 7.3): qwen3 never prepends BOS. A file
    // claiming add_bos_token = true is a broken conversion; honoring
    // it silently is worse than refusing it loudly.
    match container.metadata.get("tokenizer.ggml.add_bos_token") {
        None | Some(MetaValue::Bool(false)) => {}
        Some(MetaValue::Bool(true)) => {
            return Err(LoaderError::Structure {
                reason: "add_bos_token is true; qwen3 never prepends BOS".to_string(),
            })
        }
        Some(other) => {
            return Err(LoaderError::WrongType {
                key: "tokenizer.ggml.add_bos_token".to_string(),
                want: "bool",
                found: other.kind(),
            })
        }
    }
    let template_src = get_str(&container, "tokenizer.chat_template")?.to_string();
    let chat_template = ChatTemplate::new(template_src)?;

    let mut container = container; // consume the arrays without cloning
    let tokens = take_str_array(&mut container, "tokenizer.ggml.tokens")?;
    if u64::from(eos) >= tokens.len() as u64 {
        return Err(LoaderError::Structure {
            reason: format!(
                "eos_token_id {eos} is outside the vocab of {}",
                tokens.len()
            ),
        });
    }
    let type_ints = take_i32_array(&mut container, "tokenizer.ggml.token_type")?;
    if type_ints.len() != tokens.len() {
        return Err(LoaderError::Structure {
            reason: format!(
                "token_type has {} entries, tokens has {}",
                type_ints.len(),
                tokens.len()
            ),
        });
    }
    let mut token_types = Vec::with_capacity(type_ints.len());
    for v in type_ints {
        token_types.push(TokenType::from_i32(v)?);
    }

    let merge_strs = take_str_array(&mut container, "tokenizer.ggml.merges")?;
    let mut merges = Vec::with_capacity(merge_strs.len());
    for m in merge_strs {
        // byte-level tokens encode real spaces as G-with-breve, so a
        // merge is exactly "left right": one space, both sides full.
        match m.split_once(' ') {
            Some((a, b)) if !a.is_empty() && !b.is_empty() && !b.contains(' ') => {
                merges.push((a.to_string(), b.to_string()))
            }
            _ => {
                return Err(LoaderError::Structure {
                    reason: format!("malformed merge entry \"{m}\""),
                })
            }
        }
    }

    // The metadata under-reports stopping (book 7.3, landmine 2):
    // assemble eos plus <|endoftext|> looked up by string.
    let mut stop_token_ids = vec![eos];
    if let Some(pos) = tokens.iter().position(|t| t == "<|endoftext|>") {
        let id = pos as u32;
        if id != eos {
            stop_token_ids.push(id);
        }
    }

    // ---- pass 2: the tensor contract, still no weight allocation.
    let mut expected = expected_tensors(cfg);
    let optional_output = vec![cfg.vocab_size as usize, cfg.hidden_size as usize];
    let region_len = container.file_len - container.data_start;
    let mut plan: Vec<(&crate::loader::container::TensorInfo, Vec<usize>, u64, u64)> = Vec::new();
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();

    for info in &container.tensors {
        // dims arrive in ne order (dims[0] contiguous); ours reverse.
        let mut shape: Vec<usize> = Vec::with_capacity(info.dims.len());
        for d in info.dims.iter().rev() {
            shape.push(usize::try_from(*d).map_err(|_| LoaderError::Structure {
                reason: format!("tensor \"{}\" dimension {d} overflows usize", info.name),
            })?);
        }
        let want = match expected.remove(&info.name) {
            Some(w) => w,
            None if info.name == "output.weight" => optional_output.clone(),
            None => {
                return Err(LoaderError::UnexpectedTensor {
                    name: info.name.clone(),
                })
            }
        };
        if shape != want {
            return Err(LoaderError::WrongShape {
                name: info.name.clone(),
                want,
                found: shape,
            });
        }

        let mut n_elems: u64 = 1;
        for d in &info.dims {
            n_elems = n_elems
                .checked_mul(*d)
                .ok_or_else(|| LoaderError::Structure {
                    reason: format!("tensor \"{}\" element count overflows", info.name),
                })?;
        }
        let byte_len = tensor_byte_len(info.type_id, n_elems, info.dims[0], &info.name)?;
        let end = info
            .offset
            .checked_add(byte_len)
            .ok_or_else(|| LoaderError::Structure {
                reason: format!("tensor \"{}\" range overflows", info.name),
            })?;
        if end > region_len as u64 {
            return Err(LoaderError::Structure {
                reason: format!(
                    "tensor \"{}\" ends at data byte {end}, region has {region_len}",
                    info.name
                ),
            });
        }
        ranges.push((info.offset, end, &info.name));
        plan.push((info, shape, n_elems, byte_len));
    }

    // structural lies (overlap) refuse before completeness does
    ranges.sort();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(LoaderError::Structure {
                reason: format!(
                    "tensors \"{}\" and \"{}\" overlap in the data region",
                    pair[0].2, pair[1].2
                ),
            });
        }
    }

    if let Some(name) = expected.keys().next() {
        return Err(LoaderError::MissingTensor { name: name.clone() });
    }

    // ---- pass 3: everything validated; now materialize the weights.
    let mut tensors = HashMap::new();
    for (info, shape, n_elems, byte_len) in plan {
        let start = container.data_start + info.offset as usize;
        let data = dequantize(
            info.type_id,
            &bytes[start..start + byte_len as usize],
            n_elems as usize,
            info.dims[0],
            &info.name,
        )?;
        tensors.insert(info.name.clone(), Tensor::new(shape, data));
    }

    Ok(Yamf {
        family,
        tensors,
        tokenizer: TokenizerData {
            tokens,
            merges,
            token_types,
            pre,
            stop_token_ids,
        },
        chat_template,
    })
}

fn take_str_array(c: &mut Container, key: &'static str) -> Result<Vec<String>, LoaderError> {
    match c.metadata.remove(key) {
        None => Err(LoaderError::MissingKey { key }),
        Some(MetaValue::Array {
            elem_type_id,
            items,
        }) => {
            // the declared element type must be string (8) even when
            // the array is empty
            if elem_type_id != 8 {
                return Err(LoaderError::WrongType {
                    key: key.to_string(),
                    want: "array of strings",
                    found: "array of another type",
                });
            }
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    MetaValue::Str(s) => out.push(s),
                    other => {
                        return Err(LoaderError::WrongType {
                            key: key.to_string(),
                            want: "array of strings",
                            found: other.kind(),
                        })
                    }
                }
            }
            Ok(out)
        }
        Some(other) => Err(LoaderError::WrongType {
            key: key.to_string(),
            want: "array",
            found: other.kind(),
        }),
    }
}

fn take_i32_array(c: &mut Container, key: &'static str) -> Result<Vec<i32>, LoaderError> {
    match c.metadata.remove(key) {
        None => Err(LoaderError::MissingKey { key }),
        Some(MetaValue::Array {
            elem_type_id,
            items,
        }) => {
            if elem_type_id != 5 {
                return Err(LoaderError::WrongType {
                    key: key.to_string(),
                    want: "array of i32",
                    found: "array of another type",
                });
            }
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    MetaValue::I32(v) => out.push(v),
                    other => {
                        return Err(LoaderError::WrongType {
                            key: key.to_string(),
                            want: "array of i32",
                            found: other.kind(),
                        })
                    }
                }
            }
            Ok(out)
        }
        Some(other) => Err(LoaderError::WrongType {
            key: key.to_string(),
            want: "array",
            found: other.kind(),
        }),
    }
}

#[cfg(test)]
#[path = "yamf_tests.rs"]
mod tests;
