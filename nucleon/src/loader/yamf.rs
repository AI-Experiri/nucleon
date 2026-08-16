//! The gate and the Yamf — book chapters 7.7 and 7.8.
//!
//! `load` runs once, at startup. Exactly two things come out: a
//! complete `Yamf` (Yet Another Model Format — ironically not a
//! format: memory only, never serialized), or a refusal naming what
//! was expected, what the file held, and why that cannot be
//! supported. Nothing GGUF-shaped crosses this border.

use std::collections::HashMap;
use std::path::Path;

use crate::loader::config::{family_config, get_str, get_u32_exact, FamilyConfig, Qwen3Config};

/// Truncate an echoed untrusted string in error messages.
fn echo(s: &str) -> String {
    const MAX: usize = 64;
    if s.chars().count() > MAX {
        let mut out: String = s.chars().take(MAX).collect();
        out.push_str("...");
        out
    } else {
        s.to_string()
    }
}
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
    env: minijinja::Environment<'static>,
    source: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, LoaderError> {
        let mut env = minijinja::Environment::new();
        // the real Qwen3 template calls tojson (minijinja's "json"
        // feature) and Python-style string methods like startswith
        // and strip; pycompat supplies the latter. Without these the
        // gate would bless a template that dies at first render.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat".to_string(), source.clone())
            .map_err(|e| LoaderError::Template {
                reason: e.to_string(),
            })?;
        Ok(Self { env, source })
    }

    /// The compiled environment, for the chat module's renderer.
    /// Crate-internal: minijinja types never cross the crate API.
    #[allow(dead_code)] // consumed when the chat chapter lands
    pub(crate) fn environment(&self) -> &minijinja::Environment<'static> {
        &self.env
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
    // refuse FIFOs, devices, and directories: opening a FIFO with no
    // writer blocks forever, so the kind is checked BEFORE open; the
    // post-open fstat re-check closes the swap race for everything a
    // plain open survives (a FIFO swapped in between the two calls
    // can still block the open; full immunity needs O_NONBLOCK,
    // which std does not expose).
    use std::io::Read;
    if !std::fs::metadata(path)?.is_file() {
        return Err(LoaderError::Structure {
            reason: format!("{} is not a regular file", path.display()),
        });
    }
    let file = std::fs::File::open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(LoaderError::Structure {
            reason: format!("{} is not a regular file", path.display()),
        });
    }
    // 100 GB caps everything the plan targets (Qwen3.8-27B GGUF at
    // f16 is 56 GB); the metadata length is only a hint. The read
    // itself is capped so a file growing between check and read
    // cannot escape.
    const MAX_FILE_BYTES: u64 = 100 * 1024 * 1024 * 1024;
    if meta.len() > MAX_FILE_BYTES {
        return Err(LoaderError::Structure {
            reason: format!(
                "{} is {} bytes; nucleon caps loads at {MAX_FILE_BYTES}",
                path.display(),
                meta.len()
            ),
        });
    }
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    let mut capped = file.take(MAX_FILE_BYTES + 1);
    capped.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(LoaderError::Structure {
            reason: format!(
                "{} grew past {MAX_FILE_BYTES} bytes during load",
                path.display()
            ),
        });
    }
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
        .any(|t| crate::loader::dequant::is_quantized(t.type_id))
    {
        let qv = get_u32_exact(&container, "general.quantization_version")?;
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
            reason: format!(
                "tokenizer model \"{}\"; nucleon supports: gpt2 (byte-level BPE)",
                echo(model)
            ),
        });
    }
    let pre = get_str(&container, "tokenizer.ggml.pre")?.to_string();
    if pre != "qwen2" {
        return Err(LoaderError::Structure {
            reason: format!("pre-tokenizer \"{}\"; nucleon supports: qwen2", echo(&pre)),
        });
    }
    let eos = get_u32_exact(&container, "tokenizer.ggml.eos_token_id")?;
    // the BOS landmine (book 7.3): qwen3 never prepends BOS. A file
    // claiming add_bos_token = true is a broken conversion; honoring
    // it silently is worse than refusing it loudly.
    for (key, human) in [
        ("tokenizer.ggml.add_bos_token", "BOS"),
        ("tokenizer.ggml.add_eos_token", "EOS"),
    ] {
        match container.metadata.get(key) {
            None | Some(MetaValue::Bool(false)) => {}
            Some(MetaValue::Bool(true)) => {
                return Err(LoaderError::Structure {
                    reason: format!(
                    "{key} is true; qwen3 relies on the chat template, not tokenizer auto-{human}"
                ),
                })
            }
            Some(other) => {
                return Err(LoaderError::WrongType {
                    key: key.to_string(),
                    want: "bool",
                    found: other.kind(),
                })
            }
        }
    }
    let template_src = get_str(&container, "tokenizer.chat_template")?.to_string();
    let template_source_ref = template_src.clone();
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
    // BPE needs a bijective vocab: duplicates make ids ambiguous
    let mut vocab_set = std::collections::HashSet::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        if !vocab_set.insert(t.as_str()) {
            return Err(LoaderError::Structure {
                reason: format!("duplicate token at id {i}"),
            });
        }
    }

    // byte-level BPE encodes raw bytes into 256 base tokens (HF
    // ByteLevel::alphabet); if any is missing, ordinary bytes cannot
    // be encoded at all. tokenizers has no re-export for the
    // alphabet, so we build it the same way (bytes not in
    // !..~ / ¡..¬ / ®..ÿ shift by +256)
    for b in 0u16..256 {
        let c = byte_level_char(b as u8);
        let s: String = std::iter::once(c).collect();
        if !vocab_set.contains(s.as_str()) {
            return Err(LoaderError::Structure {
                reason: format!(
                    "byte-level base token for byte 0x{:02x} (\"{s}\") is missing from the vocab",
                    b
                ),
            });
        }
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
    let vocab_set: std::collections::HashSet<&str> = tokens.iter().map(|t| t.as_str()).collect();
    let mut merge_seen = std::collections::HashSet::new();
    let mut merges = Vec::with_capacity(merge_strs.len());
    for m in merge_strs {
        // byte-level tokens encode real spaces as G-with-breve, so a
        // merge is exactly "left right": one space, both sides full.
        let (a, b) = match m.split_once(' ') {
            Some((a, b)) if !a.is_empty() && !b.is_empty() && !b.contains(' ') => (a, b),
            _ => {
                return Err(LoaderError::Structure {
                    reason: format!("malformed merge entry \"{}\"", echo(&m)),
                })
            }
        };
        // a merge only makes sense if both sides and their product
        // are vocabulary entries; otherwise the BPE build fails later
        // and the gate would have lied about validating the border
        let product = format!("{a}{b}");
        for piece in [a, b, product.as_str()] {
            if !vocab_set.contains(piece) {
                return Err(LoaderError::Structure {
                    reason: format!(
                        "merge \"{}\" refers to \"{}\", which is not in the vocab",
                        echo(&m),
                        echo(piece)
                    ),
                });
            }
        }
        if !merge_seen.insert((a.to_string(), b.to_string())) {
            return Err(LoaderError::Structure {
                reason: format!("duplicate merge entry \"{}\"", echo(&m)),
            });
        }
        merges.push((a.to_string(), b.to_string()));
    }

    // The metadata under-reports stopping (book 7.3, landmine 2):
    // assemble eos plus <|endoftext|>, looked up by string. A qwen3
    // vocab without that token is a broken conversion; the promise
    // that stop_token_ids is COMPLETE is worth refusing over.
    let endoftext = tokens
        .iter()
        .position(|t| t == "<|endoftext|>")
        .ok_or_else(|| LoaderError::Structure {
            reason: "vocab has no <|endoftext|> token; the stop set cannot be assembled"
                .to_string(),
        })? as u32;
    // the ChatML end marker is what the template actually emits; an
    // eos id that names any other token would generate forever
    if tokens[eos as usize] != "<|im_end|>" {
        return Err(LoaderError::Structure {
            reason: format!(
                "eos_token_id {eos} names \"{}\", expected \"<|im_end|>\"",
                echo(&tokens[eos as usize])
            ),
        });
    }
    // both stop tokens AND the ChatML opener the template emits
    // must be typed Control: anything else and the tokenizer will
    // not register them as special, so template markers would
    // tokenize as plain text
    let im_start = tokens
        .iter()
        .position(|t| t == "<|im_start|>")
        .ok_or_else(|| LoaderError::Structure {
            reason: "vocab has no <|im_start|> token; the ChatML template cannot render"
                .to_string(),
        })? as u32;
    for id in [eos, endoftext, im_start] {
        if token_types[id as usize] != TokenType::Control {
            return Err(LoaderError::Structure {
                reason: format!(
                    "template marker {id} (\"{}\") has token_type {:?}, expected Control",
                    echo(&tokens[id as usize]),
                    token_types[id as usize]
                ),
            });
        }
    }

    // Optional template markers the ChatML template may emit under
    // tools/thinking modes. Each rule: (a) if the compiled template
    // MENTIONS the marker by name, the vocab must contain it (the
    // conversion would otherwise BPE-split the marker at render
    // time); (b) whenever it IS in the vocab, it must be
    // USER_DEFINED so the tokenizer registers it atomically.
    let template_src = chat_template.source();
    for marker in [
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ] {
        let position = tokens.iter().position(|t| t == marker);
        if template_src.contains(marker) && position.is_none() {
            return Err(LoaderError::Structure {
                reason: format!("chat template mentions \"{marker}\" but the vocab does not"),
            });
        }
        if let Some(pos) = position {
            if token_types[pos] != TokenType::UserDefined {
                return Err(LoaderError::Structure {
                    reason: format!(
                        "optional marker {pos} (\"{marker}\") has token_type {:?}, expected UserDefined",
                        token_types[pos]
                    ),
                });
            }
        }
    }
    let mut stop_token_ids = vec![eos];
    if endoftext != eos {
        stop_token_ids.push(endoftext);
    }

    // Last step of pass 1: render smoke test. Every tokenizer piece
    // has passed structural checks; now prove the template really
    // emits ChatML with them, so a syntactically valid but non-
    // ChatML template does not load and later render broken prompts.
    // canary strings that MUST appear in the rendered output IN
    // ORDER: markers, both roles, both contents. Uses two distinct
    // nonces, so a template that hardcodes canary text or that only
    // renders messages[0] both fail. The nonces are asserted absent
    // from template_source_ref so a template that literalizes them
    // cannot slip through.
    let user_nonce = "nucleon_smoke_user_9c31f2";
    let asst_nonce = "nucleon_smoke_assistant_44a7e0";
    if template_source_ref.contains(user_nonce) || template_source_ref.contains(asst_nonce) {
        return Err(LoaderError::Template {
            reason: "chat template contains a smoke-test nonce literally; nonce clash".to_string(),
        });
    }
    let smoke = chat_template
        .environment()
        .get_template("chat")
        .expect("added at ChatTemplate::new")
        .render(minijinja::context! {
            messages => vec![
                minijinja::context! { role => "user", content => user_nonce },
                minijinja::context! { role => "assistant", content => asst_nonce },
            ],
            add_generation_prompt => true,
            enable_thinking => false,
        })
        .map_err(|e| LoaderError::Template {
            reason: format!("render smoke test failed: {e}"),
        })?;
    let mut cursor = 0;
    for marker in [
        "<|im_start|>",
        "user",
        user_nonce,
        "<|im_end|>",
        "<|im_start|>",
        "assistant",
        asst_nonce,
        "<|im_end|>",
        "<|im_start|>assistant",
    ] {
        match smoke[cursor..].find(marker) {
            Some(rel) => cursor += rel + marker.len(),
            None => {
                return Err(LoaderError::Template {
                    reason: format!(
                        "chat template rendered without \"{marker}\" in the expected order; not a valid ChatML template"
                    ),
                })
            }
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

    // structural lies (overlap) refuse before completeness does;
    // the spec's zero-padding rule applies between tensors too, and
    // to the leading gap before the first tensor
    ranges.sort();
    if let Some(first) = ranges.first() {
        let head = &bytes[container.data_start..container.data_start + first.0 as usize];
        if head.iter().any(|b| *b != 0) {
            return Err(LoaderError::Structure {
                reason: format!("padding before tensor \"{}\" is not zeroed", first.2),
            });
        }
    }
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(LoaderError::Structure {
                reason: format!(
                    "tensors \"{}\" and \"{}\" overlap in the data region",
                    pair[0].2, pair[1].2
                ),
            });
        }
        let gap_start = container.data_start + pair[0].1 as usize;
        let gap_end = container.data_start + pair[1].0 as usize;
        if bytes[gap_start..gap_end].iter().any(|b| *b != 0) {
            return Err(LoaderError::Structure {
                reason: format!(
                    "padding between tensors \"{}\" and \"{}\" is not zeroed",
                    pair[0].2, pair[1].2
                ),
            });
        }
    }
    // trailing bytes after the last tensor must also be zero
    if let Some(last) = ranges.last() {
        let tail_start = container.data_start + last.1 as usize;
        let tail_end = container.data_start + region_len;
        if bytes[tail_start..tail_end].iter().any(|b| *b != 0) {
            return Err(LoaderError::Structure {
                reason: format!("trailing bytes after tensor \"{}\" are not zeroed", last.2),
            });
        }
    }

    if !expected.is_empty() {
        // deterministic diagnostics: name the alphabetically first
        let name = expected.keys().min().expect("nonempty").clone();
        return Err(LoaderError::MissingTensor { name });
    }

    // decoded-memory budget: dequant expands Q8_0 to ~3.76x while
    // load_bytes also holds the raw file bytes. 120 GB fits the
    // flagship Qwen3.8-27B Q8_0 (~104 GB dequantized); anything
    // bigger refuses cleanly instead of OOM-aborting.
    const MAX_DECODED_BYTES: u64 = 120 * 1024 * 1024 * 1024;
    let mut decoded_bytes: u64 = 0;
    for (info, _shape, n_elems, _byte_len) in &plan {
        let elem_total = n_elems
            .checked_mul(4)
            .ok_or_else(|| LoaderError::Structure {
                reason: format!("tensor \"{}\" decoded size overflows", info.name),
            })?;
        decoded_bytes =
            decoded_bytes
                .checked_add(elem_total)
                .ok_or_else(|| LoaderError::Structure {
                    reason: "total decoded size overflows".to_string(),
                })?;
    }
    if decoded_bytes > MAX_DECODED_BYTES {
        return Err(LoaderError::Structure {
            reason: format!(
                "dequantized weights would be {decoded_bytes} bytes; nucleon caps at {MAX_DECODED_BYTES}"
            ),
        });
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

/// The HF ByteLevel byte-to-character mapping (GPT-2's
/// bytes_to_unicode): printable bytes map to themselves, and the
/// remaining ~68 bytes get sequential characters starting at 256.
/// Ported from tokenizers' pre_tokenizers::byte_level::bytes_char.
fn byte_level_char(b: u8) -> char {
    fn printable(b: u8) -> bool {
        let x = b as u32;
        (0x21..=0x7E).contains(&x) || (0xA1..=0xAC).contains(&x) || (0xAE..=0xFF).contains(&x)
    }
    if printable(b) {
        return char::from_u32(b as u32).expect("printable range");
    }
    // non-printable bytes get 256 + n, where n counts how many
    // earlier bytes were also non-printable, in byte order
    let mut n: u32 = 0;
    for cand in 0..b {
        if !printable(cand) {
            n += 1;
        }
    }
    char::from_u32(256 + n).expect("PUA range")
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
