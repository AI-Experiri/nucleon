//! The GGUF container parser — book chapter 7.2, byte by byte.
//!
//! Everything little-endian. Layout: magic "GGUF", version (must be
//! 3), tensor count, metadata count, the metadata key-values, the
//! tensor infos, zero-padding to the alignment, then tensor data.
//! This file knows bytes and structure only; what the values MEAN is
//! config.rs and yamf.rs.

use std::collections::HashMap;

use crate::loader::error::LoaderError;

/// Spec caps, used as sanity limits so a lying header cannot make us
/// allocate absurdly before bounds checks catch it.
const MAX_TENSORS: u64 = 1_000_000;
const MAX_METADATA_KVS: u64 = 100_000;
const MAX_TENSOR_NAME_BYTES: usize = 64;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_ARRAY_NESTING: u32 = 8;
/// Largest legitimate arrays are the vocab (151,936) and merges; a
/// hostile file could otherwise turn N honest bytes into ~32N bytes
/// of MetaValue objects.
const MAX_ARRAY_ELEMS: u64 = 10_000_000;

/// One metadata value, spec type ids 0..=12.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// The declared element type id survives even when the array is
    /// empty, so typed consumers can validate it.
    Array {
        elem_type_id: u32,
        items: Vec<MetaValue>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// The type's name for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            MetaValue::U8(_) => "u8",
            MetaValue::I8(_) => "i8",
            MetaValue::U16(_) => "u16",
            MetaValue::I16(_) => "i16",
            MetaValue::U32(_) => "u32",
            MetaValue::I32(_) => "i32",
            MetaValue::F32(_) => "f32",
            MetaValue::Bool(_) => "bool",
            MetaValue::Str(_) => "string",
            MetaValue::Array { .. } => "array",
            MetaValue::U64(_) => "u64",
            MetaValue::I64(_) => "i64",
            MetaValue::F64(_) => "f64",
        }
    }
}

/// One tensor's directory entry. `dims` stays in the file's ne order
/// (dims[0] = contiguous row length, REVERSED vs our row-major
/// shapes); yamf.rs does the reversal, so the trap has one home.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub offset: u64,
}

/// The parsed container: everything before the tensor data, plus
/// where that data starts.
#[derive(Debug)]
pub struct Container {
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    // read by tests today; the inspect command will want it too
    #[allow(dead_code)]
    pub alignment: u64,
    pub data_start: usize,
    pub file_len: usize,
}

/// A cursor over the file bytes; every read is bounds-checked and a
/// failure names what was being read and where.
struct Rd<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], LoaderError> {
        let end = self.pos.checked_add(n).ok_or(LoaderError::Structure {
            reason: format!("length overflow reading {what}"),
        })?;
        if end > self.b.len() {
            return Err(LoaderError::Truncated {
                reading: what,
                at: self.pos,
            });
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, LoaderError> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, LoaderError> {
        let s = self.take(2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, LoaderError> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, LoaderError> {
        let s = self.take(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    fn string(&mut self, what: &'static str) -> Result<String, LoaderError> {
        // no legitimate metadata string approaches this: the largest
        // real one is the ~4 KB chat template
        const MAX_STRING_VALUE_BYTES: usize = 10_000_000;
        self.string_capped(MAX_STRING_VALUE_BYTES, what)
    }

    /// A string whose spec-bounded length is checked BEFORE any byte
    /// is copied, so a hostile length cannot force the allocation.
    fn string_capped(&mut self, max: usize, what: &'static str) -> Result<String, LoaderError> {
        let len = self.u64(what)?;
        let len = usize::try_from(len).map_err(|_| LoaderError::Structure {
            reason: format!("string length {len} overflows usize ({what})"),
        })?;
        if len > max {
            return Err(LoaderError::Structure {
                reason: format!("{what} of {len} bytes exceeds the spec's {max}"),
            });
        }
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| LoaderError::Structure {
            reason: format!("invalid UTF-8 in {what}"),
        })
    }
}

/// The fewest bytes one encoded value of this type can occupy;
/// used to bound array counts before allocating.
fn min_encoded_size(type_id: u32) -> u64 {
    match type_id {
        0 | 1 | 7 => 1, // u8, i8, bool
        2..=3 => 2,     // u16, i16
        4..=6 => 4,     // u32, i32, f32
        8 => 8,         // string: at least its u64 length
        9 => 12,        // array: elem type + count
        _ => 8,         // u64, i64, f64
    }
}

/// Read one metadata value of the given type id.
fn read_value(
    r: &mut Rd<'_>,
    type_id: u32,
    key: &str,
    depth: u32,
) -> Result<MetaValue, LoaderError> {
    if depth > MAX_ARRAY_NESTING {
        return Err(LoaderError::Structure {
            reason: format!("metadata key \"{key}\" nests arrays deeper than {MAX_ARRAY_NESTING}"),
        });
    }
    Ok(match type_id {
        0 => MetaValue::U8(r.u8("a u8 value")?),
        1 => MetaValue::I8(r.u8("an i8 value")? as i8),
        2 => MetaValue::U16(r.u16("a u16 value")?),
        3 => MetaValue::I16(r.u16("an i16 value")? as i16),
        4 => MetaValue::U32(r.u32("a u32 value")?),
        5 => MetaValue::I32(r.u32("an i32 value")? as i32),
        6 => MetaValue::F32(f32::from_bits(r.u32("an f32 value")?)),
        7 => match r.u8("a bool value")? {
            0 => MetaValue::Bool(false),
            1 => MetaValue::Bool(true),
            other => {
                return Err(LoaderError::Structure {
                    reason: format!("bool byte for key \"{key}\" is {other}, must be 0 or 1"),
                })
            }
        },
        8 => MetaValue::Str(r.string("a string value")?),
        9 => {
            let elem_type_id = r.u32("an array's element type")?;
            if !(0..=12).contains(&elem_type_id) {
                return Err(LoaderError::UnknownMetaType {
                    key: key.to_string(),
                    type_id: elem_type_id,
                });
            }
            let count = r.u64("an array's element count")?;
            // every element consumes at least min_encoded_size bytes,
            // so a count the remaining file cannot hold is a lie; the
            // check runs BEFORE any allocation grows
            let min = min_encoded_size(elem_type_id);
            let remaining = (r.b.len() - r.pos) as u64;
            if count > MAX_ARRAY_ELEMS {
                return Err(LoaderError::Structure {
                    reason: format!(
                        "array under key \"{key}\" claims {count} elements, above the {MAX_ARRAY_ELEMS} cap"
                    ),
                });
            }
            if count > remaining / min {
                return Err(LoaderError::Structure {
                    reason: format!(
                        "array under key \"{key}\" claims {count} elements, file cannot hold them"
                    ),
                });
            }
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(read_value(r, elem_type_id, key, depth + 1)?);
            }
            MetaValue::Array {
                elem_type_id,
                items,
            }
        }
        10 => MetaValue::U64(r.u64("a u64 value")?),
        11 => MetaValue::I64(r.u64("an i64 value")? as i64),
        12 => MetaValue::F64(f64::from_bits(r.u64("an f64 value")?)),
        other => {
            return Err(LoaderError::UnknownMetaType {
                key: key.to_string(),
                type_id: other,
            })
        }
    })
}

/// Parse the container out of the raw file bytes.
pub fn parse(bytes: &[u8]) -> Result<Container, LoaderError> {
    let mut r = Rd { b: bytes, pos: 0 };

    let magic = r.take(4, "the magic bytes")?;
    if magic != b"GGUF" {
        return Err(LoaderError::NotGguf {
            found: [magic[0], magic[1], magic[2], magic[3]],
        });
    }
    let version = r.u32("the version field")?;
    if version != 3 {
        return Err(LoaderError::UnsupportedVersion { found: version });
    }
    let tensor_count = r.u64("the tensor count")?;
    if tensor_count > MAX_TENSORS {
        return Err(LoaderError::Structure {
            reason: format!("tensor count {tensor_count} exceeds the sanity cap {MAX_TENSORS}"),
        });
    }
    let kv_count = r.u64("the metadata count")?;
    if kv_count > MAX_METADATA_KVS {
        return Err(LoaderError::Structure {
            reason: format!("metadata count {kv_count} exceeds the sanity cap {MAX_METADATA_KVS}"),
        });
    }

    let mut metadata = HashMap::new();
    for _ in 0..kv_count {
        // spec: keys are ASCII, at most 65535 bytes; the cap applies
        // before any byte is copied
        let key = r.string_capped(65_535, "a metadata key")?;
        if !key.is_ascii() {
            return Err(LoaderError::Structure {
                reason: format!("metadata key \"{key}\" is not ASCII (the spec requires it)"),
            });
        }
        // duplicate keys refuse BEFORE their value is parsed, so a
        // hostile duplicate cannot attach an expensive payload
        if metadata.contains_key(&key) {
            return Err(LoaderError::Structure {
                reason: format!("duplicate metadata key \"{key}\""),
            });
        }
        let type_id = r.u32("a metadata value type")?;
        let value = read_value(&mut r, type_id, &key, 0)?;
        metadata.insert(key, value);
    }

    // The alignment can be declared anywhere in the metadata, so it
    // is resolved after all keys are read. Default 32; must be a
    // positive multiple of 8.
    // spec: general.alignment is a u32; anything else is refused.
    let alignment = match metadata.get("general.alignment") {
        None => 32,
        Some(MetaValue::U32(a)) => u64::from(*a),
        Some(other) => {
            return Err(LoaderError::WrongType {
                key: "general.alignment".to_string(),
                want: "u32",
                found: other.kind(),
            })
        }
    };
    if alignment == 0 || alignment % 8 != 0 {
        return Err(LoaderError::Structure {
            reason: format!("alignment {alignment} is not a positive multiple of 8"),
        });
    }

    let mut tensors = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..tensor_count {
        let name = r.string_capped(MAX_TENSOR_NAME_BYTES, "a tensor name")?;
        if !seen.insert(name.clone()) {
            return Err(LoaderError::Structure {
                reason: format!("duplicate tensor \"{name}\""),
            });
        }
        let n_dims = r.u32("a tensor's dimension count")?;
        if n_dims == 0 || n_dims > MAX_TENSOR_DIMS {
            return Err(LoaderError::Structure {
                reason: format!("tensor \"{name}\" has {n_dims} dims; the spec allows 1 to 4"),
            });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = r.u64("a tensor dimension")?;
            if d == 0 {
                return Err(LoaderError::Structure {
                    reason: format!("tensor \"{name}\" has a zero dimension"),
                });
            }
            dims.push(d);
        }
        let type_id = r.u32("a tensor's ggml type")?;
        let offset = r.u64("a tensor's data offset")?;
        if offset % alignment != 0 {
            return Err(LoaderError::Structure {
                reason: format!("tensor \"{name}\" offset {offset} is not aligned to {alignment}"),
            });
        }
        tensors.push(TensorInfo {
            name,
            dims,
            type_id,
            offset,
        });
    }

    // Zero-pad to the alignment; the tensor data region starts there.
    let align = usize::try_from(alignment).map_err(|_| LoaderError::Structure {
        reason: format!("alignment {alignment} overflows usize"),
    })?;
    let rem = r.pos % align;
    let pad = if rem == 0 { 0 } else { align - rem };
    let data_start = r.pos.checked_add(pad).ok_or(LoaderError::Structure {
        reason: "padding overflows the file position".to_string(),
    })?;
    if data_start > bytes.len() {
        return Err(LoaderError::Truncated {
            reading: "the padding before tensor data",
            at: r.pos,
        });
    }
    if bytes[r.pos..data_start].iter().any(|b| *b != 0) {
        return Err(LoaderError::Structure {
            reason: "padding before tensor data is not zeroed".to_string(),
        });
    }

    Ok(Container {
        metadata,
        tensors,
        alignment,
        data_start,
        file_len: bytes.len(),
    })
}

#[cfg(test)]
#[path = "container_tests.rs"]
mod tests;
