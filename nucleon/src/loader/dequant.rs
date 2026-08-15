//! ggml tensor types to f32 — book chapter 7.5.
//!
//! Two types today: F32 (id 0) copies through; Q8_0 (id 8) is
//! 34-byte blocks of an f16 scale followed by 32 signed bytes,
//! value = f32(scale) * q. Anything else refuses by name; more
//! types arrive in Part III.

use half::f16;

use crate::loader::error::LoaderError;

pub const GGML_F32: u32 = 0;
pub const GGML_Q8_0: u32 = 8;

const Q8_0_BLOCK_VALUES: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34; // 2 (f16 scale) + 32 (i8 values)

/// How many bytes a tensor of this type and element count occupies.
/// `row_len` is the contiguous dimension (GGUF dims[0]): Q8_0 blocks
/// live inside rows and never span them, so each row must be a whole
/// number of 32-value blocks. The bounds checks in yamf.rs are built
/// on this.
pub fn tensor_byte_len(
    type_id: u32,
    n_elems: u64,
    row_len: u64,
    name: &str,
) -> Result<u64, LoaderError> {
    if row_len == 0 || !n_elems.is_multiple_of(row_len) {
        return Err(LoaderError::Structure {
            reason: format!("tensor \"{name}\": {n_elems} values cannot be rows of {row_len}"),
        });
    }
    match type_id {
        GGML_F32 => n_elems
            .checked_mul(4)
            .ok_or_else(|| LoaderError::Structure {
                reason: format!("tensor \"{name}\" byte length overflows"),
            }),
        GGML_Q8_0 => {
            if !row_len.is_multiple_of(Q8_0_BLOCK_VALUES) {
                return Err(LoaderError::Structure {
                    reason: format!(
                        "tensor \"{name}\" has rows of {row_len} values, not a multiple of 32 (Q8_0 blocks never span rows)"
                    ),
                });
            }
            (n_elems / Q8_0_BLOCK_VALUES)
                .checked_mul(Q8_0_BLOCK_BYTES)
                .ok_or_else(|| LoaderError::Structure {
                    reason: format!("tensor \"{name}\" byte length overflows"),
                })
        }
        other => Err(LoaderError::UnsupportedTensorType {
            name: name.to_string(),
            type_id: other,
        }),
    }
}

/// Decode a tensor's raw bytes into f32 values.
pub fn dequantize(
    type_id: u32,
    bytes: &[u8],
    n_elems: usize,
    row_len: u64,
    name: &str,
) -> Result<Vec<f32>, LoaderError> {
    let expected = tensor_byte_len(type_id, n_elems as u64, row_len, name)?;
    if bytes.len() as u64 != expected {
        return Err(LoaderError::Structure {
            reason: format!(
                "tensor \"{name}\": {} bytes on disk, type needs {expected}",
                bytes.len()
            ),
        });
    }
    match type_id {
        GGML_F32 => {
            let mut out = Vec::with_capacity(n_elems);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        GGML_Q8_0 => {
            let mut out = Vec::with_capacity(n_elems);
            for block in bytes.chunks_exact(Q8_0_BLOCK_BYTES as usize) {
                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                for &q in &block[2..] {
                    out.push(d * f32::from(q as i8));
                }
            }
            Ok(out)
        }
        // tensor_byte_len already refused everything else
        _ => unreachable!("unsupported type ids never reach dequantize"),
    }
}

#[cfg(test)]
#[path = "dequant_tests.rs"]
mod tests;
