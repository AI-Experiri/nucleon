//! `LoaderError` — every way the gate can refuse a file.
//!
//! Book contract (chapter 7.7): each variant carries what was found
//! AND what nucleon supports, so the refusal message doubles as
//! compatibility documentation. Display is written by hand; reading
//! it is the lesson.

use std::fmt;

#[derive(Debug)]
pub enum LoaderError {
    /// The OS failed us: open, metadata, or mmap.
    Io(std::io::Error),
    /// The first four bytes are not "GGUF".
    NotGguf { found: [u8; 4] },
    /// The version field is not the one we implement.
    UnsupportedVersion { found: u32 },
    /// `general.architecture` names a family we do not implement.
    UnsupportedArchitecture { found: String },
    /// A tensor uses a ggml type we cannot decode yet.
    UnsupportedTensorType { name: String, type_id: u32 },
    /// A metadata value carries a type id outside the spec's 0..=12.
    UnknownMetaType { key: String, type_id: u32 },
    /// A key the family requires is absent.
    MissingKey { key: &'static str },
    /// A key exists with the wrong value type.
    WrongType {
        key: String,
        want: &'static str,
        found: &'static str,
    },
    /// The expected tensor set is incomplete.
    MissingTensor { name: String },
    /// The file carries a tensor the family does not explain.
    UnexpectedTensor { name: String },
    /// A tensor's (reversed) dims do not match the family's shape.
    WrongShape {
        name: String,
        want: Vec<usize>,
        found: Vec<usize>,
    },
    /// The file ended before the field being read did.
    Truncated { reading: &'static str, at: usize },
    /// A structural rule broke: bounds, alignment, overlap, counts.
    Structure { reason: String },
    /// The chat template failed to parse at the gate.
    Template { reason: String },
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoaderError::Io(e) => write!(f, "io error: {e}"),
            LoaderError::NotGguf { found } => {
                write!(
                    f,
                    "not a GGUF file (first bytes {found:?}, expected \"GGUF\")"
                )
            }
            LoaderError::UnsupportedVersion { found } => {
                if *found == 50_331_648 {
                    // 3 with its bytes swapped: a big-endian file.
                    write!(
                        f,
                        "GGUF version {found}; nucleon supports 3 \
                         (this value is 3 read big-endian; big-endian files are not supported)"
                    )
                } else {
                    write!(f, "GGUF version {found}; nucleon supports 3")
                }
            }
            LoaderError::UnsupportedArchitecture { found } => {
                write!(f, "architecture \"{found}\"; nucleon supports: qwen3")
            }
            LoaderError::UnsupportedTensorType { name, type_id } => {
                write!(
                    f,
                    "tensor \"{name}\" has ggml type {type_id}; supported: F32 (0), Q8_0 (8)"
                )
            }
            LoaderError::UnknownMetaType { key, type_id } => {
                write!(f, "metadata key \"{key}\" has unknown value type id {type_id} (spec defines 0..=12)")
            }
            LoaderError::MissingKey { key } => write!(f, "missing metadata key \"{key}\""),
            LoaderError::WrongType { key, want, found } => {
                write!(
                    f,
                    "metadata key \"{key}\" has type {found}, expected {want}"
                )
            }
            LoaderError::MissingTensor { name } => write!(f, "missing tensor \"{name}\""),
            LoaderError::UnexpectedTensor { name } => {
                write!(
                    f,
                    "unexpected tensor \"{name}\" (not part of the qwen3 family)"
                )
            }
            LoaderError::WrongShape { name, want, found } => {
                write!(
                    f,
                    "tensor \"{name}\" has shape {found:?}, expected {want:?}"
                )
            }
            LoaderError::Truncated { reading, at } => {
                write!(f, "file ends inside {reading} (at byte {at})")
            }
            LoaderError::Structure { reason } => write!(f, "malformed file: {reason}"),
            LoaderError::Template { reason } => {
                write!(f, "chat template does not parse: {reason}")
            }
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<std::io::Error> for LoaderError {
    fn from(e: std::io::Error) -> Self {
        LoaderError::Io(e)
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
