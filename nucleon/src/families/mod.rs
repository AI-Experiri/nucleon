//! Model families. Today: qwen3. When a second family arrives it
//! becomes a sibling submodule and this barrel dispatches.

pub mod qwen3;

pub use qwen3::{from_yamf as qwen3_from_yamf, FamilyError, Qwen3};
