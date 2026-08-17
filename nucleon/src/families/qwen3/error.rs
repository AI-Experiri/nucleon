//! `FamilyError` — the ways the family module can refuse or fail.
//!
//! Build covers rejected inputs at the border (Yamf's fields are
//! public, so from_yamf re-validates rather than trust). Forward
//! covers runtime failures from the compute layer (mlx-rs errors,
//! shape mismatches). Same shape as LoaderError / TokenizerError.

use std::fmt;

#[derive(Debug)]
pub enum FamilyError {
    /// `from_yamf` could not assemble the family from the Yamf.
    Build { reason: String },
    /// A runtime failure inside the forward pass.
    Forward { reason: String },
}

impl FamilyError {
    // Every from_yamf refusal builds a custom message inline, so
    // no build() constructor helper is needed today; add one when
    // it becomes worth it.
    pub(crate) fn forward(e: impl fmt::Display) -> Self {
        FamilyError::Forward {
            reason: e.to_string(),
        }
    }
}

impl fmt::Display for FamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FamilyError::Build { reason } => write!(f, "family build failed: {reason}"),
            FamilyError::Forward { reason } => write!(f, "forward pass failed: {reason}"),
        }
    }
}

impl std::error::Error for FamilyError {}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
