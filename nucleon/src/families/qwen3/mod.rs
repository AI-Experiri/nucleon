//! Qwen3 family — the forward pass, book chapter 9.
//!
//! from_yamf moves the loader's Yamf into named fields; forward
//! runs the sequence-in / logits-out pass exactly as
//! modeling_qwen3.py defines it. No cache (that's Part III); no
//! sampling (that's the sampler chapter); no threading. One
//! contract: ids -> logits, in the shapes the config numbers say.

mod build;
mod error;
mod forward;

pub use build::{from_yamf, Qwen3};
pub use error::FamilyError;
