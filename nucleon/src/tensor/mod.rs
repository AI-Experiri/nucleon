//! Tensor: the one data structure everything else speaks.
//!
//! One concern per file: core.rs (struct, invariant, constructors),
//! access.rs (reads), display.rs (debug printing). Future chapters add
//! files here (view.rs, iter.rs, ...), never lines to a monolith.

mod access;
mod core;
mod display;

// consumed by the loader; until then only tests reference it
#[allow(unused_imports)]
pub(crate) use core::checked_element_count;
pub use core::Tensor;
