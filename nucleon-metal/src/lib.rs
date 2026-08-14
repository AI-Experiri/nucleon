//! nucleon-metal — the Apple Silicon backend (chapters 10-11).
//!
//! Rust host code via objc2-metal (no C/C++), GPU math in `kernels/*.metal`
//! — the only non-Rust files in the project. `device` is the plumbing
//! (compile kernels, buffers, dispatch); `ops` wraps each kernel as a safe
//! function, parity-tested against plain-Rust reference loops.

pub mod backend;
pub mod device;
pub mod ops;
