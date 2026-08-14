//! The Backend seam: ops trait + the CPU reference implementation.

mod cpu;
mod ops;

pub use cpu::CpuBackend;
pub use ops::Backend;
