//! nucleon-mlx — nucleon's compute layer.
//!
//! A thin wrapper over Apple's MLX (via the community `mlx-rs` Rust
//! crate). Every use of MLX inside nucleon flows through THIS crate,
//! so if `mlx-rs` breaks compat on a version bump we have exactly
//! one file to update. Book chapter 3 (MLX) tells the story; this is
//! the code side of it.
//!
//! Today the wrapper only re-exports what the rest of nucleon needs
//! and exposes the pinned-version-triple as runtime calls. As the
//! Qwen3 family lands, the actual op wrappers move here too.

// Re-exports of the mlx-rs surface the rest of nucleon uses. Names
// mirror mlx-rs / mlx-c directly (Array not Tensor, ops::* under
// their mlx-rs names, fast::* untouched) — the wrapping principle
// is: same names, same signatures. This crate exists so that if
// mlx-rs breaks compat on a version bump we have exactly one file
// to update.
pub use mlx_rs::{fast, ops, Array, Device, Dtype};

/// Build an f32 Array from a Vec<f32> and a row-major shape given
/// as usize (nucleon-natural). mlx-rs's `Array::from_slice` wants
/// `&[i32]` for the shape; this helper does the conversion in one
/// place so the rest of nucleon never sees the i32/usize seam.
pub fn array_from_f32(data: Vec<f32>, shape: &[usize]) -> Array {
    let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&data, &shape_i32)
}

/// Read an Array's shape back as `Vec<usize>` (nucleon-natural).
/// The underlying `Array::shape()` returns `&[i32]`; this is a plain
/// widening cast (dims are non-negative, always safe from i32 to
/// usize on aarch64).
pub fn shape_usize(a: &Array) -> Vec<usize> {
    a.shape().iter().map(|&d| d as usize).collect()
}

/// Row-major offset for a flat index vector. Handy in tests when a
/// fixture wants to spot-check `a[i][j]` on an Array; combine with
/// `a.as_slice::<f32>()[offset]`.
pub fn row_major_offset(shape: &[i32], idx: &[usize]) -> usize {
    assert_eq!(shape.len(), idx.len(), "index rank must match shape");
    let mut off = 0usize;
    let mut stride = 1usize;
    for (dim, i) in shape.iter().rev().zip(idx.iter().rev()) {
        off += i * stride;
        stride *= *dim as usize;
    }
    off
}

/// The three pinned versions this build links, all baked into the
/// binary at build time (static link). If we bump `mlx-rs` in the
/// workspace `Cargo.toml`, we update these three constants to match
/// — they are the ground truth the book chapter documents.
const MLX_RS_VERSION: &str = "0.25.3";
const MLX_C_VERSION: &str = "0.5.0";
const MLX_VERSION: &str = "0.30.6";

/// The `mlx-rs` (Rust wrapper) version. Exact-pinned in the
/// workspace `Cargo.toml`.
pub fn mlx_rs_version() -> &'static str {
    MLX_RS_VERSION
}

/// The `mlx-c` (Apple's C API) version transitively pinned by
/// `mlx-rs` (via a git submodule commit).
pub fn mlx_c_version() -> &'static str {
    MLX_C_VERSION
}

/// The Apple MLX (C++ core) version transitively pinned by `mlx-c`
/// (via CMake `FetchContent`). Baked into our binary at build time;
/// a `brew upgrade mlx` on the host does not change what this
/// returns.
pub fn mlx_version() -> &'static str {
    MLX_VERSION
}

/// One-line summary suitable for logging or a `nucleon inspect`
/// verdict block: `mlx-rs 0.25.3 → mlx-c 0.5.0 → MLX 0.30.6`.
pub fn version_triple() -> String {
    format!(
        "mlx-rs {} → mlx-c {} → MLX {}",
        mlx_rs_version(),
        mlx_c_version(),
        mlx_version()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_triple_names_all_three() {
        let s = version_triple();
        assert!(s.contains("mlx-rs"));
        assert!(s.contains("mlx-c"));
        assert!(s.contains("MLX"));
    }

    #[test]
    fn version_pins_are_the_ones_the_book_documents() {
        assert_eq!(mlx_rs_version(), "0.25.3");
        assert_eq!(mlx_c_version(), "0.5.0");
        assert_eq!(mlx_version(), "0.30.6");
    }
}
