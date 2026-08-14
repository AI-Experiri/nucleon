//! The Tensor struct, its invariant, and its constructors.
//!
//! Deliberately dumb: row-major, contiguous, f32-only, no strides, no
//! views (those arrive as separate files in this folder when the story
//! needs them). Storage moves behind the Backend for the GPU; this type
//! stays the CPU/reference form.

/// An n-dimensional array of `f32`, row-major and contiguous.
///
/// No `PartialEq`: exact float equality is a trap (NaN != NaN, rounding).
/// Tests compare `data()` slices explicitly.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<f32>,
}

/// Number of elements a shape describes, or `None` on usize overflow.
/// `pub(crate)` so the loader can validate untrusted checkpoint shapes
/// with the same arithmetic the constructors trust.
pub(crate) fn checked_element_count(shape: &[usize]) -> Option<usize> {
    let mut count: usize = 1;
    for &dim in shape {
        count = count.checked_mul(dim)?;
    }
    Some(count)
}

/// Constructor-side wrapper: overflow here is a caller bug. In release
/// builds a plain `product()` would wrap and silently break the
/// shape/data invariant every method relies on.
fn element_count(shape: &[usize]) -> usize {
    checked_element_count(shape)
        .unwrap_or_else(|| panic!("shape {shape:?} element count overflows usize"))
}

impl Tensor {
    /// A tensor from raw parts. Trusted-caller API: external data such as
    /// checkpoint files is validated by the loader before tensors are built.
    ///
    /// # Panics
    /// If `data.len()` does not match the shape's element count, or the
    /// element count overflows usize — caller bugs, not runtime conditions.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected = element_count(&shape);
        assert_eq!(
            data.len(),
            expected,
            "shape {shape:?} wants {expected} elements, got {}",
            data.len()
        );
        Self { shape, data }
    }

    /// All zeros with the given shape. Trusted-caller API: the shape is
    /// assumed sane (engine-internal or loader-validated).
    ///
    /// # Panics
    /// On element-count overflow; a huge-but-valid shape fails at
    /// allocation. The loader's max-size guard is the real protection for
    /// untrusted inputs.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = element_count(&shape);
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    /// Build a tensor by calling `f` on every index, in row-major order.
    /// Mainly for readable test fixtures.
    ///
    /// # Panics
    /// If the shape's element count overflows usize (trusted-caller API,
    /// like all constructors here).
    pub fn from_fn(shape: Vec<usize>, mut f: impl FnMut(&[usize]) -> f32) -> Self {
        let len = element_count(&shape);
        let mut data = Vec::with_capacity(len);
        let mut index = vec![0; shape.len()];
        for _ in 0..len {
            data.push(f(&index));
            // odometer increment: bump the last dim, carry leftward on overflow
            for dim in (0..shape.len()).rev() {
                index[dim] += 1;
                if index[dim] < shape[dim] {
                    break;
                }
                index[dim] = 0;
            }
        }
        Self { shape, data }
    }
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod tests;
