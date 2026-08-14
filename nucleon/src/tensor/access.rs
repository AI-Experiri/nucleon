//! Reading a tensor: shape, raw data, indexed element, borrowed row.

use super::core::Tensor;

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Element at a multi-dimensional index, row-major: the last index moves
    /// fastest, so for shape [rows, cols] the flat position is i*cols + j.
    ///
    /// # Panics
    /// On rank mismatch or an out-of-bounds index.
    pub fn at(&self, index: &[usize]) -> f32 {
        assert_eq!(
            index.len(),
            self.shape.len(),
            "index rank {} does not match tensor rank {}",
            index.len(),
            self.shape.len()
        );
        let mut flat = 0;
        for (dim, &i) in index.iter().enumerate() {
            assert!(
                i < self.shape[dim],
                "index {i} out of bounds for dim {dim} of size {}",
                self.shape[dim]
            );
            flat = flat * self.shape[dim] + i;
        }
        self.data[flat]
    }

    /// One row of a 2-D tensor as a borrowed slice. No copy.
    ///
    /// # Panics
    /// If the tensor is not 2-D or `i` is out of bounds.
    pub fn row(&self, i: usize) -> &[f32] {
        assert_eq!(self.shape.len(), 2, "row() needs a 2-D tensor");
        let cols = self.shape[1];
        assert!(
            i < self.shape[0],
            "row {i} out of bounds for {} rows",
            self.shape[0]
        );
        &self.data[i * cols..(i + 1) * cols]
    }
}

#[cfg(test)]
#[path = "access_tests.rs"]
mod tests;
