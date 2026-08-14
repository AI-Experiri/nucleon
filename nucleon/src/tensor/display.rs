//! Debug printing: small tensors as labeled matrices.

use super::core::Tensor;

/// Small tensors print a shape header plus matrix rows; anything empty,
/// over 64 elements, or above rank 2 prints the header alone. The header
/// disambiguates shapes whose rows look identical (scalar vs [1] vs [1,1]).
impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?})", self.shape)?;
        if self.data.is_empty() || self.data.len() > 64 || self.shape.len() > 2 {
            return Ok(());
        }
        writeln!(f)?;
        let cols = if self.shape.len() == 2 {
            self.shape[1]
        } else {
            self.data.len()
        };
        for row in self.data.chunks(cols.max(1)) {
            write!(f, "[")?;
            for (i, v) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{v}")?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "display_tests.rs"]
mod tests;
