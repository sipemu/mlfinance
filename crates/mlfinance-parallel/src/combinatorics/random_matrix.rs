use ndarray::Array2;

/// Generate a random matrix of the given size with exactly the specified rank.
///
/// Constructs the matrix as the product U * V where U is `(rows x rank)` and
/// V is `(rank x cols)`, both filled with pseudo-random values derived from
/// `seed`. Equivalent to Snippet 21.4.
///
/// # Arguments
///
/// * `rows` - Number of rows in the output matrix.
/// * `cols` - Number of columns in the output matrix.
/// * `rank` - Desired rank (clamped to `min(rows, cols)`).
/// * `seed` - Seed for the deterministic pseudo-random number generator.
///
/// # Returns
///
/// An `Array2<f64>` of shape `(rows, cols)` with rank at most `rank`.
pub fn random_matrix_of_rank(rows: usize, cols: usize, rank: usize, seed: u64) -> Array2<f64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let rank = rank.min(rows).min(cols);

    // Simple LCG-based PRNG for reproducibility
    let mut state = seed;
    let mut next_f64 = || -> f64 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        (state as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };

    // A = U * V where U is (rows x rank) and V is (rank x cols)
    let mut u = Array2::zeros((rows, rank));
    let mut v = Array2::zeros((rank, cols));

    for i in 0..rows {
        for j in 0..rank {
            u[[i, j]] = next_f64();
        }
    }
    for i in 0..rank {
        for j in 0..cols {
            v[[i, j]] = next_f64();
        }
    }

    u.dot(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_matrix_of_rank() {
        let m = random_matrix_of_rank(10, 8, 3, 42);
        assert_eq!(m.shape(), &[10, 8]);
    }

    #[test]
    fn test_rank_1_matrix() {
        let m = random_matrix_of_rank(5, 5, 1, 42);
        assert_eq!(m.shape(), &[5, 5]);
        // Rank-1: all rows should be proportional
        let row0: Vec<f64> = m.row(0).to_vec();
        for i in 1..5 {
            let row_i: Vec<f64> = m.row(i).to_vec();
            // Find ratio
            let mut ratio = None;
            for j in 0..5 {
                if row0[j].abs() > 1e-10 {
                    ratio = Some(row_i[j] / row0[j]);
                    break;
                }
            }
            if let Some(r) = ratio {
                for j in 0..5 {
                    assert!((row_i[j] - r * row0[j]).abs() < 1e-10);
                }
            }
        }
    }
}
