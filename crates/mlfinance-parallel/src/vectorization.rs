use itertools::Itertools;

/// Compute the Cartesian product of multiple vectors of `f64` values.
///
/// Given a list of sets, returns every possible combination formed by picking
/// one element from each set. Equivalent to Snippet 20.2.
///
/// # Arguments
///
/// * `sets` - Slice of vectors; each vector represents one dimension of the
///   product space.
///
/// # Returns
///
/// A vector of vectors where each inner vector has `sets.len()` elements.
/// Returns `vec![vec![]]` (a single empty combination) when `sets` is empty.
pub fn cartesian_product(sets: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if sets.is_empty() {
        return vec![vec![]];
    }
    sets.iter()
        .map(|s| s.iter().cloned())
        .multi_cartesian_product()
        .collect()
}

/// Generate all index tuples for a multi-dimensional grid with the given dimensions.
///
/// # Arguments
///
/// * `dims` - Slice of dimension sizes. For example `[2, 3]` produces all
///   `(i, j)` with `0 <= i < 2` and `0 <= j < 3`.
///
/// # Returns
///
/// A vector of index vectors. The total count equals the product of all dimensions.
pub fn index_product(dims: &[usize]) -> Vec<Vec<usize>> {
    if dims.is_empty() {
        return vec![vec![]];
    }
    dims.iter()
        .map(|&d| 0..d)
        .multi_cartesian_product()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_product() {
        let sets = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = cartesian_product(&sets);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], vec![1.0, 3.0]);
        assert_eq!(result[3], vec![2.0, 4.0]);
    }

    #[test]
    fn test_index_product() {
        let dims = vec![2, 3];
        let result = index_product(&dims);
        assert_eq!(result.len(), 6);
    }
}
