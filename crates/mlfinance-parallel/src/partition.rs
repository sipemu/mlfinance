/// Linear partitioning of `[0, num_atoms)` into `num_threads` contiguous groups.
///
/// Each group receives approximately the same number of atoms (indices).
/// Equivalent to Snippet 20.5 (`linParts`).
///
/// # Arguments
///
/// * `num_atoms` - Total number of elements to partition.
/// * `num_threads` - Number of partitions (groups) to create.
///
/// # Returns
///
/// A vector of `(start, end)` pairs where each pair defines a half-open range.
/// Returns an empty vector when either argument is zero.
pub fn lin_parts(num_atoms: usize, num_threads: usize) -> Vec<(usize, usize)> {
    if num_threads == 0 || num_atoms == 0 {
        return vec![];
    }
    let num_threads = num_threads.min(num_atoms);
    let mut parts = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let start = (i * num_atoms) / num_threads;
        let end = ((i + 1) * num_atoms) / num_threads;
        parts.push((start, end));
    }
    parts
}

/// Nested partitioning for tasks with O(n^2) complexity.
///
/// Each partition receives approximately equal total work when the cost per atom
/// increases linearly with its index. This is useful for matrix operations on
/// triangular regions. Equivalent to Snippet 20.6 (`nestedParts`).
///
/// # Arguments
///
/// * `num_atoms` - Total number of elements to partition.
/// * `num_threads` - Number of partitions (groups) to create.
/// * `upper_triangle` - If `true`, partitions for an upper-triangle workload
///   (work decreases with index); if `false`, for a lower-triangle workload
///   (work increases with index).
///
/// # Returns
///
/// A vector of `(start, end)` pairs. Empty partitions are omitted, so the
/// returned length may be less than `num_threads`.
pub fn nested_parts(
    num_atoms: usize,
    num_threads: usize,
    upper_triangle: bool,
) -> Vec<(usize, usize)> {
    if num_threads == 0 || num_atoms == 0 {
        return vec![];
    }
    let num_threads = num_threads.min(num_atoms);
    let n = num_atoms as f64;
    let mut parts = Vec::with_capacity(num_threads);

    for i in 0..num_threads {
        let frac_start = i as f64 / num_threads as f64;
        let frac_end = (i + 1) as f64 / num_threads as f64;

        let start = if upper_triangle {
            // Upper triangle: work decreases with index
            let s = n * (n + 1.0) / 2.0;
            (n - (n * n - 2.0 * s * frac_start).max(0.0).sqrt()).round() as usize
        } else {
            // Lower triangle: work increases with index
            let s = n * (n + 1.0) / 2.0;
            ((2.0 * s * frac_start).sqrt()).round() as usize
        };

        let end = if upper_triangle {
            let s = n * (n + 1.0) / 2.0;
            (n - (n * n - 2.0 * s * frac_end).max(0.0).sqrt()).round() as usize
        } else {
            let s = n * (n + 1.0) / 2.0;
            ((2.0 * s * frac_end).sqrt()).round() as usize
        };

        let start = start.min(num_atoms);
        let end = end.min(num_atoms);
        if start < end {
            parts.push((start, end));
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lin_parts() {
        let parts = lin_parts(10, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], (0, 3));
        assert_eq!(parts[1], (3, 6));
        assert_eq!(parts[2], (6, 10));
    }

    #[test]
    fn test_lin_parts_covers_all() {
        let parts = lin_parts(100, 7);
        assert_eq!(parts[0].0, 0);
        assert_eq!(parts.last().unwrap().1, 100);
        // No gaps
        for i in 1..parts.len() {
            assert_eq!(parts[i].0, parts[i - 1].1);
        }
    }

    #[test]
    fn test_nested_parts() {
        let parts = nested_parts(10, 3, false);
        assert!(!parts.is_empty());
        // All partitions should be non-empty
        for &(s, e) in &parts {
            assert!(s < e);
        }
    }
}
