//! Basic sampling utilities for event-based analysis.
//!
//! Provides simple methods for generating sample indices from a time series.

/// Generate `n` evenly-spaced sample indices between `start` and `end` (inclusive).
///
/// # Arguments
/// - `start`: the first index (inclusive)
/// - `end`: the last index (inclusive)
/// - `n`: number of sample points to generate
///
/// # Returns
/// A `Vec<usize>` of `n` evenly-spaced indices. Returns an empty vec if `n == 0`.
/// Returns `[start]` if `n == 1`. If `start > end`, returns empty.
pub fn linspace_sample(start: usize, end: usize, n: usize) -> Vec<usize> {
    if n == 0 || start > end {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }

    let range = end - start;
    (0..n)
        .map(|i| {
            let frac = i as f64 / (n - 1) as f64;
            start + (frac * range as f64).round() as usize
        })
        .collect()
}

/// Generate `n` pseudo-random sample indices from `[0, total)` using a
/// simple LCG (linear congruential generator) seeded by `seed`.
///
/// This is a deterministic sampling method suitable for reproducible experiments.
/// The returned indices are sorted in ascending order and deduplicated.
///
/// # Arguments
/// - `n`: number of samples desired
/// - `total`: the range `[0, total)` from which to sample
/// - `seed`: random seed for reproducibility
///
/// # Returns
/// A sorted, deduplicated `Vec<usize>` of at most `n` indices.
pub fn uniform_sample(n: usize, total: usize, seed: u64) -> Vec<usize> {
    if n == 0 || total == 0 {
        return vec![];
    }

    // Simple LCG: x_{n+1} = (a*x_n + c) mod m
    let a: u64 = 6364136223846793005;
    let c: u64 = 1442695040888963407;

    let mut state = seed;
    let mut indices: Vec<usize> = Vec::with_capacity(n);

    // Generate more candidates than needed to handle duplicates
    let max_attempts = n * 3 + 100;
    for _ in 0..max_attempts {
        state = state.wrapping_mul(a).wrapping_add(c);
        let idx = (state >> 16) as usize % total;
        indices.push(idx);
        if indices.len() >= n * 2 {
            break;
        }
    }

    // Sort and deduplicate
    indices.sort_unstable();
    indices.dedup();
    indices.truncate(n);
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace_basic() {
        let samples = linspace_sample(0, 100, 5);
        assert_eq!(samples, vec![0, 25, 50, 75, 100]);
    }

    #[test]
    fn test_linspace_single() {
        let samples = linspace_sample(10, 20, 1);
        assert_eq!(samples, vec![10]);
    }

    #[test]
    fn test_linspace_two() {
        let samples = linspace_sample(0, 10, 2);
        assert_eq!(samples, vec![0, 10]);
    }

    #[test]
    fn test_linspace_empty() {
        assert_eq!(linspace_sample(0, 10, 0), Vec::<usize>::new());
    }

    #[test]
    fn test_linspace_start_gt_end() {
        assert_eq!(linspace_sample(10, 5, 3), Vec::<usize>::new());
    }

    #[test]
    fn test_uniform_sample_basic() {
        let samples = uniform_sample(5, 100, 42);
        assert!(samples.len() <= 5);
        assert!(samples.iter().all(|&x| x < 100));
        // Should be sorted
        for w in samples.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_uniform_sample_deterministic() {
        let s1 = uniform_sample(10, 1000, 12345);
        let s2 = uniform_sample(10, 1000, 12345);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_uniform_sample_different_seeds() {
        let s1 = uniform_sample(10, 1000, 1);
        let s2 = uniform_sample(10, 1000, 2);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_uniform_sample_empty() {
        assert_eq!(uniform_sample(0, 100, 42), Vec::<usize>::new());
        assert_eq!(uniform_sample(5, 0, 42), Vec::<usize>::new());
    }
}
