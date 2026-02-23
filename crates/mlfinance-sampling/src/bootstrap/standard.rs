use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Standard IID bootstrap for comparison.
///
/// Draws `num_samples` indices uniformly at random (with replacement) from
/// `0..num_observations`. This is the classical bootstrap and does not account
/// for temporal dependencies.
///
/// # Arguments
/// * `num_observations` - size of the population to draw from
/// * `num_samples` - number of samples to draw
/// * `seed` - random seed for reproducibility
///
/// # Returns
/// A vector of `num_samples` indices drawn uniformly with replacement.
pub fn standard_bootstrap(num_observations: usize, num_samples: usize, seed: u64) -> Vec<usize> {
    if num_observations == 0 || num_samples == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_samples)
        .map(|_| rng.gen_range(0..num_observations))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_bootstrap_empty() {
        let result = standard_bootstrap(0, 5, 42);
        assert!(result.is_empty());
    }

    #[test]
    fn test_standard_bootstrap_zero_samples() {
        let result = standard_bootstrap(10, 0, 42);
        assert!(result.is_empty());
    }

    #[test]
    fn test_standard_bootstrap_correct_length() {
        let result = standard_bootstrap(100, 50, 42);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_standard_bootstrap_in_range() {
        let n = 10;
        let result = standard_bootstrap(n, 100, 42);
        for &idx in &result {
            assert!(idx < n);
        }
    }

    #[test]
    fn test_standard_bootstrap_deterministic() {
        let r1 = standard_bootstrap(100, 50, 123);
        let r2 = standard_bootstrap(100, 50, 123);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_standard_bootstrap_single_observation() {
        let result = standard_bootstrap(1, 10, 42);
        assert_eq!(result.len(), 10);
        for &idx in &result {
            assert_eq!(idx, 0);
        }
    }
}
