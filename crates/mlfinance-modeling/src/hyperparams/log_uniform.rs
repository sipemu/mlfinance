//! LogUniform distribution (Snippet 9.4).
//!
//! A distribution where the log of the random variable is uniformly distributed.
//! Useful for sampling hyperparameters that span several orders of magnitude
//! (e.g., learning rates, regularization coefficients).

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Sample from a log-uniform distribution in [low, high].
///
/// The logarithm of the sampled value is uniformly distributed in [ln(low), ln(high)].
///
/// # Arguments
/// * `low` - Lower bound (must be > 0).
/// * `high` - Upper bound (must be > low).
/// * `n` - Number of samples to draw.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
/// Vector of `n` samples from the log-uniform distribution.
///
/// # Panics
/// Panics if `low <= 0`, `high <= low`.
pub fn log_uniform_sample(low: f64, high: f64, n: usize, seed: u64) -> Vec<f64> {
    assert!(low > 0.0, "low must be positive, got {}", low);
    assert!(high > low, "high must be greater than low");

    let mut rng = StdRng::seed_from_u64(seed);
    let log_low = low.ln();
    let log_high = high.ln();

    (0..n)
        .map(|_| {
            let log_val = rng.gen_range(log_low..=log_high);
            log_val.exp()
        })
        .collect()
}

/// A log-uniform distribution parameterized by [low, high].
#[derive(Debug, Clone)]
pub struct LogUniform {
    /// Lower bound (must be > 0).
    pub low: f64,
    /// Upper bound (must be > low).
    pub high: f64,
}

impl LogUniform {
    /// Create a new LogUniform distribution over `[low, high]`.
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (must be positive).
    /// * `high` - Upper bound (must be greater than `low`).
    ///
    /// # Panics
    ///
    /// Panics if `low <= 0` or `high <= low`.
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low > 0.0, "low must be positive, got {}", low);
        assert!(high > low, "high must be greater than low");
        LogUniform { low, high }
    }

    /// Sample a single value from this log-uniform distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator to draw from.
    ///
    /// # Returns
    ///
    /// A sample in `[self.low, self.high]` whose logarithm is uniformly distributed.
    pub fn sample(&self, rng: &mut impl rand::Rng) -> f64 {
        let log_low = self.low.ln();
        let log_high = self.high.ln();
        let log_val = rng.gen_range(log_low..=log_high);
        log_val.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_uniform_sample_range() {
        let samples = log_uniform_sample(0.001, 1000.0, 1000, 42);
        for &s in &samples {
            assert!(s >= 0.001, "Sample below low: {}", s);
            assert!(s <= 1000.0, "Sample above high: {}", s);
        }
    }

    #[test]
    fn test_log_uniform_sample_distribution() {
        // Samples should span orders of magnitude roughly uniformly in log-space
        let samples = log_uniform_sample(0.001, 1000.0, 10000, 42);

        // Count samples in each order of magnitude
        let count_below_01 = samples.iter().filter(|&&s| s < 0.01).count();
        let count_01_1 = samples
            .iter()
            .filter(|&&s| (0.01..0.1).contains(&s))
            .count();
        let count_1_10 = samples.iter().filter(|&&s| (0.1..1.0).contains(&s)).count();
        let count_10_100 = samples
            .iter()
            .filter(|&&s| (1.0..10.0).contains(&s))
            .count();
        let count_100_1000 = samples
            .iter()
            .filter(|&&s| (10.0..100.0).contains(&s))
            .count();
        let count_above_100 = samples.iter().filter(|&&s| s >= 100.0).count();

        // Each decade should have roughly similar count (total range is 6 decades)
        // With 10000 samples, each should get ~1666
        // Allow wide tolerance
        let counts = vec![
            count_below_01,
            count_01_1,
            count_1_10,
            count_10_100,
            count_100_1000,
            count_above_100,
        ];
        for &c in &counts {
            assert!(
                c > 500,
                "Expected reasonable samples in each decade, got {:?}",
                counts
            );
        }
    }

    #[test]
    fn test_log_uniform_sample_reproducible() {
        let s1 = log_uniform_sample(0.01, 10.0, 50, 42);
        let s2 = log_uniform_sample(0.01, 10.0, 50, 42);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "low must be positive")]
    fn test_log_uniform_sample_invalid_low() {
        log_uniform_sample(0.0, 1.0, 10, 42);
    }

    #[test]
    #[should_panic(expected = "high must be greater than low")]
    fn test_log_uniform_sample_invalid_range() {
        log_uniform_sample(10.0, 1.0, 10, 42);
    }

    #[test]
    fn test_log_uniform_struct() {
        let dist = LogUniform::new(0.01, 100.0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let s = dist.sample(&mut rng);
            assert!((0.01..=100.0).contains(&s), "Sample out of range: {}", s);
        }
    }

    #[test]
    fn test_log_uniform_zero_samples() {
        let samples = log_uniform_sample(0.1, 10.0, 0, 42);
        assert!(samples.is_empty());
    }
}
