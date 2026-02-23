//! Probabilistic Sharpe Ratio (PSR).
//!
//! Computes the probability that the true Sharpe Ratio exceeds a benchmark,
//! accounting for estimation error due to non-normality (skewness and kurtosis).

use statrs::distribution::{ContinuousCDF, Normal};

/// Probabilistic Sharpe Ratio: probability that true SR > benchmark SR.
///
/// PSR adjusts for non-normality of returns by incorporating the skewness and
/// excess kurtosis of the return distribution:
///
///   PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - gamma3*SR + (gamma4-1)/4 * SR^2) )
///
/// where:
/// - SR is the observed Sharpe Ratio
/// - SR* is the benchmark Sharpe Ratio
/// - n is the number of observations
/// - gamma3 is the skewness
/// - gamma4 is the excess kurtosis
/// - Phi is the standard normal CDF
///
/// # Arguments
/// * `observed_sr` - The observed (estimated) Sharpe Ratio.
/// * `benchmark_sr` - The benchmark Sharpe Ratio to test against.
/// * `n_observations` - Number of return observations.
/// * `skewness` - Skewness of the return distribution.
/// * `kurtosis` - Excess kurtosis of the return distribution.
///
/// # Returns
/// PSR in `[0, 1]`, the probability that the true SR exceeds the benchmark.
/// Returns 0.0 if inputs are invalid.
pub fn probabilistic_sharpe_ratio(
    observed_sr: f64,
    benchmark_sr: f64,
    n_observations: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    if n_observations < 2 {
        return 0.0;
    }

    let n = n_observations as f64;
    let sr = observed_sr;

    // Variance of the SR estimator adjusted for non-normality
    let denom_sq = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr;

    if denom_sq <= 0.0 {
        // Degenerate case: return based on sign of difference
        return if observed_sr > benchmark_sr { 1.0 } else { 0.0 };
    }

    let denom = denom_sq.sqrt();
    let numerator = (sr - benchmark_sr) * (n - 1.0).sqrt();

    if denom < 1e-15 {
        return if numerator > 0.0 { 1.0 } else { 0.0 };
    }

    let z = numerator / denom;

    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");
    normal.cdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psr_above_benchmark() {
        // High observed SR, low benchmark, many observations
        let psr = probabilistic_sharpe_ratio(2.0, 0.0, 252, 0.0, 3.0);
        assert!(psr > 0.95, "psr={} should be > 0.95", psr);
    }

    #[test]
    fn test_psr_below_benchmark() {
        let psr = probabilistic_sharpe_ratio(0.5, 2.0, 252, 0.0, 3.0);
        assert!(psr < 0.05, "psr={} should be < 0.05", psr);
    }

    #[test]
    fn test_psr_equal_to_benchmark() {
        let psr = probabilistic_sharpe_ratio(1.0, 1.0, 252, 0.0, 3.0);
        assert!(
            (psr - 0.5).abs() < 0.01,
            "psr={} should be ~0.5 when SR equals benchmark",
            psr
        );
    }

    #[test]
    fn test_psr_skewness_effect() {
        // Negative skewness should reduce PSR (wider confidence interval)
        let psr_no_skew = probabilistic_sharpe_ratio(1.5, 1.0, 100, 0.0, 3.0);
        let psr_neg_skew = probabilistic_sharpe_ratio(1.5, 1.0, 100, -1.0, 3.0);
        // Negative skewness with positive SR increases the denominator
        // because -gamma3*SR = -(-1)*1.5 = +1.5, so denom increases,
        // meaning z decreases, meaning PSR decreases. Wait, let's check:
        // denom_sq = 1 - (-1)*1.5 + (3-1)/4 * 1.5^2 = 1 + 1.5 + 1.125 = 3.625
        // vs no skew: 1 - 0 + 0.5*2.25 = 2.125
        // Larger denom => smaller z => lower PSR
        assert!(
            psr_neg_skew < psr_no_skew,
            "negative skewness should reduce PSR: {} vs {}",
            psr_neg_skew,
            psr_no_skew
        );
    }

    #[test]
    fn test_psr_few_observations() {
        assert_eq!(probabilistic_sharpe_ratio(1.0, 0.0, 0, 0.0, 3.0), 0.0);
        assert_eq!(probabilistic_sharpe_ratio(1.0, 0.0, 1, 0.0, 3.0), 0.0);
    }

    #[test]
    fn test_psr_normal_returns() {
        // For normal returns (skew=0, excess_kurtosis=0) with few observations
        // and a modest SR, the PSR should be between 0.5 and 1.0
        let psr = probabilistic_sharpe_ratio(0.3, 0.0, 30, 0.0, 0.0);
        assert!(
            psr > 0.5 && psr < 1.0,
            "psr={} should be in (0.5, 1.0)",
            psr,
        );
    }

    #[test]
    fn test_psr_range() {
        let psr = probabilistic_sharpe_ratio(0.8, 0.5, 100, -0.5, 4.0);
        assert!(psr >= 0.0 && psr <= 1.0);
    }
}
