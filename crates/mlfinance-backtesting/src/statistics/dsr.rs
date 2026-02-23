//! Deflated Sharpe Ratio (DSR).
//!
//! Adjusts the Probabilistic Sharpe Ratio for the multiple testing problem.
//! When many strategies are tested, the expected maximum Sharpe Ratio under
//! the null increases, and the DSR accounts for this.

use statrs::distribution::{ContinuousCDF, Normal};

/// Deflated Sharpe Ratio: adjusts PSR for multiple testing.
///
/// The DSR replaces the benchmark SR in the PSR formula with the expected
/// maximum SR under the null hypothesis of no skill:
///
///   E[max(SR)] ~ sr_std * [(1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))]
///
/// where gamma is the Euler-Mascheroni constant, N is the number of trials,
/// and sr_std is the standard deviation of SR estimates.
///
/// # Arguments
/// * `observed_sr` - The observed (estimated) Sharpe Ratio.
/// * `sr_std` - Standard deviation of the SR estimates across trials.
/// * `n_observations` - Number of return observations per strategy.
/// * `n_trials` - Number of strategies/configurations tested.
/// * `skewness` - Skewness of the return distribution.
/// * `kurtosis` - Excess kurtosis of the return distribution.
///
/// # Returns
/// DSR in `[0, 1]`, the probability that the observed SR exceeds what would
/// be expected by chance given the number of trials. Returns 0.0 if invalid.
pub fn deflated_sharpe_ratio(
    observed_sr: f64,
    sr_std: f64,
    n_observations: usize,
    n_trials: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    if n_observations < 2 || n_trials == 0 || sr_std < 0.0 {
        return 0.0;
    }

    // Expected maximum SR under the null
    let expected_max_sr = expected_max_sr_null(n_trials, sr_std);

    // Use PSR formula with the deflated benchmark
    let n = n_observations as f64;
    let sr = observed_sr;

    let denom_sq = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr;

    if denom_sq <= 0.0 {
        return if observed_sr > expected_max_sr {
            1.0
        } else {
            0.0
        };
    }

    let denom = denom_sq.sqrt();
    let numerator = (sr - expected_max_sr) * (n - 1.0).sqrt();

    if denom < 1e-15 {
        return if numerator > 0.0 { 1.0 } else { 0.0 };
    }

    let z = numerator / denom;

    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");
    normal.cdf(z)
}

/// Expected maximum SR under the null hypothesis.
///
/// Uses the approximation:
///   E[max(SR)] ~ sr_std * [(1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))]
///
/// For N=1, returns 0.
fn expected_max_sr_null(n_trials: usize, sr_std: f64) -> f64 {
    if n_trials <= 1 {
        return 0.0;
    }

    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");
    let euler_mascheroni = 0.5772156649015329;
    let n = n_trials as f64;

    let p1 = 1.0 - 1.0 / n;
    let p2 = 1.0 - 1.0 / (n * std::f64::consts::E);

    // Clamp probabilities to avoid infinity from inverse CDF
    let p1 = p1.min(1.0 - 1e-15);
    let p2 = p2.min(1.0 - 1e-15);

    let z1 = normal.inverse_cdf(p1);
    let z2 = normal.inverse_cdf(p2);

    sr_std * ((1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsr_single_trial() {
        // With only one trial, expected max SR = 0, so DSR = PSR
        let dsr = deflated_sharpe_ratio(1.0, 1.0, 252, 1, 0.0, 3.0);
        assert!(dsr > 0.5, "dsr={} with single trial should be > 0.5", dsr);
    }

    #[test]
    fn test_dsr_many_trials_reduces() {
        let dsr_few = deflated_sharpe_ratio(1.5, 1.0, 252, 5, 0.0, 3.0);
        let dsr_many = deflated_sharpe_ratio(1.5, 1.0, 252, 1000, 0.0, 3.0);
        assert!(
            dsr_many < dsr_few,
            "more trials should reduce DSR: {} vs {}",
            dsr_many,
            dsr_few
        );
    }

    #[test]
    fn test_dsr_range() {
        let dsr = deflated_sharpe_ratio(2.0, 0.5, 100, 50, -0.3, 4.0);
        assert!(dsr >= 0.0 && dsr <= 1.0, "dsr={} out of range", dsr);
    }

    #[test]
    fn test_dsr_invalid_inputs() {
        assert_eq!(deflated_sharpe_ratio(1.0, 1.0, 0, 10, 0.0, 3.0), 0.0);
        assert_eq!(deflated_sharpe_ratio(1.0, 1.0, 252, 0, 0.0, 3.0), 0.0);
    }

    #[test]
    fn test_expected_max_sr_null_increases() {
        let sr1 = expected_max_sr_null(10, 1.0);
        let sr2 = expected_max_sr_null(100, 1.0);
        let sr3 = expected_max_sr_null(1000, 1.0);
        assert!(sr1 < sr2, "more trials should increase expected max SR");
        assert!(sr2 < sr3, "more trials should increase expected max SR");
    }

    #[test]
    fn test_expected_max_sr_null_single() {
        assert_eq!(expected_max_sr_null(1, 1.0), 0.0);
    }

    #[test]
    fn test_dsr_high_sr_many_trials() {
        // A genuinely high SR should still pass even with many trials
        let dsr = deflated_sharpe_ratio(5.0, 1.0, 1000, 100, 0.0, 3.0);
        assert!(dsr > 0.5, "very high SR should survive deflation: {}", dsr);
    }
}
