//! Strategy failure probability (Snippet 15.5).
//!
//! Computes the probability that a strategy's true precision is below the
//! break-even precision, given an estimated precision from a finite sample.

use statrs::distribution::{ContinuousCDF, Normal};

/// Probability that strategy fails: P\[p < p*\].
///
/// Given an estimated precision `p_hat` from `n` observations and a break-even
/// precision `p*`, computes the probability that the true precision is below
/// the break-even level using a normal approximation to the binomial:
///
///   z = (p_hat - p*) / sqrt(p* * (1 - p*) / n)
///   P\[fail\] = Phi(-z) = 1 - Phi(z)
///
/// # Arguments
/// * `estimated_precision` - Estimated hit rate from the sample.
/// * `n_observations` - Number of observations in the sample.
/// * `break_even_precision` - Break-even precision (minimum for profitability).
///
/// # Returns
/// Failure probability in `[0, 1]`. Returns 0.0 if inputs are invalid.
pub fn strategy_failure_probability(
    estimated_precision: f64,
    n_observations: usize,
    break_even_precision: f64,
) -> f64 {
    if n_observations == 0 {
        return 0.0;
    }
    if break_even_precision <= 0.0 {
        return 0.0;
    }
    if break_even_precision >= 1.0 {
        return 1.0;
    }

    let n = n_observations as f64;
    let se = (break_even_precision * (1.0 - break_even_precision) / n).sqrt();

    if se < 1e-15 {
        return if estimated_precision >= break_even_precision {
            0.0
        } else {
            1.0
        };
    }

    let z = (estimated_precision - break_even_precision) / se;
    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");
    1.0 - normal.cdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_probability_high_precision() {
        // Estimated precision well above break-even
        let prob = strategy_failure_probability(0.7, 1000, 0.5);
        assert!(prob < 0.01, "prob={} should be very low", prob);
    }

    #[test]
    fn test_failure_probability_at_break_even() {
        // Estimated = break-even -> ~50% failure probability
        let prob = strategy_failure_probability(0.5, 1000, 0.5);
        assert!((prob - 0.5).abs() < 0.01, "prob={} should be ~0.5", prob);
    }

    #[test]
    fn test_failure_probability_below_break_even() {
        // Estimated below break-even
        let prob = strategy_failure_probability(0.4, 1000, 0.5);
        assert!(prob > 0.99, "prob={} should be very high", prob);
    }

    #[test]
    fn test_failure_probability_few_observations() {
        // With few observations, more uncertainty
        let prob_few = strategy_failure_probability(0.6, 10, 0.5);
        let prob_many = strategy_failure_probability(0.6, 1000, 0.5);
        assert!(
            prob_few > prob_many,
            "fewer observations should give higher failure prob: {} vs {}",
            prob_few,
            prob_many
        );
    }

    #[test]
    fn test_failure_probability_invalid() {
        assert_eq!(strategy_failure_probability(0.5, 0, 0.5), 0.0);
    }

    #[test]
    fn test_failure_probability_edge_break_even() {
        assert_eq!(strategy_failure_probability(0.5, 100, 0.0), 0.0);
        assert_eq!(strategy_failure_probability(0.5, 100, 1.0), 1.0);
    }

    #[test]
    fn test_failure_probability_range() {
        let prob = strategy_failure_probability(0.55, 200, 0.5);
        assert!(prob >= 0.0 && prob <= 1.0, "prob={} out of range", prob);
    }
}
