//! Sharpe Ratio from precision and frequency (Snippet 15.1).
//!
//! Derives the expected Sharpe Ratio from the hit rate (precision), betting
//! frequency, and average win/loss ratio.

/// Compute SR from precision (hit rate) and frequency.
///
/// The expected Sharpe Ratio of a strategy can be decomposed as:
///   SR = (p * avg_win - (1-p) * avg_loss) / std
///
/// With `avg_win_loss_ratio = avg_win / avg_loss`, and assuming unit loss:
///   E[return per bet] = p * R - (1 - p)
///   Var[return per bet] = p * (1 - p) * (R + 1)^2
///
/// where R = avg_win_loss_ratio. The annualized SR is then:
///   SR = E / sqrt(Var) * sqrt(freq)
///
/// # Arguments
/// * `precision` - Probability of positive return (hit rate) in `(0, 1)`.
/// * `freq` - Number of bets per year.
/// * `avg_win_loss_ratio` - Ratio of average win to average loss (must be > 0).
///
/// # Returns
/// Expected annualized Sharpe Ratio. Returns 0.0 if inputs are invalid.
pub fn sr_from_precision(precision: f64, freq: f64, avg_win_loss_ratio: f64) -> f64 {
    if precision <= 0.0 || precision >= 1.0 || freq <= 0.0 || avg_win_loss_ratio <= 0.0 {
        return 0.0;
    }

    let p = precision;
    let r = avg_win_loss_ratio;

    // Expected return per bet (normalized by avg_loss = 1)
    let expected = p * r - (1.0 - p);

    // Variance per bet
    let variance = p * (1.0 - p) * (r + 1.0).powi(2);

    if variance <= 0.0 {
        return 0.0;
    }

    let std = variance.sqrt();
    let sr_per_bet = expected / std;

    // Annualize
    sr_per_bet * freq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_from_precision_balanced() {
        // 50% win rate with 1:1 win/loss -> SR = 0
        let sr = sr_from_precision(0.5, 252.0, 1.0);
        assert!(sr.abs() < 1e-10, "sr={} should be ~0", sr);
    }

    #[test]
    fn test_sr_from_precision_positive() {
        // 60% win rate with 1:1 -> positive SR
        let sr = sr_from_precision(0.6, 252.0, 1.0);
        assert!(sr > 0.0, "sr={} should be positive", sr);
    }

    #[test]
    fn test_sr_from_precision_negative() {
        // 40% win rate with 1:1 -> negative SR
        let sr = sr_from_precision(0.4, 252.0, 1.0);
        assert!(sr < 0.0, "sr={} should be negative", sr);
    }

    #[test]
    fn test_sr_from_precision_high_win_loss() {
        // 40% hit rate but 3:1 win/loss -> should be positive
        // E = 0.4 * 3 - 0.6 = 1.2 - 0.6 = 0.6 > 0
        let sr = sr_from_precision(0.4, 252.0, 3.0);
        assert!(sr > 0.0, "sr={} should be positive with high W/L", sr);
    }

    #[test]
    fn test_sr_from_precision_frequency_effect() {
        let sr_low = sr_from_precision(0.55, 52.0, 1.5);
        let sr_high = sr_from_precision(0.55, 252.0, 1.5);
        assert!(
            sr_high > sr_low,
            "higher frequency should give higher annualized SR"
        );
    }

    #[test]
    fn test_sr_from_precision_invalid() {
        assert_eq!(sr_from_precision(0.0, 252.0, 1.0), 0.0);
        assert_eq!(sr_from_precision(1.0, 252.0, 1.0), 0.0);
        assert_eq!(sr_from_precision(0.5, 0.0, 1.0), 0.0);
        assert_eq!(sr_from_precision(0.5, 252.0, 0.0), 0.0);
        assert_eq!(sr_from_precision(-0.1, 252.0, 1.0), 0.0);
    }
}
