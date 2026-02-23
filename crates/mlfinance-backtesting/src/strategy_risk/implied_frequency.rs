//! Implied frequency from target Sharpe Ratio (Snippet 15.4).
//!
//! Given a target Sharpe Ratio, precision, and win/loss ratio, computes the
//! minimum betting frequency required.

/// Implied frequency from target SR and precision.
///
/// From the SR formula:
///   SR = SR_per_bet * sqrt(freq)
///   freq = (SR / SR_per_bet)^2
///
/// where SR_per_bet depends on precision and win/loss ratio.
///
/// # Arguments
/// * `target_sr` - Target annualized Sharpe Ratio.
/// * `precision` - Hit rate (probability of positive return) in `(0, 1)`.
/// * `avg_win_loss_ratio` - Ratio of average win to average loss (must be > 0).
///
/// # Returns
/// Required number of bets per year. Returns 0.0 if inputs are invalid or
/// SR per bet is zero (e.g., precision = 0.5 with 1:1 W/L and nonzero target).
pub fn implied_frequency(target_sr: f64, precision: f64, avg_win_loss_ratio: f64) -> f64 {
    if precision <= 0.0 || precision >= 1.0 || avg_win_loss_ratio <= 0.0 {
        return 0.0;
    }

    let p = precision;
    let r = avg_win_loss_ratio;

    // SR per bet
    let expected = p * r - (1.0 - p);
    let variance = p * (1.0 - p) * (r + 1.0).powi(2);

    if variance <= 0.0 {
        return 0.0;
    }

    let sr_per_bet = expected / variance.sqrt();

    if sr_per_bet.abs() < 1e-15 {
        // Cannot achieve any nonzero SR with this precision/W-L
        return if target_sr.abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        };
    }

    // freq = (target_sr / sr_per_bet)^2
    let freq = (target_sr / sr_per_bet).powi(2);

    if freq.is_finite() {
        freq
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy_risk::sr_from_precision::sr_from_precision;

    #[test]
    fn test_implied_frequency_roundtrip() {
        let original_freq = 252.0;
        let precision = 0.55;
        let wl_ratio = 1.5;

        let sr = sr_from_precision(precision, original_freq, wl_ratio);
        let recovered = implied_frequency(sr, precision, wl_ratio);

        assert!(
            (recovered - original_freq).abs() < 1.0,
            "recovered={} vs original={}",
            recovered,
            original_freq
        );
    }

    #[test]
    fn test_implied_frequency_higher_sr_needs_more() {
        let f1 = implied_frequency(1.0, 0.55, 1.0);
        let f2 = implied_frequency(2.0, 0.55, 1.0);
        assert!(
            f2 > f1,
            "higher SR should require higher frequency: {} vs {}",
            f2,
            f1
        );
    }

    #[test]
    fn test_implied_frequency_better_precision_needs_less() {
        let f1 = implied_frequency(1.0, 0.55, 1.0);
        let f2 = implied_frequency(1.0, 0.65, 1.0);
        assert!(
            f2 < f1,
            "higher precision should require less frequency: {} vs {}",
            f2,
            f1
        );
    }

    #[test]
    fn test_implied_frequency_invalid() {
        assert_eq!(implied_frequency(1.0, 0.0, 1.0), 0.0);
        assert_eq!(implied_frequency(1.0, 1.0, 1.0), 0.0);
        assert_eq!(implied_frequency(1.0, 0.5, 0.0), 0.0);
    }

    #[test]
    fn test_implied_frequency_zero_target() {
        let f = implied_frequency(0.0, 0.55, 1.5);
        assert!(
            f.abs() < 1e-10,
            "zero target SR should need zero frequency, got {}",
            f
        );
    }
}
