//! Implied precision from target Sharpe Ratio (Snippet 15.3).
//!
//! Given a target Sharpe Ratio, betting frequency, and win/loss ratio,
//! computes the minimum hit rate (precision) required.

/// Compute the Sharpe ratio per bet as a function of precision p and win/loss ratio r.
fn sr_per_bet(p: f64, r: f64) -> f64 {
    let expected = p * r - (1.0 - p);
    let variance = p * (1.0 - p) * (r + 1.0).powi(2);
    if variance <= 0.0 {
        0.0
    } else {
        expected / variance.sqrt()
    }
}

/// Find root of `f(x) = target` via bisection on [lo, hi].
fn bisect(target: f64, r: f64, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let val = sr_per_bet(mid, r);
        if (val - target).abs() < 1e-12 {
            return mid;
        }
        if val < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Implied precision from target SR and frequency.
///
/// Solves for `p` in the SR equation:
///   SR = (p * R - (1-p)) / sqrt(p * (1-p) * (R+1)^2) * sqrt(freq)
///
/// Uses a bisection method since the equation is not analytically invertible
/// in the general case.
///
/// # Arguments
/// * `target_sr` - Target annualized Sharpe Ratio.
/// * `freq` - Number of bets per year.
/// * `avg_win_loss_ratio` - Ratio of average win to average loss (must be > 0).
///
/// # Returns
/// Required precision in `(0, 1)`. Returns 0.0 if inputs are invalid or no
/// solution exists.
pub fn implied_precision(target_sr: f64, freq: f64, avg_win_loss_ratio: f64) -> f64 {
    if freq <= 0.0 || avg_win_loss_ratio <= 0.0 {
        return 0.0;
    }

    let r = avg_win_loss_ratio;
    let target = target_sr / freq.sqrt();
    let eps = 1e-10;

    let sr_lo = sr_per_bet(eps, r);
    let sr_hi = sr_per_bet(1.0 - eps, r);

    if target < sr_lo {
        return eps;
    }
    if target > sr_hi {
        return 0.0;
    }

    bisect(target, r, eps, 1.0 - eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy_risk::sr_from_precision::sr_from_precision;

    #[test]
    fn test_implied_precision_roundtrip() {
        let original_precision = 0.55;
        let freq = 252.0;
        let wl_ratio = 1.5;

        let sr = sr_from_precision(original_precision, freq, wl_ratio);
        let recovered = implied_precision(sr, freq, wl_ratio);

        assert!(
            (recovered - original_precision).abs() < 1e-4,
            "recovered={} vs original={}",
            recovered,
            original_precision
        );
    }

    #[test]
    fn test_implied_precision_zero_sr() {
        // SR = 0 with 1:1 W/L -> precision = 0.5
        let p = implied_precision(0.0, 252.0, 1.0);
        assert!(
            (p - 0.5).abs() < 1e-4,
            "p={} should be ~0.5 for zero SR with 1:1",
            p
        );
    }

    #[test]
    fn test_implied_precision_higher_sr_needs_higher_precision() {
        let p1 = implied_precision(1.0, 252.0, 1.0);
        let p2 = implied_precision(2.0, 252.0, 1.0);
        assert!(
            p2 > p1,
            "higher SR should require higher precision: {} vs {}",
            p2,
            p1
        );
    }

    #[test]
    fn test_implied_precision_invalid() {
        assert_eq!(implied_precision(1.0, 0.0, 1.0), 0.0);
        assert_eq!(implied_precision(1.0, 252.0, 0.0), 0.0);
    }
}
