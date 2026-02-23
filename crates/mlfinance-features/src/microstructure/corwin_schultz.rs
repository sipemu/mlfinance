/// Corwin-Schultz bid-ask spread estimator from high-low prices.
///
/// Estimates the bid-ask spread using the relationship between high-low
/// price ranges over one-period and two-period intervals.
///
/// # Arguments
///
/// * `highs` - High prices for each period.
/// * `lows` - Low prices for each period.
///
/// # Returns
///
/// Vector of estimated spreads of length min(highs.len(), lows.len()) - 1.
/// Each value is non-negative. Returns an empty vector if fewer than 2 periods.
pub fn corwin_schultz_spread(highs: &[f64], lows: &[f64]) -> Vec<f64> {
    let n = highs.len().min(lows.len());
    if n < 2 {
        return vec![];
    }

    let mut spreads = Vec::with_capacity(n - 1);

    for t in 0..(n - 1) {
        let beta_val = beta(highs[t], lows[t], highs[t + 1], lows[t + 1]);

        // Two-day high-low
        let high_2d = highs[t].max(highs[t + 1]);
        let low_2d = lows[t].min(lows[t + 1]);
        let gamma_val = gamma(high_2d, low_2d);

        let alpha_val = alpha(beta_val, gamma_val);

        // Spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
        let spread = if alpha_val > 0.0 {
            let exp_a = alpha_val.exp();
            2.0 * (exp_a - 1.0) / (1.0 + exp_a)
        } else {
            0.0
        };

        spreads.push(spread.max(0.0));
    }

    spreads
}

/// Beta component of CS estimator.
///
/// beta = sum of squared log(high/low) for consecutive periods.
fn beta(high_t: f64, low_t: f64, high_t1: f64, low_t1: f64) -> f64 {
    if high_t <= 0.0 || low_t <= 0.0 || high_t1 <= 0.0 || low_t1 <= 0.0 {
        return 0.0;
    }
    let ln_hl_t = (high_t / low_t).ln();
    let ln_hl_t1 = (high_t1 / low_t1).ln();
    ln_hl_t.powi(2) + ln_hl_t1.powi(2)
}

/// Gamma component of CS estimator.
///
/// gamma = [ln(high_2d / low_2d)]^2
fn gamma(high_2d: f64, low_2d: f64) -> f64 {
    if high_2d <= 0.0 || low_2d <= 0.0 {
        return 0.0;
    }
    (high_2d / low_2d).ln().powi(2)
}

/// Alpha from beta and gamma.
///
/// alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
fn alpha(beta_val: f64, gamma_val: f64) -> f64 {
    let sqrt2 = 2.0_f64.sqrt();
    let denom = 3.0 - 2.0 * sqrt2;

    if denom.abs() < 1e-15 || beta_val < 0.0 {
        return 0.0;
    }

    let term1 = ((2.0 * beta_val).sqrt() - beta_val.sqrt()) / denom;
    let term2 = (gamma_val / denom).max(0.0).sqrt();

    term1 - term2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_computation() {
        let b = beta(105.0, 100.0, 106.0, 101.0);
        assert!(b > 0.0);
    }

    #[test]
    fn test_gamma_computation() {
        let g = gamma(106.0, 100.0);
        let expected = (106.0_f64 / 100.0).ln().powi(2);
        assert!((g - expected).abs() < 1e-10);
    }

    #[test]
    fn test_corwin_schultz_basic() {
        let highs = vec![105.0, 106.0, 104.0, 107.0, 105.0];
        let lows = vec![100.0, 101.0, 99.0, 102.0, 100.0];
        let spreads = corwin_schultz_spread(&highs, &lows);
        assert_eq!(spreads.len(), 4);
        for &s in &spreads {
            assert!(s >= 0.0);
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_corwin_schultz_short() {
        let spreads = corwin_schultz_spread(&[105.0], &[100.0]);
        assert!(spreads.is_empty());
    }

    #[test]
    fn test_alpha_computation() {
        let a = alpha(0.001, 0.002);
        assert!(a.is_finite());
    }

    #[test]
    fn test_beta_zero_prices() {
        let b = beta(0.0, 100.0, 105.0, 100.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_gamma_zero_prices() {
        let g = gamma(0.0, 100.0);
        assert_eq!(g, 0.0);
    }
}
