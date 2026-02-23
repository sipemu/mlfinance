//! Daily volatility via EWMA standard deviation (Snippet 3.1).
//!
//! Computes daily volatility as the exponentially weighted moving average
//! of the standard deviation of log returns.

use mlfinance_core::math;
use mlfinance_core::types::Timestamp;

/// Compute daily volatility as EWMA std of log returns.
///
/// Given a price series and corresponding timestamps, computes log returns
/// and then applies an EWMA standard deviation estimator with the given span.
///
/// # Arguments
/// * `prices` - Slice of closing prices (must have length >= 2).
/// * `timestamps` - Slice of timestamps corresponding to each price.
///   Must have the same length as `prices`. Currently used for alignment
///   purposes; the EWMA span is specified in number of observations.
/// * `span` - EWMA span in number of observations (e.g., 100).
///
/// # Returns
/// A vector of daily volatility estimates. The first element corresponds
/// to the first price (volatility is 0.0 there since no return exists).
/// Length equals `prices.len()`.
///
/// Returns an empty vector if `prices` has fewer than 2 elements.
pub fn daily_volatility(prices: &[f64], timestamps: &[Timestamp], span: usize) -> Vec<f64> {
    let _ = timestamps; // timestamps reserved for future date-aware windowing

    if prices.len() < 2 {
        return vec![0.0; prices.len()];
    }

    let log_ret = math::log_returns(prices);

    // ewma_std expects at least 2 values; log_ret has prices.len() - 1 elements
    match math::ewma_std(&log_ret, span) {
        Ok(ewma_vol) => {
            // Prepend a 0.0 for the first price (no return available)
            let mut result = Vec::with_capacity(prices.len());
            result.push(0.0);
            result.extend_from_slice(&ewma_vol);
            result
        }
        Err(_) => {
            // If ewma_std fails (e.g., insufficient data), return zeros
            vec![0.0; prices.len()]
        }
    }
}

use mlfinance_core::error::{MlFinanceError, Result};
use mlfinance_core::types::OhlcvBar;
use std::f64::consts::LN_2;

/// Parkinson volatility estimator using high-low range.
///
/// More efficient than close-to-close as it captures intraday range.
///
/// ```text
/// sigma^2 = (1 / (4 * n * ln(2))) * sum(ln(H_i / L_i)^2)
/// ```
///
/// # Arguments
///
/// * `bars` - Slice of OHLCV bars.
/// * `window` - Rolling window size in number of bars.
///
/// # Returns
///
/// A vector of rolling Parkinson volatility estimates with the same length
/// as `bars`. Indices `< window - 1` contain `NaN`.
///
/// # Errors
///
/// Returns [`MlFinanceError::EmptySeries`] if `bars` is empty,
/// [`MlFinanceError::InvalidParameter`] if `window == 0`, or
/// [`MlFinanceError::InsufficientData`] if `window > bars.len()`.
pub fn parkinson_volatility(bars: &[OhlcvBar], window: usize) -> Result<Vec<f64>> {
    if bars.is_empty() {
        return Err(MlFinanceError::EmptySeries);
    }
    if window == 0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "window must be >= 1".to_string(),
        });
    }
    if window > bars.len() {
        return Err(MlFinanceError::InsufficientData {
            expected: window,
            actual: bars.len(),
        });
    }

    let n = bars.len();
    let mut result = vec![f64::NAN; n];

    // Precompute ln(H/L)^2 for each bar
    let hl_sq: Vec<f64> = bars
        .iter()
        .map(|b| {
            let ratio = (b.high / b.low).ln();
            ratio * ratio
        })
        .collect();

    // Compute rolling sum
    let mut sum: f64 = hl_sq[..window].iter().sum();
    let coeff = 1.0 / (4.0 * window as f64 * LN_2);
    result[window - 1] = (coeff * sum).sqrt();

    for i in window..n {
        sum += hl_sq[i] - hl_sq[i - window];
        result[i] = (coeff * sum).sqrt();
    }

    Ok(result)
}

/// Garman-Klass volatility estimator using OHLC prices.
///
/// More efficient than Parkinson as it also uses open/close information.
///
/// ```text
/// sigma^2 = (1/n) * sum( 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2 )
/// ```
///
/// # Arguments
///
/// * `bars` - Slice of OHLCV bars.
/// * `window` - Rolling window size in number of bars.
///
/// # Returns
///
/// A vector of rolling Garman-Klass volatility estimates with the same
/// length as `bars`. Indices `< window - 1` contain `NaN`.
///
/// # Errors
///
/// Returns [`MlFinanceError::EmptySeries`] if `bars` is empty,
/// [`MlFinanceError::InvalidParameter`] if `window == 0`, or
/// [`MlFinanceError::InsufficientData`] if `window > bars.len()`.
pub fn garman_klass_volatility(bars: &[OhlcvBar], window: usize) -> Result<Vec<f64>> {
    if bars.is_empty() {
        return Err(MlFinanceError::EmptySeries);
    }
    if window == 0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "window must be >= 1".to_string(),
        });
    }
    if window > bars.len() {
        return Err(MlFinanceError::InsufficientData {
            expected: window,
            actual: bars.len(),
        });
    }

    let n = bars.len();
    let mut result = vec![f64::NAN; n];
    let co_coeff = 2.0 * LN_2 - 1.0;

    // Precompute per-bar GK terms
    let gk_terms: Vec<f64> = bars
        .iter()
        .map(|b| {
            let hl = (b.high / b.low).ln();
            let co = (b.close / b.open).ln();
            0.5 * hl * hl - co_coeff * co * co
        })
        .collect();

    // Rolling sum
    let mut sum: f64 = gk_terms[..window].iter().sum();
    let variance = sum / window as f64;
    result[window - 1] = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    for i in window..n {
        sum += gk_terms[i] - gk_terms[i - window];
        let variance = sum / window as f64;
        result[i] = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    }

    Ok(result)
}

/// Compute sample variance of a slice. Returns 0 if length <= 1.
fn sample_variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// Compute Yang-Zhang volatility for a single window.
fn compute_yz_window(
    overnight: &[f64],
    cc_ret: &[f64],
    rs_term: &[f64],
    k: f64,
    window: usize,
) -> f64 {
    let sigma_overnight_sq = sample_variance(overnight);
    let sigma_close_sq = sample_variance(cc_ret);
    let sigma_rs_sq = rs_term.iter().sum::<f64>() / window as f64;
    let sigma_yz_sq = sigma_overnight_sq + k * sigma_close_sq + (1.0 - k) * sigma_rs_sq;
    if sigma_yz_sq > 0.0 {
        sigma_yz_sq.sqrt()
    } else {
        0.0
    }
}

/// Validate Yang-Zhang input parameters.
fn validate_yz_inputs(bars_len: usize, window: usize) -> Result<()> {
    if bars_len == 0 {
        return Err(MlFinanceError::EmptySeries);
    }
    if window == 0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "window must be >= 1".to_string(),
        });
    }
    if window > bars_len {
        return Err(MlFinanceError::InsufficientData {
            expected: window,
            actual: bars_len,
        });
    }
    Ok(())
}

/// Yang-Zhang volatility estimator, the most efficient among OHLC estimators.
///
/// Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
/// components:
///
/// ```text
/// sigma_yz^2 = sigma_overnight^2 + k * sigma_close^2 + (1-k) * sigma_rs^2
/// ```
///
/// where `k = 0.34 / (1.34 + (n+1)/(n-1))` and
/// `sigma_rs^2 = (1/n) * sum( ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C) )`.
///
/// Yang-Zhang needs a previous close for overnight returns, so the
/// effective window starts one bar earlier. Indices `< window` are `NaN`.
///
/// # Arguments
///
/// * `bars` - Slice of OHLCV bars.
/// * `window` - Rolling window size in number of bars.
///
/// # Returns
///
/// A vector of rolling Yang-Zhang volatility estimates with the same length
/// as `bars`. Indices `< window` contain `NaN`.
///
/// # Errors
///
/// Returns [`MlFinanceError::EmptySeries`] if `bars` is empty,
/// [`MlFinanceError::InvalidParameter`] if `window == 0`, or
/// [`MlFinanceError::InsufficientData`] if `window > bars.len()`.
pub fn yang_zhang_volatility(bars: &[OhlcvBar], window: usize) -> Result<Vec<f64>> {
    validate_yz_inputs(bars.len(), window)?;

    let n = bars.len();
    let mut result = vec![f64::NAN; n];

    // We need previous close for overnight returns, so the first bar with a
    // valid overnight return is index 1. For a window of size `window`, the
    // first complete output is at index `window` (window overnight returns
    // from indices 1..=window).

    if n < window + 1 {
        // Not enough data to compute even one complete window with overnight returns
        return Ok(result);
    }

    // Precompute per-bar values (only valid from index 1 onward):
    // overnight return: o_i = ln(Open_i / Close_{i-1})
    // close-to-close return: c_i = ln(Close_i / Close_{i-1})
    // Rogers-Satchell term: rs_i = ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)
    let mut overnight = vec![0.0; n]; // index 0 unused
    let mut cc_ret = vec![0.0; n]; // index 0 unused
    let mut rs_term = vec![0.0; n];

    for i in 1..n {
        overnight[i] = (bars[i].open / bars[i - 1].close).ln();
        cc_ret[i] = (bars[i].close / bars[i - 1].close).ln();
    }

    for i in 0..n {
        let ho = (bars[i].high / bars[i].open).ln();
        let hc = (bars[i].high / bars[i].close).ln();
        let lo = (bars[i].low / bars[i].open).ln();
        let lc = (bars[i].low / bars[i].close).ln();
        rs_term[i] = ho * hc + lo * lc;
    }

    let k = if window == 1 {
        0.0
    } else {
        0.34 / (1.34 + (window as f64 + 1.0) / (window as f64 - 1.0))
    };

    for (offset, result_val) in result[window..].iter_mut().enumerate() {
        let start = offset + 1;
        let end = offset + window + 1;
        *result_val = compute_yz_window(
            &overnight[start..end],
            &cc_ret[start..end],
            &rs_term[start..end],
            k,
            window,
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_timestamps(n: usize) -> Vec<Timestamp> {
        (0..n).map(|_| Utc::now()).collect()
    }

    #[test]
    fn test_daily_volatility_basic() {
        let prices = vec![100.0, 101.0, 102.0, 100.5, 103.0, 101.0];
        let ts = make_timestamps(prices.len());
        let vol = daily_volatility(&prices, &ts, 3);
        assert_eq!(vol.len(), prices.len());
        // First element should be 0
        assert_eq!(vol[0], 0.0);
        // Subsequent elements should be non-negative
        for v in &vol[1..] {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_daily_volatility_constant_prices() {
        let prices = vec![100.0, 100.0, 100.0, 100.0];
        let ts = make_timestamps(prices.len());
        let vol = daily_volatility(&prices, &ts, 3);
        assert_eq!(vol.len(), 4);
        // With constant prices, log returns are 0 => volatility is 0
        for v in &vol {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_daily_volatility_single_price() {
        let prices = vec![100.0];
        let ts = make_timestamps(1);
        let vol = daily_volatility(&prices, &ts, 3);
        assert_eq!(vol.len(), 1);
        assert_eq!(vol[0], 0.0);
    }

    #[test]
    fn test_daily_volatility_empty() {
        let prices: Vec<f64> = vec![];
        let ts: Vec<Timestamp> = vec![];
        let vol = daily_volatility(&prices, &ts, 3);
        assert!(vol.is_empty());
    }

    #[test]
    fn test_daily_volatility_increasing_prices() {
        // Steadily increasing prices should produce positive volatility
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let ts = make_timestamps(prices.len());
        let vol = daily_volatility(&prices, &ts, 5);
        assert_eq!(vol.len(), 20);
        // Later volatilities should be positive
        assert!(vol[5] > 0.0);
    }

    // ---- Helper to build OhlcvBar slices for new estimators ----

    fn make_bar(open: f64, high: f64, low: f64, close: f64) -> OhlcvBar {
        OhlcvBar {
            timestamp: Utc::now(),
            open,
            high,
            low,
            close,
            volume: 1000.0,
            vwap: (high + low + close) / 3.0,
        }
    }

    fn make_constant_bars(n: usize, price: f64) -> Vec<OhlcvBar> {
        (0..n)
            .map(|_| make_bar(price, price, price, price))
            .collect()
    }

    /// Bars with small intraday range (low volatility)
    fn make_low_vol_bars() -> Vec<OhlcvBar> {
        vec![
            make_bar(100.0, 100.5, 99.5, 100.2),
            make_bar(100.2, 100.7, 99.8, 100.3),
            make_bar(100.3, 100.6, 99.9, 100.1),
            make_bar(100.1, 100.4, 99.7, 100.0),
            make_bar(100.0, 100.3, 99.6, 100.2),
            make_bar(100.2, 100.5, 99.8, 100.1),
        ]
    }

    /// Bars with large intraday range (high volatility)
    fn make_high_vol_bars() -> Vec<OhlcvBar> {
        vec![
            make_bar(100.0, 105.0, 95.0, 103.0),
            make_bar(103.0, 108.0, 97.0, 99.0),
            make_bar(99.0, 106.0, 93.0, 104.0),
            make_bar(104.0, 110.0, 96.0, 97.0),
            make_bar(97.0, 104.0, 91.0, 102.0),
            make_bar(102.0, 107.0, 94.0, 98.0),
        ]
    }

    // ============================================================
    // Parkinson Volatility Tests
    // ============================================================

    #[test]
    fn test_parkinson_basic() {
        let bars = make_low_vol_bars();
        let vol = parkinson_volatility(&bars, 3).unwrap();
        assert_eq!(vol.len(), bars.len());
        // First two elements should be NaN
        assert!(vol[0].is_nan());
        assert!(vol[1].is_nan());
        // Remaining should be positive finite
        for v in &vol[2..] {
            assert!(v.is_finite());
            assert!(*v > 0.0);
        }
    }

    #[test]
    fn test_parkinson_window_1() {
        let bars = make_low_vol_bars();
        let vol = parkinson_volatility(&bars, 1).unwrap();
        assert_eq!(vol.len(), bars.len());
        // All values should be valid (no leading NaNs)
        for v in &vol {
            assert!(v.is_finite());
            assert!(*v > 0.0);
        }
        // Manually verify first element: sqrt( ln(100.5/99.5)^2 / (4*1*ln2) )
        let ratio = (100.5_f64 / 99.5).ln();
        let expected = (ratio * ratio / (4.0 * LN_2)).sqrt();
        assert!((vol[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parkinson_constant_prices() {
        let bars = make_constant_bars(5, 100.0);
        let vol = parkinson_volatility(&bars, 3).unwrap();
        // With H == L, ln(H/L) = 0 => vol = 0
        // (NaN check for first two, then ~0 for the rest)
        // Actually ln(100/100)=0, so result is 0.0 for window positions
        for v in &vol[2..] {
            assert!(v.is_finite());
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_parkinson_high_vs_low_vol() {
        let low_vol = parkinson_volatility(&make_low_vol_bars(), 3).unwrap();
        let high_vol = parkinson_volatility(&make_high_vol_bars(), 3).unwrap();
        // Compare at the last index where both are valid
        let idx = 5;
        assert!(high_vol[idx] > low_vol[idx]);
    }

    #[test]
    fn test_parkinson_empty() {
        let bars: Vec<OhlcvBar> = vec![];
        let err = parkinson_volatility(&bars, 3);
        assert!(err.is_err());
    }

    #[test]
    fn test_parkinson_window_zero() {
        let bars = make_low_vol_bars();
        let err = parkinson_volatility(&bars, 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_parkinson_window_exceeds_len() {
        let bars = make_low_vol_bars();
        let err = parkinson_volatility(&bars, 100);
        assert!(err.is_err());
    }

    // ============================================================
    // Garman-Klass Volatility Tests
    // ============================================================

    #[test]
    fn test_garman_klass_basic() {
        let bars = make_low_vol_bars();
        let vol = garman_klass_volatility(&bars, 3).unwrap();
        assert_eq!(vol.len(), bars.len());
        assert!(vol[0].is_nan());
        assert!(vol[1].is_nan());
        for v in &vol[2..] {
            assert!(v.is_finite());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_garman_klass_window_1() {
        let bars = vec![make_bar(100.0, 105.0, 96.0, 102.0)];
        let vol = garman_klass_volatility(&bars, 1).unwrap();
        assert_eq!(vol.len(), 1);
        assert!(vol[0].is_finite());
        // Manual calculation
        let hl = (105.0_f64 / 96.0).ln();
        let co = (102.0_f64 / 100.0).ln();
        let gk_var = 0.5 * hl * hl - (2.0 * LN_2 - 1.0) * co * co;
        let expected = if gk_var > 0.0 { gk_var.sqrt() } else { 0.0 };
        assert!((vol[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_garman_klass_constant_prices() {
        let bars = make_constant_bars(5, 50.0);
        let vol = garman_klass_volatility(&bars, 3).unwrap();
        for v in &vol[2..] {
            assert!(v.is_finite());
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_garman_klass_high_vs_low_vol() {
        let low_vol = garman_klass_volatility(&make_low_vol_bars(), 3).unwrap();
        let high_vol = garman_klass_volatility(&make_high_vol_bars(), 3).unwrap();
        let idx = 5;
        assert!(high_vol[idx] > low_vol[idx]);
    }

    #[test]
    fn test_garman_klass_empty() {
        let bars: Vec<OhlcvBar> = vec![];
        assert!(garman_klass_volatility(&bars, 3).is_err());
    }

    #[test]
    fn test_garman_klass_window_zero() {
        let bars = make_low_vol_bars();
        assert!(garman_klass_volatility(&bars, 0).is_err());
    }

    #[test]
    fn test_garman_klass_window_exceeds_len() {
        let bars = make_low_vol_bars();
        assert!(garman_klass_volatility(&bars, 100).is_err());
    }

    // ============================================================
    // Yang-Zhang Volatility Tests
    // ============================================================

    #[test]
    fn test_yang_zhang_basic() {
        let bars = make_low_vol_bars();
        let vol = yang_zhang_volatility(&bars, 3).unwrap();
        assert_eq!(vol.len(), bars.len());
        // First `window` indices should be NaN
        for v in &vol[..3] {
            assert!(v.is_nan());
        }
        // Remaining should be finite and non-negative
        for v in &vol[3..] {
            assert!(v.is_finite());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_yang_zhang_window_1() {
        // Window 1 is a special case: k=0, overnight/close variance = 0
        // sigma_yz^2 = sigma_rs^2 for the single bar
        let bars = make_high_vol_bars();
        let vol = yang_zhang_volatility(&bars, 1).unwrap();
        assert_eq!(vol.len(), bars.len());
        // Index 0 should be NaN (no previous close for overnight return)
        assert!(vol[0].is_nan());
        // Index 1 onward should be finite
        for v in &vol[1..] {
            assert!(v.is_finite());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_yang_zhang_constant_prices() {
        let bars = make_constant_bars(6, 100.0);
        let vol = yang_zhang_volatility(&bars, 3).unwrap();
        // All computed values should be ~0
        for v in &vol[3..] {
            assert!(v.is_finite());
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_yang_zhang_high_vs_low_vol() {
        let low_vol = yang_zhang_volatility(&make_low_vol_bars(), 3).unwrap();
        let high_vol = yang_zhang_volatility(&make_high_vol_bars(), 3).unwrap();
        let idx = 5;
        assert!(high_vol[idx] > low_vol[idx]);
    }

    #[test]
    fn test_yang_zhang_empty() {
        let bars: Vec<OhlcvBar> = vec![];
        assert!(yang_zhang_volatility(&bars, 3).is_err());
    }

    #[test]
    fn test_yang_zhang_window_zero() {
        let bars = make_low_vol_bars();
        assert!(yang_zhang_volatility(&bars, 0).is_err());
    }

    #[test]
    fn test_yang_zhang_window_exceeds_len() {
        let bars = make_low_vol_bars();
        assert!(yang_zhang_volatility(&bars, 100).is_err());
    }

    #[test]
    fn test_yang_zhang_output_length() {
        let bars = make_low_vol_bars();
        let vol = yang_zhang_volatility(&bars, 4).unwrap();
        assert_eq!(vol.len(), bars.len());
        // NaN for indices 0..4, valid for 4..6
        for v in &vol[..4] {
            assert!(v.is_nan());
        }
        for v in &vol[4..] {
            assert!(v.is_finite());
        }
    }

    // ============================================================
    // Cross-estimator comparison tests
    // ============================================================

    #[test]
    fn test_all_estimators_consistent_sign() {
        // All three should produce non-negative volatilities for non-degenerate bars
        let bars = make_high_vol_bars();
        let pk = parkinson_volatility(&bars, 3).unwrap();
        let gk = garman_klass_volatility(&bars, 3).unwrap();
        let yz = yang_zhang_volatility(&bars, 3).unwrap();

        for v in &pk[2..] {
            assert!(*v >= 0.0);
        }
        for v in &gk[2..] {
            assert!(*v >= 0.0);
        }
        for v in &yz[3..] {
            assert!(*v >= 0.0);
        }
    }
}
