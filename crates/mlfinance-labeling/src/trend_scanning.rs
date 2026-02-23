//! Trend scanning labeling method (de Prado, AFML Chapter 3.5).
//!
//! For each observation, scans forward over a range of window lengths,
//! fits a simple linear regression of price on time, and selects the
//! window whose slope has the highest absolute t-statistic. The sign
//! of that t-statistic determines the trend label.

use mlfinance_core::error::{MlFinanceError, Result};

/// Result for a single observation's trend scan.
#[derive(Debug, Clone)]
pub struct TrendScanResult {
    /// The t-statistic of the best-fit linear trend.
    pub t_stat: f64,
    /// The label: 1 for uptrend, -1 for downtrend, 0 if no significant trend.
    pub label: i32,
    /// The window length that produced the best fit.
    pub best_window: usize,
    /// R-squared of the best fit.
    pub r_squared: f64,
}

/// Internal: fit simple OLS y = a + b * x where x = [0, 1, ..., n-1] and return (t_stat, r_squared).
///
/// Uses closed-form expressions for the regression on equally spaced x values:
/// - x_mean = (n-1) / 2
/// - Sxx = n * (n^2 - 1) / 12
/// - b = Sxy / Sxx
/// - t_stat = b / se(b)
/// - R^2 = 1 - SSR / SS_tot
///
/// Requires `y.len() >= 3` (need at least 3 points for a meaningful regression
/// with n - 2 degrees of freedom).
fn ols_t_stat(y: &[f64]) -> (f64, f64) {
    let n = y.len();
    debug_assert!(n >= 3);

    let nf = n as f64;

    // x = [0, 1, ..., n-1]
    let x_mean = (nf - 1.0) / 2.0;

    // Sxx = sum((x_i - x_mean)^2) = n*(n^2 - 1)/12
    let sxx = nf * (nf * nf - 1.0) / 12.0;

    // y_mean
    let y_sum: f64 = y.iter().sum();
    let y_mean = y_sum / nf;

    // Sxy = sum((x_i - x_mean) * (y_i - y_mean))
    let mut sxy = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        sxy += (i as f64 - x_mean) * (yi - y_mean);
    }

    // Slope
    let beta = sxy / sxx;

    // Intercept
    let alpha = y_mean - beta * x_mean;

    // Residual sum of squares (SSR) and total sum of squares (SS_tot)
    let mut ssr = 0.0;
    let mut ss_tot = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        let predicted = alpha + beta * i as f64;
        let residual = yi - predicted;
        ssr += residual * residual;
        ss_tot += (yi - y_mean) * (yi - y_mean);
    }

    // R-squared
    let r_squared = if ss_tot.abs() < f64::EPSILON {
        // All y values are identical; perfect "fit" but slope is zero
        1.0
    } else {
        1.0 - ssr / ss_tot
    };

    // Standard error of beta: se(b) = sqrt(SSR / ((n-2) * Sxx))
    let dof = nf - 2.0;
    if dof <= 0.0 || sxx.abs() < f64::EPSILON {
        return (0.0, r_squared);
    }

    let se_beta_sq = ssr / (dof * sxx);
    if se_beta_sq <= 0.0 {
        // Perfect fit (SSR == 0) means the t-stat is conceptually infinite.
        // Return a large finite value with the correct sign.
        let t_stat = if beta.abs() < f64::EPSILON {
            0.0
        } else {
            beta.signum() * f64::MAX
        };
        return (t_stat, r_squared);
    }

    let se_beta = se_beta_sq.sqrt();
    let t_stat = beta / se_beta;

    (t_stat, r_squared)
}

/// Scan a single event index for the best-fitting trend window.
fn scan_single_event(
    prices: &[f64],
    t: usize,
    min_w: usize,
    effective_max: usize,
) -> TrendScanResult {
    let mut best_abs_t = f64::NEG_INFINITY;
    let mut best_t_stat = 0.0;
    let mut best_r2 = 0.0;
    let mut best_w = min_w;

    for w in min_w..=effective_max {
        let (t_stat, r2) = ols_t_stat(&prices[t..t + w]);
        let abs_t = t_stat.abs();
        if abs_t > best_abs_t {
            best_abs_t = abs_t;
            best_t_stat = t_stat;
            best_r2 = r2;
            best_w = w;
        }
    }

    let label = if best_t_stat > 0.0 {
        1
    } else if best_t_stat < 0.0 {
        -1
    } else {
        0
    };
    TrendScanResult {
        t_stat: best_t_stat,
        label,
        best_window: best_w,
        r_squared: best_r2,
    }
}

/// Validate trend scanning parameters.
fn validate_trend_params(min_w: usize, max_window: usize, prices_len: usize) -> Result<()> {
    if min_w < 3 {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!("min_window must be at least 3, got {}", min_w),
        });
    }
    if max_window < min_w {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!(
                "max_window ({}) must be >= min_window ({})",
                max_window, min_w
            ),
        });
    }
    if prices_len < min_w {
        return Err(MlFinanceError::InsufficientData {
            expected: min_w,
            actual: prices_len,
        });
    }
    if max_window > prices_len {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!(
                "max_window ({}) must be <= prices.len() ({})",
                max_window, prices_len
            ),
        });
    }
    Ok(())
}

/// Scan for trends at each observation point.
///
/// For each index `t` in `t_events`, looks forward over windows of length
/// `[min_window..=max_window]`, fits an OLS regression of price on time,
/// and selects the window whose slope has the highest absolute t-statistic.
/// The sign of that t-statistic determines the trend label.
///
/// # Arguments
///
/// * `prices` - Price series (e.g., close prices).
/// * `t_events` - Optional slice of observation indices to scan. If `None`,
///   scans all indices from `0` through `prices.len() - min_window`.
/// * `max_window` - Maximum forward-looking window length. Must be
///   `<= prices.len()`.
/// * `min_window` - Minimum forward-looking window length. Defaults to `3`
///   if `None`. Must be `>= 3`.
///
/// # Returns
///
/// A vector of [`TrendScanResult`] structs, one per event index.
///
/// # Errors
///
/// Returns [`MlFinanceError::InvalidParameter`] if `min_window < 3` or
/// `max_window < min_window`. Returns [`MlFinanceError::InsufficientData`]
/// if `prices.len() < min_window`. Returns
/// [`MlFinanceError::IndexOutOfBounds`] if a provided event index leaves
/// fewer than `min_window` remaining observations.
pub fn trend_scanning_labels(
    prices: &[f64],
    t_events: Option<&[usize]>,
    max_window: usize,
    min_window: Option<usize>,
) -> Result<Vec<TrendScanResult>> {
    let min_w = min_window.unwrap_or(3);
    validate_trend_params(min_w, max_window, prices.len())?;

    let default_events: Vec<usize>;
    let events = match t_events {
        Some(ev) => ev,
        None => {
            default_events = (0..=prices.len() - min_w).collect();
            &default_events
        }
    };

    let mut results = Vec::with_capacity(events.len());
    for &t in events {
        let remaining = prices.len() - t;
        if remaining < min_w {
            return Err(MlFinanceError::IndexOutOfBounds {
                index: t,
                length: prices.len(),
            });
        }
        results.push(scan_single_event(
            prices,
            t,
            min_w,
            max_window.min(remaining),
        ));
    }

    Ok(results)
}

/// Convenience: get just the labels as a `Vec<i32>` from trend scanning.
///
/// Calls [`trend_scanning_labels`] with `t_events = None` and `min_window = None`
/// (default of 3), then extracts only the labels.
///
/// # Arguments
///
/// * `prices` - Price series (close prices).
/// * `max_window` - Maximum forward-looking window length.
///
/// # Returns
///
/// A vector of labels (`1` for uptrend, `-1` for downtrend, `0` for flat).
///
/// # Errors
///
/// Propagates any error from [`trend_scanning_labels`].
pub fn trend_scanning_label_series(prices: &[f64], max_window: usize) -> Result<Vec<i32>> {
    let results = trend_scanning_labels(prices, None, max_window, None)?;
    Ok(results.iter().map(|r| r.label).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strong_uptrend() {
        // Monotonically increasing prices: all labels should be 1
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let results = trend_scanning_labels(&prices, None, 10, None).unwrap();
        for r in &results {
            assert_eq!(r.label, 1, "expected uptrend label, got {}", r.label);
            assert!(r.t_stat > 0.0, "expected positive t-stat for uptrend");
        }
    }

    #[test]
    fn test_strong_downtrend() {
        // Monotonically decreasing prices: all labels should be -1
        let prices: Vec<f64> = (0..20).map(|i| 200.0 - i as f64).collect();
        let results = trend_scanning_labels(&prices, None, 10, None).unwrap();
        for r in &results {
            assert_eq!(r.label, -1, "expected downtrend label, got {}", r.label);
            assert!(r.t_stat < 0.0, "expected negative t-stat for downtrend");
        }
    }

    #[test]
    fn test_flat_prices() {
        // All prices identical: t-stats should be zero, labels should be 0
        let prices = vec![50.0; 20];
        let results = trend_scanning_labels(&prices, None, 10, None).unwrap();
        for r in &results {
            assert_eq!(r.label, 0, "expected flat label for constant prices");
            // t-stat should be exactly 0 (or very near) since beta = 0
            assert!(
                r.t_stat.abs() < 1e-10,
                "expected near-zero t-stat, got {}",
                r.t_stat
            );
        }
    }

    #[test]
    fn test_v_shape_transition() {
        // V-shape: goes down then up. Labels should transition from -1 to 1.
        let mut prices = Vec::new();
        for i in 0..10 {
            prices.push(100.0 - i as f64 * 2.0); // down from 100 to 82
        }
        for i in 0..10 {
            prices.push(82.0 + i as f64 * 2.0); // up from 82 to 100
        }

        let results = trend_scanning_labels(&prices, None, 5, None).unwrap();

        // Early points (purely in the downtrend) should be labeled -1
        assert_eq!(results[0].label, -1, "early in downtrend should be -1");
        assert_eq!(results[1].label, -1, "early in downtrend should be -1");

        // Points well into the uptrend should be labeled 1
        let last_idx = results.len() - 1;
        assert_eq!(results[last_idx].label, 1, "late in uptrend should be 1");
    }

    #[test]
    fn test_too_few_prices() {
        // Fewer than min_window (3) prices should error
        let prices = vec![1.0, 2.0];
        let result = trend_scanning_labels(&prices, None, 3, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_min_window_greater_than_max_window() {
        let prices: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = trend_scanning_labels(&prices, None, 3, Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_min_window_too_small() {
        let prices: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = trend_scanning_labels(&prices, None, 10, Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_t_events() {
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let events = vec![0, 5, 10];
        let results = trend_scanning_labels(&prices, Some(&events), 10, None).unwrap();
        assert_eq!(results.len(), 3, "should produce one result per event");
        for r in &results {
            assert_eq!(r.label, 1, "uptrend should give label 1");
        }
    }

    #[test]
    fn test_r_squared_in_range() {
        // R-squared should be in [0, 1] for all results
        let prices: Vec<f64> = (0..30)
            .map(|i| 100.0 + (i as f64).sin() * 5.0 + i as f64 * 0.5)
            .collect();
        let results = trend_scanning_labels(&prices, None, 10, None).unwrap();
        for (idx, r) in results.iter().enumerate() {
            assert!(
                r.r_squared >= -1e-10 && r.r_squared <= 1.0 + 1e-10,
                "R-squared out of range at index {}: {}",
                idx,
                r.r_squared
            );
        }
    }

    #[test]
    fn test_best_window_in_range() {
        let prices: Vec<f64> = (0..25).map(|i| 50.0 + i as f64 * 0.3).collect();
        let min_w = 4;
        let max_w = 8;
        let results = trend_scanning_labels(&prices, None, max_w, Some(min_w)).unwrap();
        for (idx, r) in results.iter().enumerate() {
            assert!(
                r.best_window >= min_w && r.best_window <= max_w,
                "best_window out of range at index {}: {} (expected [{}, {}])",
                idx,
                r.best_window,
                min_w,
                max_w
            );
        }
    }

    #[test]
    fn test_known_values_linear_series() {
        // For a perfectly linear series y = 2*x + 10 with n points,
        // the regression should recover exact slope with perfect R^2
        // and a very large |t-stat| (infinite in theory, capped in practice).
        let n = 10;
        let prices: Vec<f64> = (0..n).map(|i| 10.0 + 2.0 * i as f64).collect();

        // Use exactly one window covering all points
        let events = vec![0usize];
        let results = trend_scanning_labels(&prices, Some(&events), n, Some(n)).unwrap();

        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.best_window, n);
        assert_eq!(r.label, 1);

        // R-squared should be essentially 1.0 (perfect linear fit)
        assert!(
            (r.r_squared - 1.0).abs() < 1e-10,
            "R^2 should be 1.0 for perfect linear data, got {}",
            r.r_squared
        );

        // t-stat should be very large (slope is exact, residuals near zero)
        assert!(
            r.t_stat.abs() > 1e6,
            "t-stat should be very large for perfect linear data, got {}",
            r.t_stat
        );
    }

    #[test]
    fn test_convenience_label_series() {
        let prices: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();
        let labels = trend_scanning_label_series(&prices, 5).unwrap();
        // All should be 1 for monotonic uptrend
        for &l in &labels {
            assert_eq!(l, 1);
        }
    }

    #[test]
    fn test_max_window_exceeds_length() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = trend_scanning_labels(&prices, None, 10, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ols_t_stat_basic() {
        // y = [0, 1, 2, 3, 4]: perfect linear series
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let (t, r2) = ols_t_stat(&y);
        assert!(t.abs() > 1e6, "perfect linear should have huge t-stat");
        assert!(
            (r2 - 1.0).abs() < 1e-10,
            "perfect linear should have R^2 = 1"
        );
    }

    #[test]
    fn test_ols_t_stat_constant() {
        // All same values: beta = 0, t-stat = 0
        let y = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (t, _r2) = ols_t_stat(&y);
        assert!(t.abs() < 1e-10, "constant series should have t-stat ~0");
    }

    #[test]
    fn test_event_near_end_of_series() {
        // Events near the end should still work with smaller effective windows
        let prices: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // Index 8: only 2 remaining points (8, 9), which is less than min_window=3
        // So we should not include it in default events
        let results = trend_scanning_labels(&prices, None, 5, None).unwrap();
        // Default events: 0..=7 (prices.len() - min_w = 10 - 3 = 7)
        assert_eq!(results.len(), 8);
    }
}
