use super::adf::adf_test;

/// Supremum ADF test for bubble detection.
///
/// Runs ADF test on expanding windows starting from `min_window`.
///
/// # Arguments
///
/// * `series` - Time series to test for explosive behavior.
/// * `min_window` - Minimum window size to start expanding from.
/// * `max_lags` - Maximum number of lagged differences in each ADF regression.
///
/// # Returns
///
/// Vector of ADF statistics for each window size from `min_window` to `n`.
/// Returns an empty vector if the series is shorter than `min_window`.
pub fn sadf(series: &[f64], min_window: usize, max_lags: usize) -> Vec<f64> {
    let n = series.len();
    if n < min_window {
        return vec![];
    }

    let mut stats = Vec::with_capacity(n - min_window + 1);

    for end in min_window..=n {
        let window = &series[..end];
        let (adf_stat, _) = adf_test(window, max_lags);
        stats.push(adf_stat);
    }

    stats
}

/// Get the SADF statistic (supremum of all expanding-window ADF stats).
///
/// # Arguments
///
/// * `series` - Time series to test for explosive behavior.
/// * `min_window` - Minimum window size to start expanding from.
/// * `max_lags` - Maximum number of lagged differences in each ADF regression.
///
/// # Returns
///
/// The maximum ADF statistic across all expanding windows. Positive values
/// indicate evidence of explosive behavior (potential bubble).
pub fn sadf_stat(series: &[f64], min_window: usize, max_lags: usize) -> f64 {
    let stats = sadf(series, min_window, max_lags);
    stats
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sadf_basic() {
        let series: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.1).sin() + i as f64 * 0.01)
            .collect();
        let stats = sadf(&series, 20, 1);
        assert!(!stats.is_empty());
        assert_eq!(stats.len(), 50 - 20 + 1);
    }

    #[test]
    fn test_sadf_stat_value() {
        let series: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.1).sin() + i as f64 * 0.01)
            .collect();
        let stat = sadf_stat(&series, 20, 1);
        assert!(stat.is_finite());
    }

    #[test]
    fn test_sadf_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let stats = sadf(&series, 10, 1);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_sadf_stat_is_supremum() {
        let series: Vec<f64> = (0..60).map(|i| (i as f64 * 0.05).sin() * 10.0).collect();
        let stats = sadf(&series, 20, 1);
        let sup = sadf_stat(&series, 20, 1);
        let max_stat = stats
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((sup - max_stat).abs() < 1e-10);
    }
}
