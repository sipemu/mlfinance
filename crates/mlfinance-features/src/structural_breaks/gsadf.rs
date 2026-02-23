use super::adf::adf_test;

/// Generalized SADF: runs ADF on all possible sub-windows.
///
/// For each possible start point and end point (with minimum window size),
/// compute the ADF statistic. Returns all ADF statistics.
///
/// # Arguments
///
/// * `series` - Time series to test for explosive behavior.
/// * `min_window` - Minimum sub-window size for ADF tests.
/// * `max_lags` - Maximum number of lagged differences in each ADF regression.
///
/// # Returns
///
/// Vector of all ADF statistics from every sub-window. Returns an empty
/// vector if the series is shorter than `min_window`.
pub fn gsadf(series: &[f64], min_window: usize, max_lags: usize) -> Vec<f64> {
    let n = series.len();
    if n < min_window {
        return vec![];
    }

    let mut stats = Vec::new();

    for start in 0..n {
        for end in (start + min_window)..=n {
            let window = &series[start..end];
            let (adf_stat, _) = adf_test(window, max_lags);
            if adf_stat.is_finite() {
                stats.push(adf_stat);
            }
        }
    }

    stats
}

/// Get the GSADF statistic (supremum of all sub-window ADF stats).
///
/// # Arguments
///
/// * `series` - Time series to test for explosive behavior.
/// * `min_window` - Minimum sub-window size for ADF tests.
/// * `max_lags` - Maximum number of lagged differences in each ADF regression.
///
/// # Returns
///
/// The maximum ADF statistic across all sub-windows. Positive values indicate
/// evidence of explosive behavior (potential bubble).
pub fn gsadf_stat(series: &[f64], min_window: usize, max_lags: usize) -> f64 {
    let stats = gsadf(series, min_window, max_lags);
    stats.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsadf_basic() {
        let series: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.1).sin() + i as f64 * 0.01)
            .collect();
        let stats = gsadf(&series, 15, 1);
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_gsadf_stat_value() {
        let series: Vec<f64> = (0..30)
            .map(|i| (i as f64 * 0.1).sin() + i as f64 * 0.01)
            .collect();
        let stat = gsadf_stat(&series, 15, 1);
        assert!(stat.is_finite());
    }

    #[test]
    fn test_gsadf_more_stats_than_sadf() {
        let series: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let g_stats = gsadf(&series, 15, 1);
        let s_stats = super::super::sadf::sadf(&series, 15, 1);
        // GSADF considers all sub-windows, SADF only expanding from 0
        assert!(g_stats.len() >= s_stats.len());
    }

    #[test]
    fn test_gsadf_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let stats = gsadf(&series, 10, 1);
        assert!(stats.is_empty());
    }
}
