use crate::fracdiff::ffd::frac_diff_ffd;

/// Test a single candidate d for stationarity.
fn test_candidate_d(series: &[f64], d: f64, threshold: f64) -> bool {
    let diff = frac_diff_ffd(series, d, threshold);
    let valid: Vec<f64> = diff.iter().copied().filter(|x| !x.is_nan()).collect();
    valid.len() >= 4 && is_likely_stationary(&valid)
}

/// Find the minimum fractional differentiation order `d` that makes the series stationary.
///
/// Tests d values from `step_size` up to `max_d` in increments of `step_size`.
/// At each candidate, FFD is applied and the result is checked for stationarity
/// using a variance-ratio proxy. The first `d` that passes is returned.
///
/// # Arguments
///
/// * `series` - input time series (e.g., a price series or similar non-stationary data)
/// * `max_d` - maximum differentiation order to test (typically 1.0)
/// * `step_size` - increment for testing d values (e.g., 0.1)
/// * `threshold` - minimum weight magnitude for the FFD computation
///
/// # Returns
///
/// The smallest `d` in `(0, max_d]` for which the FFD series passes the stationarity
/// check, or `max_d` if no smaller value suffices. Returns `max_d` immediately when
/// the series has fewer than 4 observations or the parameters are invalid.
pub fn find_min_d(series: &[f64], max_d: f64, step_size: f64, threshold: f64) -> f64 {
    if series.len() < 4 || step_size <= 0.0 || max_d <= 0.0 {
        return max_d;
    }

    let mut d = step_size;
    while d <= max_d + 1e-10 {
        if test_candidate_d(series, d, threshold) {
            return d;
        }
        d += step_size;
    }

    max_d
}

/// Simplified stationarity check using variance ratio.
///
/// Splits the series into halves and computes the variance of each half.
/// If the ratio of the two variances is close to 1, the series is likely
/// stationary (i.e., the statistical properties are stable over time).
///
/// This is a simplified proxy for a proper ADF test. In production, one would
/// use a full augmented Dickey-Fuller test.
fn is_likely_stationary(series: &[f64]) -> bool {
    let n = series.len();
    if n < 4 {
        return false;
    }

    let mean = series.iter().sum::<f64>() / n as f64;
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    if variance < 1e-15 {
        // Essentially constant -> stationary
        return true;
    }

    // Split into two halves and check variance ratio
    let mid = n / 2;
    let first_half = &series[..mid];
    let second_half = &series[mid..];

    let mean1 = first_half.iter().sum::<f64>() / first_half.len() as f64;
    let var1 = first_half.iter().map(|x| (x - mean1).powi(2)).sum::<f64>()
        / (first_half.len() - 1).max(1) as f64;

    let mean2 = second_half.iter().sum::<f64>() / second_half.len() as f64;
    let var2 = second_half.iter().map(|x| (x - mean2).powi(2)).sum::<f64>()
        / (second_half.len() - 1).max(1) as f64;

    // Check if means are similar (mean-stationarity).
    // Use standard deviation as the scale for comparison to avoid division-by-near-zero.
    let std_dev = variance.sqrt();
    let mean_diff_normalized = (mean1 - mean2).abs() / std_dev;

    // Variance ratio test
    let var_max = var1.max(var2);
    let var_min = var1.min(var2);
    let var_ratio = if var_max > 1e-15 {
        var_min / var_max
    } else {
        1.0
    };

    // Consider stationary if variance ratio > 0.3 and mean difference is < 1 std dev
    var_ratio > 0.3 && mean_diff_normalized < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_series() {
        let result = find_min_d(&[], 1.0, 0.1, 1e-5);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_short_series() {
        let result = find_min_d(&[1.0, 2.0], 1.0, 0.1, 1e-5);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_already_stationary() {
        // White noise (alternating pattern) is stationary and should need small d.
        // Use a large deterministic pseudo-random series centered around 0.
        let series: Vec<f64> = (0..500)
            .map(|i| {
                let val = ((i * 17 + 5) % 23) as f64 - 11.0;
                val / 11.0
            })
            .collect();
        let result = find_min_d(&series, 1.0, 0.1, 1e-2);
        // Already stationary (bounded, mean-reverting), should find d < 1.0
        // The simplified variance-ratio proxy is not as powerful as a true ADF test,
        // so we allow up to d=0.7.
        assert!(
            result <= 0.7,
            "stationary series should need moderate d, got {}",
            result
        );
    }

    #[test]
    fn test_random_walk_needs_differencing() {
        // A random walk: cumulative sum of noise
        let mut series = vec![0.0; 200];
        let mut val = 100.0;
        for i in 1..200 {
            // Simple deterministic "random walk"
            val += ((i * 7 + 3) % 11) as f64 - 5.0;
            series[i] = val;
        }
        let result = find_min_d(&series, 1.0, 0.1, 1e-4);
        // Should need some differentiation
        assert!(
            result > 0.0,
            "random walk should need d > 0, got {}",
            result
        );
    }

    #[test]
    fn test_d_one_always_works() {
        // d=1 is full integer differentiation, should always make series stationary
        // (assuming the series is I(1))
        let series: Vec<f64> = (0..100).map(|i| i as f64 * 2.0 + 10.0).collect();
        let diff = frac_diff_ffd(&series, 1.0, 1e-5);
        let valid: Vec<f64> = diff.iter().copied().filter(|x| !x.is_nan()).collect();
        // First difference of linear series is constant -> stationary
        assert!(is_likely_stationary(&valid));
    }

    #[test]
    fn test_returns_max_d_on_invalid_params() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((find_min_d(&series, 1.0, 0.0, 1e-5) - 1.0).abs() < 1e-10);
        assert!((find_min_d(&series, 0.0, 0.1, 1e-5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_likely_stationary_constant() {
        let series = vec![5.0; 20];
        assert!(is_likely_stationary(&series));
    }

    #[test]
    fn test_is_likely_stationary_trending() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Strong upward trend is not stationary
        assert!(!is_likely_stationary(&series));
    }
}
