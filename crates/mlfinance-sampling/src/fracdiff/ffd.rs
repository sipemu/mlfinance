use crate::fracdiff::weights::get_weights_ffd;

/// Apply fixed-width window fractional differentiation (Snippet 5.3).
///
/// Unlike the expanding window version, FFD uses a fixed set of weights
/// determined by the threshold. For each index `t`, computes:
///
/// ```text
/// output[t] = sum(w[k] * series[t - k] for k in 0..weight_len)
/// ```
///
/// Only positions where the full weight window fits are computed; earlier
/// positions are set to `NaN`.
///
/// # Arguments
/// * `series` - input time series
/// * `d` - fractional differentiation order (typically `0 < d < 1`)
/// * `threshold` - minimum weight magnitude to determine window width
///
/// # Returns
/// A vector of the same length as `series`. The first `weight_len - 1` entries
/// are `NaN` because insufficient history is available.
pub fn frac_diff_ffd(series: &[f64], d: f64, threshold: f64) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }

    let w = get_weights_ffd(d, threshold);
    let weight_len = w.len();

    let mut output = vec![f64::NAN; n];

    for t in (weight_len - 1)..n {
        let mut sum = 0.0;
        for k in 0..weight_len {
            sum += w[k] * series[t - k];
        }
        output[t] = sum;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let result = frac_diff_ffd(&[], 0.5, 1e-5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_output_length() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = frac_diff_ffd(&series, 0.5, 0.1);
        assert_eq!(result.len(), series.len());
    }

    #[test]
    fn test_nan_prefix() {
        let series = vec![1.0; 20];
        let result = frac_diff_ffd(&series, 0.5, 0.01);
        let w = get_weights_ffd(0.5, 0.01);
        let weight_len = w.len();
        // First weight_len - 1 entries should be NaN
        for (t, val) in result.iter().enumerate().take(weight_len - 1) {
            assert!(val.is_nan(), "result[{}] = {} should be NaN", t, val);
        }
        // Entries from weight_len - 1 onward should be valid
        for (t, val) in result.iter().enumerate().skip(weight_len - 1) {
            assert!(!val.is_nan(), "result[{}] should not be NaN", t);
        }
    }

    #[test]
    fn test_d_one_first_difference() {
        // d=1 with threshold < 1 gives weights [1, -1]
        let series = vec![10.0, 12.0, 15.0, 13.0];
        let result = frac_diff_ffd(&series, 1.0, 0.01);
        // Weight length is 2 for d=1
        assert!(result[0].is_nan());
        assert!((result[1] - 2.0).abs() < 1e-10); // 12 - 10
        assert!((result[2] - 3.0).abs() < 1e-10); // 15 - 12
        assert!((result[3] - (-2.0)).abs() < 1e-10); // 13 - 15
    }

    #[test]
    fn test_constant_series_ffd() {
        // Fractional diff of constant with fixed window
        let series = vec![5.0; 20];
        let result = frac_diff_ffd(&series, 0.5, 0.01);
        let w = get_weights_ffd(0.5, 0.01);
        let weight_sum: f64 = w.iter().sum();

        // For constant c, ffd = c * sum(weights)
        for (t, val) in result.iter().enumerate().take(20).skip(w.len() - 1) {
            assert!(
                (val - 5.0 * weight_sum).abs() < 1e-10,
                "result[{}] = {}, expected {}",
                t,
                val,
                5.0 * weight_sum
            );
        }
    }

    #[test]
    fn test_higher_threshold_fewer_weights() {
        let series = vec![1.0; 50];
        let r1 = frac_diff_ffd(&series, 0.5, 0.001);
        let r2 = frac_diff_ffd(&series, 0.5, 0.1);
        // Higher threshold -> fewer weights -> fewer NaN values
        let nan_count_1 = r1.iter().filter(|x| x.is_nan()).count();
        let nan_count_2 = r2.iter().filter(|x| x.is_nan()).count();
        assert!(nan_count_2 <= nan_count_1);
    }
}
