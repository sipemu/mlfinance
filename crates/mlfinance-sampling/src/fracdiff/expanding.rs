use crate::fracdiff::weights::get_weights;

/// Apply fractional differentiation with expanding window (Snippet 5.2).
///
/// For each index `t`, computes the weighted sum using all available past observations:
///
/// ```text
/// output[t] = sum(w[k] * series[t - k] for k in 0..=t if |w[k]| >= threshold)
/// ```
///
/// The window expands as `t` increases, starting narrow and using more history.
/// Only weights with absolute value >= `threshold` are included in the sum.
///
/// # Arguments
/// * `series` - input time series
/// * `d` - fractional differentiation order (typically `0 < d < 1`)
/// * `threshold` - minimum weight magnitude to include
///
/// # Returns
/// A vector of the same length as `series` with fractionally differentiated values.
/// Early values where the window is too short may have reduced accuracy.
pub fn frac_diff_expanding(series: &[f64], d: f64, threshold: f64) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute all weights up to the maximum possible size
    let all_weights = get_weights(d, n);

    // Find the effective weight length (those above threshold)
    let max_weight_len = all_weights
        .iter()
        .rposition(|&w| w.abs() >= threshold)
        .map(|pos| pos + 1)
        .unwrap_or(1);

    let mut output = vec![f64::NAN; n];

    for t in 0..n {
        // Number of weights we can use at position t (limited by available history)
        let num_weights = (t + 1).min(max_weight_len);
        let mut sum = 0.0;

        for k in 0..num_weights {
            if all_weights[k].abs() < threshold {
                continue;
            }
            sum += all_weights[k] * series[t - k];
        }

        // Only output if we have enough weights (at least the ones above threshold
        // that are within our window)
        output[t] = sum;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let result = frac_diff_expanding(&[], 0.5, 1e-5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_d_zero_identity() {
        // d=0 should return the original series (only w[0]=1 matters)
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = frac_diff_expanding(&series, 0.0, 1e-5);
        for (i, (&r, &s)) in result.iter().zip(series.iter()).enumerate() {
            assert!((r - s).abs() < 1e-10, "index {}: {} != {}", i, r, s);
        }
    }

    #[test]
    fn test_d_one_first_difference() {
        // d=1 should be close to first difference
        let series = vec![10.0, 12.0, 15.0, 13.0];
        let result = frac_diff_expanding(&series, 1.0, 1e-5);
        // result[0] = 1 * 10 = 10 (no previous values)
        assert!((result[0] - 10.0).abs() < 1e-10);
        // result[1] = 1*12 + (-1)*10 = 2
        assert!((result[1] - 2.0).abs() < 1e-10);
        // result[2] = 1*15 + (-1)*12 + 0*10 = 3
        assert!((result[2] - 3.0).abs() < 1e-10);
        // result[3] = 1*13 + (-1)*15 + 0*12 + 0*10 = -2
        assert!((result[3] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_constant_series() {
        // Fractional diff of a constant should approach 0 after the first element
        let series = vec![5.0; 10];
        let result = frac_diff_expanding(&series, 0.5, 1e-5);
        // For a constant series, as t grows, the sum of weights * constant
        // should converge. The first element is just the constant.
        assert!((result[0] - 5.0).abs() < 1e-10);
        // Subsequent elements should be smaller in magnitude
        for t in 1..result.len() {
            assert!(
                result[t].abs() < result[0].abs() + 1e-10,
                "result[{}] = {} should be smaller than result[0] = {}",
                t,
                result[t],
                result[0]
            );
        }
    }

    #[test]
    fn test_output_length_matches_input() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = frac_diff_expanding(&series, 0.5, 1e-5);
        assert_eq!(result.len(), series.len());
    }
}
