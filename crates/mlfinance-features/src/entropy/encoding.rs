/// Binary encoding: encode returns as `true` (positive) or `false` (negative/zero).
///
/// # Arguments
///
/// * `values` - Slice of numeric values to encode.
///
/// # Returns
///
/// A boolean vector with the same length as `values`.
pub fn binary_encode(values: &[f64]) -> Vec<bool> {
    values.iter().map(|&v| v > 0.0).collect()
}

/// Quantile encoding: assign each value to a quantile bin.
///
/// Sorts the values and divides them into `num_bins` equal-sized bins.
/// Each value is mapped to its bin index (0..num_bins).
///
/// # Arguments
///
/// * `values` - Slice of numeric values to encode.
/// * `num_bins` - Number of quantile bins to create.
///
/// # Returns
///
/// A vector of bin indices (0-indexed) with the same length as `values`.
pub fn quantile_encode(values: &[f64], num_bins: usize) -> Vec<usize> {
    if values.is_empty() || num_bins == 0 {
        return vec![];
    }

    let n = values.len();

    // Sort values and compute quantile thresholds
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute bin boundaries
    let mut thresholds = Vec::with_capacity(num_bins - 1);
    for i in 1..num_bins {
        let idx = (i as f64 * n as f64 / num_bins as f64) as usize;
        let idx = idx.min(n - 1);
        thresholds.push(sorted[idx]);
    }

    // Assign each value to a bin
    values
        .iter()
        .map(|&v| {
            let mut bin = 0;
            for &t in &thresholds {
                if v >= t {
                    bin += 1;
                } else {
                    break;
                }
            }
            bin.min(num_bins - 1)
        })
        .collect()
}

/// Sigma encoding: assign based on standard deviation bands.
///
/// Values are categorized into bands based on their distance from the mean
/// in units of standard deviation. Band 0 is the lowest, `num_bands - 1` is the highest.
///
/// # Arguments
///
/// * `values` - Slice of numeric values to encode.
/// * `num_bands` - Number of sigma bands to create.
///
/// # Returns
///
/// A vector of band indices (0-indexed) with the same length as `values`.
pub fn sigma_encode(values: &[f64], num_bands: usize) -> Vec<usize> {
    if values.is_empty() || num_bands == 0 {
        return vec![];
    }
    if num_bands == 1 {
        return vec![0; values.len()];
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();

    if std < 1e-15 {
        // All values are the same: put everything in the middle band
        return vec![num_bands / 2; values.len()];
    }

    // Create symmetric bands around the mean
    // Band boundaries: mean +/- k * sigma for k = 1, 2, ...
    // We create num_bands - 1 boundaries centered on the mean
    let half_bands = (num_bands as f64 - 1.0) / 2.0;

    values
        .iter()
        .map(|&v| {
            let z = (v - mean) / std;
            // Map z-score to band index
            // z in (-inf, -half_bands] -> 0
            // z in (-half_bands, -half_bands+1] -> 1
            // ...
            // z in (half_bands-1, inf) -> num_bands-1
            let band = (z + half_bands).floor() as i64;
            band.clamp(0, (num_bands - 1) as i64) as usize
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_encode() {
        let values = vec![0.1, -0.2, 0.0, 0.5, -0.1];
        let encoded = binary_encode(&values);
        assert_eq!(encoded, vec![true, false, false, true, false]);
    }

    #[test]
    fn test_binary_encode_empty() {
        let encoded = binary_encode(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_quantile_encode_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let encoded = quantile_encode(&values, 2);
        assert_eq!(encoded.len(), 4);
        // Lower half should be bin 0, upper half bin 1
        assert_eq!(encoded[0], 0); // 1.0
        assert_eq!(encoded[3], 1); // 4.0
    }

    #[test]
    fn test_quantile_encode_four_bins() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let encoded = quantile_encode(&values, 4);
        assert_eq!(encoded.len(), 100);
        // First value should be bin 0
        assert_eq!(encoded[0], 0);
        // Last value should be bin 3
        assert_eq!(encoded[99], 3);
    }

    #[test]
    fn test_quantile_encode_empty() {
        let encoded = quantile_encode(&[], 4);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_sigma_encode_basic() {
        // Standard normal-like values
        let values = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let encoded = sigma_encode(&values, 3);
        assert_eq!(encoded.len(), 5);
        // All values should be in valid range
        for &e in &encoded {
            assert!(e < 3);
        }
    }

    #[test]
    fn test_sigma_encode_constant() {
        let values = vec![5.0; 10];
        let encoded = sigma_encode(&values, 3);
        // All same value -> middle band
        for &e in &encoded {
            assert_eq!(e, 1);
        }
    }

    #[test]
    fn test_sigma_encode_single_band() {
        let values = vec![1.0, 2.0, 3.0];
        let encoded = sigma_encode(&values, 1);
        assert_eq!(encoded, vec![0, 0, 0]);
    }

    #[test]
    fn test_sigma_encode_empty() {
        let encoded = sigma_encode(&[], 3);
        assert!(encoded.is_empty());
    }
}
