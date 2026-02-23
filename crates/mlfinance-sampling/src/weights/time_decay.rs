/// Compute decay factors for positive oldest_weight (linear from oldest_weight to 1.0).
fn decay_factors_positive(n: usize, oldest_weight: f64) -> Vec<f64> {
    let clamped = oldest_weight.min(1.0);
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            clamped + t * (1.0 - clamped)
        })
        .collect()
}

/// Compute decay factors for negative oldest_weight (zero out first |oldest_weight| samples).
fn decay_factors_negative(n: usize, oldest_weight: f64) -> Vec<f64> {
    let zero_count = (oldest_weight.abs() as usize).min(n);
    let remaining = n - zero_count;
    let mut factors = vec![0.0f64; n];
    if remaining > 0 {
        for (i, factor) in factors[zero_count..].iter_mut().enumerate() {
            *factor = if remaining == 1 {
                1.0
            } else {
                i as f64 / (remaining - 1) as f64
            };
        }
        factors[n - 1] = 1.0;
    }
    factors
}

/// Apply piecewise-linear time decay to sample weights (Snippet 4.11).
///
/// Multiplies each weight by a linearly interpolated decay factor so that the
/// oldest sample (first element) receives `oldest_weight` relative to the newest
/// sample (last element, which always receives factor `1.0`).
///
/// # Arguments
///
/// * `weights` - input weights in time order (oldest first)
/// * `oldest_weight` - decay factor for the oldest sample:
///   - `0 <= oldest_weight <= 1`: linear interpolation from `oldest_weight` to `1.0`.
///     A value of `1.0` means no decay.
///   - `oldest_weight < 0`: the first `|oldest_weight|` samples receive zero weight,
///     and the remaining samples decay linearly from near-zero to `1.0`.
///
/// # Returns
///
/// A vector of the same length as `weights` with the decay applied element-wise.
pub fn time_decay(weights: &[f64], oldest_weight: f64) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![weights[0] * oldest_weight.clamp(0.0, 1.0)];
    }

    let decay_factors = if oldest_weight >= 0.0 {
        decay_factors_positive(n, oldest_weight)
    } else {
        decay_factors_negative(n, oldest_weight)
    };

    weights
        .iter()
        .zip(decay_factors.iter())
        .map(|(&w, &d)| w * d)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let result = time_decay(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_no_decay() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let result = time_decay(&weights, 1.0);
        // oldest_weight = 1.0 means no decay
        for (r, &w) in result.iter().zip(weights.iter()) {
            assert!((r - w).abs() < 1e-10);
        }
    }

    #[test]
    fn test_full_decay() {
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let result = time_decay(&weights, 0.0);
        // Oldest gets 0, newest gets 1
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
        // Linear interpolation: [0, 1/3, 2/3, 1]
        assert!((result[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((result[2] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_half_decay() {
        let weights = vec![1.0, 1.0, 1.0];
        let result = time_decay(&weights, 0.5);
        // Decay factors: [0.5, 0.75, 1.0]
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.75).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_oldest_weight() {
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = time_decay(&weights, -2.0);
        // First 2 samples get zero weight
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        // Remaining 3 samples decay linearly from 0 to 1
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - 0.5).abs() < 1e-10);
        assert!((result[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_weight() {
        let weights = vec![2.0];
        let result = time_decay(&weights, 0.5);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_varying_weights() {
        let weights = vec![2.0, 4.0, 6.0];
        let result = time_decay(&weights, 0.5);
        // Decay factors: [0.5, 0.75, 1.0]
        assert!((result[0] - 1.0).abs() < 1e-10); // 2.0 * 0.5
        assert!((result[1] - 3.0).abs() < 1e-10); // 4.0 * 0.75
        assert!((result[2] - 6.0).abs() < 1e-10); // 6.0 * 1.0
    }
}
