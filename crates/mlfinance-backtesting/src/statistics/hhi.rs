//! Herfindahl-Hirschman Index (HHI) for concentration (Snippet 14.3).
//!
//! The HHI measures concentration. Applied to strategy returns, it indicates
//! whether profits are concentrated in a few periods or spread evenly.

/// Herfindahl-Hirschman Index for concentration.
///
/// HHI = sum(w_i^2) where w_i are the weights (shares). For normalized weights
/// that sum to 1, HHI ranges from 1/N (perfectly diversified) to 1 (concentrated).
///
/// # Arguments
/// * `weights` - Slice of weights/shares. They need not sum to 1; the function
///   normalizes them. Must be non-negative.
///
/// # Returns
/// HHI in `[0, 1]`. Returns 0.0 if the input is empty or all weights are zero.
pub fn hhi(weights: &[f64]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }

    let total: f64 = weights.iter().map(|w| w.abs()).sum();
    if total < 1e-15 {
        return 0.0;
    }

    weights
        .iter()
        .map(|&w| {
            let share = w.abs() / total;
            share * share
        })
        .sum()
}

/// HHI of positive vs negative returns.
///
/// Separates the return series into positive and negative returns and computes
/// the HHI of each group. This reveals whether profits or losses are
/// concentrated in a few periods.
///
/// # Arguments
/// * `returns` - Slice of return values.
///
/// # Returns
/// A tuple `(hhi_positive, hhi_negative)` where:
/// - `hhi_positive`: concentration of positive returns.
/// - `hhi_negative`: concentration of negative returns.
///   Returns `(0.0, 0.0)` if the series is empty.
pub fn hhi_concentration(returns: &[f64]) -> (f64, f64) {
    if returns.is_empty() {
        return (0.0, 0.0);
    }

    let positives: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    let negatives: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|&r| r.abs())
        .collect();

    let hhi_pos = hhi(&positives);
    let hhi_neg = hhi(&negatives);

    (hhi_pos, hhi_neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hhi_equal_weights() {
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let h = hhi(&weights);
        assert!((h - 0.25).abs() < 1e-10, "hhi={} should be 0.25", h);
    }

    #[test]
    fn test_hhi_concentrated() {
        let weights = vec![1.0, 0.0, 0.0, 0.0];
        let h = hhi(&weights);
        assert!((h - 1.0).abs() < 1e-10, "hhi={} should be 1.0", h);
    }

    #[test]
    fn test_hhi_unnormalized() {
        let weights = vec![10.0, 10.0, 10.0, 10.0];
        let h = hhi(&weights);
        assert!((h - 0.25).abs() < 1e-10, "hhi={} should still be 0.25", h);
    }

    #[test]
    fn test_hhi_empty() {
        assert_eq!(hhi(&[]), 0.0);
    }

    #[test]
    fn test_hhi_all_zeros() {
        assert_eq!(hhi(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_hhi_single() {
        let h = hhi(&[5.0]);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hhi_concentration_basic() {
        let returns = vec![0.1, -0.05, 0.2, -0.1, 0.05];
        let (hhi_pos, hhi_neg) = hhi_concentration(&returns);
        assert!(hhi_pos > 0.0);
        assert!(hhi_neg > 0.0);
    }

    #[test]
    fn test_hhi_concentration_all_positive() {
        let returns = vec![0.1, 0.2, 0.3];
        let (hhi_pos, hhi_neg) = hhi_concentration(&returns);
        assert!(hhi_pos > 0.0);
        assert_eq!(hhi_neg, 0.0);
    }

    #[test]
    fn test_hhi_concentration_all_negative() {
        let returns = vec![-0.1, -0.2, -0.3];
        let (hhi_pos, hhi_neg) = hhi_concentration(&returns);
        assert_eq!(hhi_pos, 0.0);
        assert!(hhi_neg > 0.0);
    }

    #[test]
    fn test_hhi_concentration_empty() {
        assert_eq!(hhi_concentration(&[]), (0.0, 0.0));
    }
}
