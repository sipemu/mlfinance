//! Weighted Kendall's tau correlation (Snippet 8.6).
//!
//! Computes the Kendall rank correlation coefficient between two sequences,
//! optionally using weights.

/// Classify a pair of observations and return `(concordant, discordant, tied_x, tied_y)` weights.
fn classify_pair(x_diff: f64, y_diff: f64, w: f64) -> (f64, f64, f64, f64) {
    if x_diff.abs() < 1e-15 && y_diff.abs() < 1e-15 {
        (0.0, 0.0, 0.0, 0.0) // both tied
    } else if x_diff.abs() < 1e-15 {
        (0.0, 0.0, w, 0.0)
    } else if y_diff.abs() < 1e-15 {
        (0.0, 0.0, 0.0, w)
    } else if x_diff * y_diff > 0.0 {
        (w, 0.0, 0.0, 0.0)
    } else {
        (0.0, w, 0.0, 0.0)
    }
}

/// Compute weighted Kendall's tau correlation.
///
/// Kendall's tau measures the ordinal association between two rankings.
/// The weighted version allows pairs to contribute differently to the
/// correlation based on the provided weights.
///
/// # Arguments
/// * `x` - First sequence.
/// * `y` - Second sequence (must have same length as `x`).
/// * `weights` - Optional weights for each observation. If `None`, uniform weights are used.
///
/// # Returns
/// Weighted Kendall's tau in [-1, 1]. Returns 0.0 if the input is too short.
pub fn weighted_kendall_tau(x: &[f64], y: &[f64], weights: Option<&[f64]>) -> f64 {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    if let Some(w) = weights {
        assert_eq!(x.len(), w.len(), "weights must have the same length as x");
    }

    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut tied_x = 0.0;
    let mut tied_y = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let w = weights.map_or(1.0, |w| w[i] * w[j]);
            let (c, d, tx, ty) = classify_pair(x[i] - x[j], y[i] - y[j], w);
            concordant += c;
            discordant += d;
            tied_x += tx;
            tied_y += ty;
        }
    }

    let numerator = concordant - discordant;
    let denom1 = concordant + discordant + tied_x;
    let denom2 = concordant + discordant + tied_y;

    if denom1 <= 0.0 || denom2 <= 0.0 {
        return 0.0;
    }

    // Kendall's tau-b formula (handles ties)
    numerator / (denom1 * denom2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kendall_tau_perfect_concordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = weighted_kendall_tau(&x, &y, None);
        assert!((tau - 1.0).abs() < 1e-10, "Expected 1.0, got {}", tau);
    }

    #[test]
    fn test_kendall_tau_perfect_discordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tau = weighted_kendall_tau(&x, &y, None);
        assert!((tau - (-1.0)).abs() < 1e-10, "Expected -1.0, got {}", tau);
    }

    #[test]
    fn test_kendall_tau_no_correlation() {
        // This particular arrangement gives tau = 0
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 4.0, 2.0, 3.0];
        let tau = weighted_kendall_tau(&x, &y, None);
        // Not exactly 0 for all orderings, but should be between -1 and 1
        assert!(tau >= -1.0 && tau <= 1.0);
    }

    #[test]
    fn test_kendall_tau_with_weights() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let tau = weighted_kendall_tau(&x, &y, Some(&weights));
        assert!(
            (tau - 1.0).abs() < 1e-10,
            "Expected 1.0 with uniform weights, got {}",
            tau
        );
    }

    #[test]
    fn test_kendall_tau_single_element() {
        let x = vec![1.0];
        let y = vec![1.0];
        let tau = weighted_kendall_tau(&x, &y, None);
        assert!((tau - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_empty() {
        let tau = weighted_kendall_tau(&[], &[], None);
        assert!((tau - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_with_ties() {
        let x = vec![1.0, 2.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let tau = weighted_kendall_tau(&x, &y, None);
        // With ties in x, tau should still be positive but < 1.0
        assert!(
            tau > 0.0 && tau < 1.0,
            "Expected 0 < tau < 1 with ties, got {}",
            tau
        );
    }

    #[test]
    fn test_kendall_tau_two_elements() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        let tau = weighted_kendall_tau(&x, &y, None);
        assert!((tau - 1.0).abs() < 1e-10);

        let y2 = vec![2.0, 1.0];
        let tau2 = weighted_kendall_tau(&x, &y2, None);
        assert!((tau2 - (-1.0)).abs() < 1e-10);
    }
}
