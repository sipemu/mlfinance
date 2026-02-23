use mlfinance_core::error::{MlFinanceError, Result};

/// Compute optimal number of histogram bins.
///
/// If `corr_coef` is provided, uses the KN rule:
///   bins = max(1, floor(sqrt(n / 2) * |rho|))
///
/// Otherwise uses the Sturges rule:
///   bins = max(1, ceil(log2(n) + 1))
///
/// # Arguments
///
/// * `n` - Number of data points.
/// * `corr_coef` - Optional Pearson correlation coefficient for the KN rule.
///
/// # Returns
///
/// Recommended number of bins (always >= 1).
pub fn optimal_number_of_bins(n: usize, corr_coef: Option<f64>) -> usize {
    if n == 0 {
        return 1;
    }

    match corr_coef {
        Some(rho) => {
            let rho_abs = rho.abs();
            if rho_abs < 1e-15 {
                return 1;
            }
            let bins = ((n as f64 / 2.0).sqrt() * rho_abs).floor() as usize;
            bins.max(1)
        }
        None => {
            // Sturges rule
            let bins = ((n as f64).log2() + 1.0).ceil() as usize;
            bins.max(1)
        }
    }
}

/// Mutual information between two continuous variables using histogram binning.
///
/// I(X;Y) = H(X) + H(Y) - H(X,Y)
///
/// where entropies are computed from bin count histograms.
/// If `normalize` is true, returns I(X;Y) / min(H(X), H(Y)), giving
/// Normalized Mutual Information in [0, 1].
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
/// * `n_bins` - Number of histogram bins (if `None`, uses Sturges rule).
/// * `normalize` - Whether to return normalized MI in [0, 1].
///
/// # Returns
///
/// Mutual information in bits (or normalized MI in [0, 1]).
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn mutual_information(
    x: &[f64],
    y: &[f64],
    n_bins: Option<usize>,
    normalize: bool,
) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!("x length {} != y length {}", n, y.len()),
        });
    }
    if n < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: n,
        });
    }

    let bins = n_bins.unwrap_or_else(|| optimal_number_of_bins(n, None));
    let bins = bins.max(1);

    let (hx, hy, hxy) = compute_entropies(x, y, bins)?;

    let mi = (hx + hy - hxy).max(0.0);

    if normalize {
        let min_h = hx.min(hy);
        if min_h < 1e-15 {
            return Ok(0.0);
        }
        Ok((mi / min_h).clamp(0.0, 1.0))
    } else {
        Ok(mi)
    }
}

/// Variation of Information between two continuous variables using histogram binning.
///
/// VI(X;Y) = H(X|Y) + H(Y|X) = H(X,Y) - I(X;Y)
///
/// If `normalize` is true, returns VI / H(X,Y), giving normalized VI in [0, 1].
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
/// * `n_bins` - Number of histogram bins (if `None`, uses Sturges rule).
/// * `normalize` - Whether to return normalized VI in [0, 1].
///
/// # Returns
///
/// Variation of information in bits (or normalized in [0, 1]).
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn variation_of_information(
    x: &[f64],
    y: &[f64],
    n_bins: Option<usize>,
    normalize: bool,
) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!("x length {} != y length {}", n, y.len()),
        });
    }
    if n < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: n,
        });
    }

    let bins = n_bins.unwrap_or_else(|| optimal_number_of_bins(n, None));
    let bins = bins.max(1);

    let (hx, hy, hxy) = compute_entropies(x, y, bins)?;

    let mi = (hx + hy - hxy).max(0.0);
    let vi = (hxy - mi).max(0.0);

    if normalize {
        if hxy < 1e-15 {
            return Ok(0.0);
        }
        Ok((vi / hxy).clamp(0.0, 1.0))
    } else {
        Ok(vi)
    }
}

/// Compute marginal and joint entropies from histogram binning.
///
/// Returns (H(X), H(Y), H(X,Y)) in bits (log base 2).
fn compute_entropies(x: &[f64], y: &[f64], bins: usize) -> Result<(f64, f64, f64)> {
    let n = x.len();
    let nf = n as f64;

    // Compute min/max for x and y
    let (x_min, x_max) = min_max(x)?;
    let (y_min, y_max) = min_max(y)?;

    // Width of each bin (add small epsilon to ensure max value falls within range)
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    let x_width = if x_range < 1e-15 {
        1.0
    } else {
        x_range * (1.0 + 1e-10) / bins as f64
    };
    let y_width = if y_range < 1e-15 {
        1.0
    } else {
        y_range * (1.0 + 1e-10) / bins as f64
    };

    // Build histograms
    let mut hist_x = vec![0usize; bins];
    let mut hist_y = vec![0usize; bins];
    let mut hist_xy = vec![0usize; bins * bins];

    for i in 0..n {
        let bx = if x_range < 1e-15 {
            0
        } else {
            ((x[i] - x_min) / x_width).floor() as usize
        };
        let by = if y_range < 1e-15 {
            0
        } else {
            ((y[i] - y_min) / y_width).floor() as usize
        };

        let bx = bx.min(bins - 1);
        let by = by.min(bins - 1);

        hist_x[bx] += 1;
        hist_y[by] += 1;
        hist_xy[bx * bins + by] += 1;
    }

    // Compute entropies
    let hx = entropy_from_counts(&hist_x, nf);
    let hy = entropy_from_counts(&hist_y, nf);
    let hxy = entropy_from_counts(&hist_xy, nf);

    Ok((hx, hy, hxy))
}

/// Shannon entropy from bin counts: H = -sum(p * log2(p))
fn entropy_from_counts(counts: &[usize], total: f64) -> f64 {
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum()
}

/// Compute min and max of a slice.
fn min_max(values: &[f64]) -> Result<(f64, f64)> {
    if values.is_empty() {
        return Err(MlFinanceError::EmptySeries);
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in values {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    Ok((min_val, max_val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- optimal_number_of_bins tests ----

    #[test]
    fn test_optimal_bins_sturges() {
        // For n=100, Sturges: ceil(log2(100) + 1) = ceil(6.64 + 1) = 8
        let bins = optimal_number_of_bins(100, None);
        assert_eq!(bins, 8);
    }

    #[test]
    fn test_optimal_bins_kn_rule() {
        // n=100, rho=0.5: floor(sqrt(50) * 0.5) = floor(7.07 * 0.5) = floor(3.54) = 3
        let bins = optimal_number_of_bins(100, Some(0.5));
        assert_eq!(bins, 3);
    }

    #[test]
    fn test_optimal_bins_kn_high_corr() {
        // n=100, rho=0.9: floor(sqrt(50) * 0.9) = floor(7.07 * 0.9) = floor(6.36) = 6
        let bins = optimal_number_of_bins(100, Some(0.9));
        assert_eq!(bins, 6);
    }

    #[test]
    fn test_optimal_bins_zero_corr() {
        let bins = optimal_number_of_bins(100, Some(0.0));
        assert_eq!(bins, 1);
    }

    #[test]
    fn test_optimal_bins_zero_n() {
        let bins = optimal_number_of_bins(0, None);
        assert_eq!(bins, 1);
    }

    #[test]
    fn test_optimal_bins_small_n() {
        let bins = optimal_number_of_bins(2, None);
        assert!(bins >= 1);
    }

    // ---- mutual_information tests ----

    #[test]
    fn test_mutual_info_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mi = mutual_information(&x, &x, Some(5), false).unwrap();
        assert!(mi > 0.0);
    }

    #[test]
    fn test_mutual_info_normalized_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let nmi = mutual_information(&x, &x, Some(5), true).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_info_nonnegative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let mi = mutual_information(&x, &y, Some(4), false).unwrap();
        assert!(mi >= 0.0);
    }

    #[test]
    fn test_mutual_info_normalized_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let nmi = mutual_information(&x, &y, Some(4), true).unwrap();
        assert!(nmi >= 0.0 && nmi <= 1.0);
    }

    #[test]
    fn test_mutual_info_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(mutual_information(&x, &y, None, false).is_err());
    }

    #[test]
    fn test_mutual_info_too_short() {
        let x = vec![1.0];
        let y = vec![1.0];
        assert!(mutual_information(&x, &y, None, false).is_err());
    }

    #[test]
    fn test_mutual_info_constant() {
        // If x is constant, H(X) = 0, so MI = 0
        let x = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mi = mutual_information(&x, &y, Some(3), false).unwrap();
        assert_abs_diff_eq!(mi, 0.0, epsilon = 1e-10);
    }

    // ---- variation_of_information tests ----

    #[test]
    fn test_vi_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let vi = variation_of_information(&x, &x, Some(5), false).unwrap();
        assert_abs_diff_eq!(vi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vi_normalized_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let nvi = variation_of_information(&x, &x, Some(5), true).unwrap();
        assert_abs_diff_eq!(nvi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vi_nonnegative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let vi = variation_of_information(&x, &y, Some(4), false).unwrap();
        assert!(vi >= 0.0);
    }

    #[test]
    fn test_vi_normalized_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let nvi = variation_of_information(&x, &y, Some(4), true).unwrap();
        assert!(nvi >= 0.0 && nvi <= 1.0);
    }

    #[test]
    fn test_vi_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(variation_of_information(&x, &y, None, false).is_err());
    }

    #[test]
    fn test_vi_and_mi_relationship() {
        // VI = H(X,Y) - I(X;Y) and H(X,Y) = H(X) + H(Y) - I(X;Y)
        // So VI = H(X) + H(Y) - 2*I(X;Y)
        // Also: VI + I = H(X,Y) means VI <= H(X,Y)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let mi = mutual_information(&x, &y, Some(4), false).unwrap();
        let vi = variation_of_information(&x, &y, Some(4), false).unwrap();
        // Both should be non-negative
        assert!(mi >= 0.0);
        assert!(vi >= 0.0);
    }
}
