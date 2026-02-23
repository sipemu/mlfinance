use mlfinance_core::error::{MlFinanceError, Result};

use crate::codependence::information::mutual_information;

/// Spearman's rho rank correlation.
///
/// Computes the Pearson correlation of the rank-transformed variables.
/// Ties are handled by assigning the average rank.
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
///
/// # Returns
///
/// Spearman rank correlation in [-1, 1].
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn spearmans_rho(x: &[f64], y: &[f64]) -> Result<f64> {
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

    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Pearson correlation of ranks
    let nf = n as f64;
    let mean_rx: f64 = rank_x.iter().sum::<f64>() / nf;
    let mean_ry: f64 = rank_y.iter().sum::<f64>() / nf;

    let mut cov = 0.0;
    let mut var_rx = 0.0;
    let mut var_ry = 0.0;

    for i in 0..n {
        let dx = rank_x[i] - mean_rx;
        let dy = rank_y[i] - mean_ry;
        cov += dx * dy;
        var_rx += dx * dx;
        var_ry += dy * dy;
    }

    let denom = (var_rx * var_ry).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok((cov / denom).clamp(-1.0, 1.0))
}

/// GPR distance: uses Spearman's rho with a theta-controlled mapping.
///
/// d_GPR(x, y) = 0.5 * (1 - rho_s(x, y))^theta
///
/// The theta parameter controls the sensitivity to correlation.
/// theta = 1 gives the standard GPR distance.
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
/// * `theta` - Sensitivity exponent (must be non-negative).
///
/// # Returns
///
/// GPR distance (non-negative).
///
/// # Errors
///
/// Returns an error if lengths differ, fewer than 2 samples, or theta < 0.
pub fn gpr_distance(x: &[f64], y: &[f64], theta: f64) -> Result<f64> {
    if theta < 0.0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!("theta must be non-negative, got {}", theta),
        });
    }

    let rho = spearmans_rho(x, y)?;
    let base = 0.5 * (1.0 - rho);
    Ok(base.powf(theta))
}

/// GNPR distance: combines GPR with normalized mutual information.
///
/// d_GNPR = d_GPR * (1 - I_norm(x, y))
///
/// This scales the GPR distance by the normalized mutual information,
/// incorporating both rank-based and information-theoretic dependence.
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
/// * `theta` - Sensitivity exponent for GPR (must be non-negative).
/// * `n_bins` - Number of histogram bins for MI computation (if `None`, uses Sturges rule).
///
/// # Returns
///
/// GNPR distance (non-negative, always <= GPR distance).
///
/// # Errors
///
/// Returns an error if lengths differ, fewer than 2 samples, or theta < 0.
pub fn gnpr_distance(x: &[f64], y: &[f64], theta: f64, n_bins: Option<usize>) -> Result<f64> {
    let d_gpr = gpr_distance(x, y, theta)?;
    let nmi = mutual_information(x, y, n_bins, true)?;
    Ok(d_gpr * (1.0 - nmi))
}

/// Compute ranks with average tie-breaking.
///
/// Returns a vector of ranks (1-based) where ties get the average of their ranks.
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    // Create index-value pairs and sort by value
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find the range of equal values
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Assign average rank to all tied values
        // Ranks are 1-based: positions i..j get ranks (i+1)..(j)
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- compute_ranks tests ----

    #[test]
    fn test_ranks_simple() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = compute_ranks(&values);
        assert_abs_diff_eq!(ranks[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ranks_with_ties() {
        let values = vec![1.0, 2.0, 2.0, 4.0];
        let ranks = compute_ranks(&values);
        assert_abs_diff_eq!(ranks[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[1], 2.5, epsilon = 1e-10); // average of 2 and 3
        assert_abs_diff_eq!(ranks[2], 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[3], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ranks_all_tied() {
        let values = vec![5.0, 5.0, 5.0];
        let ranks = compute_ranks(&values);
        for &r in &ranks {
            assert_abs_diff_eq!(r, 2.0, epsilon = 1e-10);
        }
    }

    // ---- spearmans_rho tests ----

    #[test]
    fn test_spearman_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearmans_rho(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let rho = spearmans_rho(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rho = spearmans_rho(&x, &x).unwrap();
        assert_abs_diff_eq!(rho, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_monotonic_nonlinear() {
        // Spearman captures any monotonic relationship
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&v| v * v * v).collect(); // cubic
        let rho = spearmans_rho(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let rho = spearmans_rho(&x, &y).unwrap();
        assert!(rho >= -1.0 && rho <= 1.0);
    }

    #[test]
    fn test_spearman_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(spearmans_rho(&x, &y).is_err());
    }

    #[test]
    fn test_spearman_too_short() {
        let x = vec![1.0];
        let y = vec![1.0];
        assert!(spearmans_rho(&x, &y).is_err());
    }

    #[test]
    fn test_spearman_constant() {
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let rho = spearmans_rho(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, 0.0, epsilon = 1e-10);
    }

    // ---- gpr_distance tests ----

    #[test]
    fn test_gpr_distance_perfect_corr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let d = gpr_distance(&x, &y, 1.0).unwrap();
        // rho = 1, d = 0.5 * (1-1)^1 = 0
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gpr_distance_negative_corr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let d = gpr_distance(&x, &y, 1.0).unwrap();
        // rho = -1, d = 0.5 * (1-(-1))^1 = 0.5 * 2 = 1.0
        assert_abs_diff_eq!(d, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gpr_distance_theta_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0];
        let d = gpr_distance(&x, &y, 0.0).unwrap();
        // Any base^0 = 1 (when base > 0)
        // d = 0.5*(1-rho), base>0 for rho<1, base^0 = 1
        // Actually: base = 0.5*(1-rho), if rho != 1 then base>0, base^0=1
        assert_abs_diff_eq!(d, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gpr_distance_nonnegative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let d = gpr_distance(&x, &y, 1.0).unwrap();
        assert!(d >= 0.0);
    }

    #[test]
    fn test_gpr_distance_negative_theta() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert!(gpr_distance(&x, &y, -1.0).is_err());
    }

    // ---- gnpr_distance tests ----

    #[test]
    fn test_gnpr_distance_perfect_corr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&v| v * 2.0).collect();
        let d = gnpr_distance(&x, &y, 1.0, Some(5)).unwrap();
        // rho=1 => d_gpr=0 => d_gnpr=0
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gnpr_distance_nonnegative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let d = gnpr_distance(&x, &y, 1.0, Some(4)).unwrap();
        assert!(d >= 0.0);
    }

    #[test]
    fn test_gnpr_distance_less_than_gpr() {
        // GNPR = GPR * (1 - NMI), so GNPR <= GPR (since NMI >= 0)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let d_gpr = gpr_distance(&x, &y, 1.0).unwrap();
        let d_gnpr = gnpr_distance(&x, &y, 1.0, Some(4)).unwrap();
        assert!(d_gnpr <= d_gpr + 1e-10);
    }

    #[test]
    fn test_gnpr_distance_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(gnpr_distance(&x, &y, 1.0, None).is_err());
    }
}
