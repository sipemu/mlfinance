use mlfinance_core::error::{MlFinanceError, Result};
use mlfinance_core::matrix::{matrix_inverse, power_iteration_eig};
use ndarray::Array2;

/// Compute Pearson correlation coefficient between two slices.
fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
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

    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok((cov / denom).clamp(-1.0, 1.0))
}

/// Angular distance: d(x,y) = arccos(rho(x,y)) / pi.
///
/// Maps Pearson correlation to a distance in [0, 1].
/// rho = 1 -> d = 0, rho = -1 -> d = 1, rho = 0 -> d = 0.5
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
///
/// # Returns
///
/// Angular distance in [0, 1].
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn angular_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    let rho = pearson_correlation(x, y)?;
    Ok(rho.clamp(-1.0, 1.0).acos() / std::f64::consts::PI)
}

/// Absolute angular distance: d(x,y) = arccos(|rho(x,y)|) / pi.
///
/// Ignores the sign of correlation; measures distance based on strength.
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
///
/// # Returns
///
/// Absolute angular distance in [0, 0.5].
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn absolute_angular_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    let rho = pearson_correlation(x, y)?;
    Ok(rho.abs().clamp(0.0, 1.0).acos() / std::f64::consts::PI)
}

/// Squared angular distance: d(x,y) = arccos(rho(x,y)^2) / pi.
///
/// Uses squared correlation (R^2), ignoring sign and emphasizing strength.
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
///
/// # Returns
///
/// Squared angular distance in [0, 0.5].
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn squared_angular_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    let rho = pearson_correlation(x, y)?;
    let rho_sq = (rho * rho).clamp(0.0, 1.0);
    Ok(rho_sq.acos() / std::f64::consts::PI)
}

/// Compute pairwise absolute distance matrix for a 1-D sample.
fn pairwise_abs_distances(vals: &[f64], n: usize) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            mat[i * n + j] = (vals[i] - vals[j]).abs();
        }
    }
    mat
}

/// Double-center a distance matrix in place and return the centered matrix.
fn double_center(mat: &[f64], n: usize) -> Vec<f64> {
    let nf = n as f64;
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            row_means[i] += mat[i * n + j];
        }
        row_means[i] /= nf;
    }
    for j in 0..n {
        for i in 0..n {
            col_means[j] += mat[i * n + j];
        }
        col_means[j] /= nf;
    }
    let grand_mean: f64 = row_means.iter().sum::<f64>() / nf;

    let mut centered = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            centered[i * n + j] = mat[i * n + j] - row_means[i] - col_means[j] + grand_mean;
        }
    }
    centered
}

/// Compute dCov², dVarX², dVarY² from centered distance matrices.
fn compute_dcov_dvars(a: &[f64], b: &[f64], n: usize) -> (f64, f64, f64) {
    let mut dcov_sq = 0.0;
    let mut dvar_x_sq = 0.0;
    let mut dvar_y_sq = 0.0;
    for i in 0..(n * n) {
        dcov_sq += a[i] * b[i];
        dvar_x_sq += a[i] * a[i];
        dvar_y_sq += b[i] * b[i];
    }
    let n_sq = (n * n) as f64;
    (dcov_sq / n_sq, dvar_x_sq / n_sq, dvar_y_sq / n_sq)
}

/// Distance correlation (Szekely et al.) -- measures both linear and nonlinear dependence.
///
/// Unlike Pearson correlation, distance correlation equals zero if and only if
/// the random variables are independent (for finite second moments).
///
/// Algorithm:
/// 1. Compute pairwise Euclidean distance matrices A, B
/// 2. Double-center them to get centered matrices a, b
/// 3. dCov^2 = mean(a * b), dVarX^2 = mean(a * a), dVarY^2 = mean(b * b)
/// 4. dCor = sqrt(dCov^2 / sqrt(dVarX^2 * dVarY^2))
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
///
/// # Returns
///
/// Distance correlation in [0, 1].
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn distance_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
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

    let a_mat = pairwise_abs_distances(x, n);
    let b_mat = pairwise_abs_distances(y, n);

    let a_centered = double_center(&a_mat, n);
    let b_centered = double_center(&b_mat, n);

    let (dcov_sq, dvar_x_sq, dvar_y_sq) = compute_dcov_dvars(&a_centered, &b_centered, n);

    let denom = (dvar_x_sq * dvar_y_sq).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok((dcov_sq / denom).sqrt().clamp(0.0, 1.0))
}

/// Kullback-Leibler distance between two correlation matrices.
///
/// KL(A, B) = 0.5 * (tr(B^{-1} * A) - N + ln(det(B) / det(A)))
///
/// Uses eigendecomposition to compute determinants (product of eigenvalues)
/// and matrix inverse for B^{-1}.
///
/// # Arguments
///
/// * `corr_a` - First correlation matrix (n x n).
/// * `corr_b` - Second correlation matrix (n x n), must match dimensions of `corr_a`.
///
/// # Returns
///
/// Non-negative KL distance (0 if matrices are identical).
///
/// # Errors
///
/// Returns an error if matrices are non-square, have different dimensions,
/// are empty, or if the inverse or eigendecomposition fails.
pub fn kullback_leibler_distance(corr_a: &Array2<f64>, corr_b: &Array2<f64>) -> Result<f64> {
    let n = corr_a.nrows();
    if n != corr_a.ncols() || n != corr_b.nrows() || n != corr_b.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "matrices must be square and same size: A is {}x{}, B is {}x{}",
                corr_a.nrows(),
                corr_a.ncols(),
                corr_b.nrows(),
                corr_b.ncols()
            ),
        });
    }
    if n == 0 {
        return Err(MlFinanceError::InsufficientData {
            expected: 1,
            actual: 0,
        });
    }

    // Compute B^{-1}
    let b_inv = matrix_inverse(corr_b)?;

    // tr(B^{-1} * A)
    let product = b_inv.dot(corr_a);
    let trace: f64 = (0..n).map(|i| product[[i, i]]).sum();

    // Compute determinants via eigendecomposition
    let (eig_a, _) = power_iteration_eig(corr_a, n, 1000, 1e-10)?;
    let (eig_b, _) = power_iteration_eig(corr_b, n, 1000, 1e-10)?;

    let log_det_a: f64 = eig_a.iter().map(|&v| v.max(1e-15).ln()).sum();
    let log_det_b: f64 = eig_b.iter().map(|&v| v.max(1e-15).ln()).sum();

    let kl = 0.5 * (trace - n as f64 + log_det_b - log_det_a);

    Ok(kl.max(0.0))
}

/// Frobenius norm distance between two matrices: ||A - B||_F / N.
///
/// Normalized by the dimension N so the result is scale-independent.
///
/// # Arguments
///
/// * `matrix_a` - First matrix.
/// * `matrix_b` - Second matrix (must have same dimensions as `matrix_a`).
///
/// # Returns
///
/// Non-negative normalized Frobenius distance.
///
/// # Errors
///
/// Returns an error if dimensions mismatch.
pub fn norm_distance(matrix_a: &Array2<f64>, matrix_b: &Array2<f64>) -> Result<f64> {
    if matrix_a.nrows() != matrix_b.nrows() || matrix_a.ncols() != matrix_b.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "matrix dimensions must match: A is {}x{}, B is {}x{}",
                matrix_a.nrows(),
                matrix_a.ncols(),
                matrix_b.nrows(),
                matrix_b.ncols()
            ),
        });
    }

    let n = matrix_a.nrows();
    if n == 0 {
        return Ok(0.0);
    }

    let mut sum_sq = 0.0;
    for i in 0..matrix_a.nrows() {
        for j in 0..matrix_a.ncols() {
            let diff = matrix_a[[i, j]] - matrix_b[[i, j]];
            sum_sq += diff * diff;
        }
    }

    Ok(sum_sq.sqrt() / n as f64)
}

/// Compute angular distance from a precomputed correlation value.
///
/// d = arccos(rho) / pi, where rho is clamped to [-1, 1].
pub fn angular_distance_from_corr(rho: f64) -> f64 {
    rho.clamp(-1.0, 1.0).acos() / std::f64::consts::PI
}

/// Compute absolute angular distance from a precomputed correlation value.
///
/// d = arccos(|rho|) / pi
pub fn absolute_angular_distance_from_corr(rho: f64) -> f64 {
    rho.abs().clamp(0.0, 1.0).acos() / std::f64::consts::PI
}

/// Compute squared angular distance from a precomputed correlation value.
///
/// d = arccos(rho^2) / pi
pub fn squared_angular_distance_from_corr(rho: f64) -> f64 {
    (rho * rho).clamp(0.0, 1.0).acos() / std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---- pearson_correlation tests ----

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let rho = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let rho = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        // Symmetric pattern should give zero correlation
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let rho = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_constant_input() {
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let rho = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(rho, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_dimension_mismatch() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(pearson_correlation(&x, &y).is_err());
    }

    #[test]
    fn test_pearson_too_short() {
        let x = vec![1.0];
        let y = vec![1.0];
        assert!(pearson_correlation(&x, &y).is_err());
    }

    // ---- angular_distance tests ----

    #[test]
    fn test_angular_distance_perfect_corr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let d = angular_distance(&x, &y).unwrap();
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_distance_negative_corr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let d = angular_distance(&x, &y).unwrap();
        assert_abs_diff_eq!(d, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_distance_uncorrelated() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let d = angular_distance(&x, &y).unwrap();
        assert_abs_diff_eq!(d, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_distance_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 3.0, 4.0, 2.0, 1.0];
        let d = angular_distance(&x, &y).unwrap();
        assert!((0.0..=1.0).contains(&d));
    }

    #[test]
    fn test_angular_distance_empty() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        assert!(angular_distance(&x, &y).is_err());
    }

    // ---- absolute_angular_distance tests ----

    #[test]
    fn test_absolute_angular_distance_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let d = absolute_angular_distance(&x, &y).unwrap();
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_absolute_angular_distance_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let d = absolute_angular_distance(&x, &y).unwrap();
        // |rho| = 1, so d = arccos(1)/pi = 0
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_absolute_angular_distance_uncorrelated() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let d = absolute_angular_distance(&x, &y).unwrap();
        assert_abs_diff_eq!(d, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_absolute_angular_distance_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 3.0, 4.0, 2.0, 1.0];
        let d = absolute_angular_distance(&x, &y).unwrap();
        assert!((0.0..=0.5).contains(&d));
    }

    // ---- squared_angular_distance tests ----

    #[test]
    fn test_squared_angular_distance_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let d = squared_angular_distance(&x, &y).unwrap();
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_angular_distance_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let d = squared_angular_distance(&x, &y).unwrap();
        // rho^2 = 1, so d = arccos(1)/pi = 0
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_angular_distance_uncorrelated() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let d = squared_angular_distance(&x, &y).unwrap();
        // rho^2 = 0, so d = arccos(0)/pi = 0.5
        assert_abs_diff_eq!(d, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_angular_distance_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 3.0, 4.0, 2.0, 1.0];
        let d = squared_angular_distance(&x, &y).unwrap();
        assert!((0.0..=0.5).contains(&d));
    }

    // ---- distance_correlation tests ----

    #[test]
    fn test_distance_correlation_perfect_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let dc = distance_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(dc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_correlation_negative_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let dc = distance_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(dc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_correlation_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dc = distance_correlation(&x, &x).unwrap();
        assert_abs_diff_eq!(dc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_correlation_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let dc = distance_correlation(&x, &y).unwrap();
        assert!((0.0..=1.0).contains(&dc));
    }

    #[test]
    fn test_distance_correlation_constant() {
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let dc = distance_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(dc, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_correlation_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(distance_correlation(&x, &y).is_err());
    }

    #[test]
    fn test_distance_correlation_too_short() {
        let x = vec![1.0];
        let y = vec![1.0];
        assert!(distance_correlation(&x, &y).is_err());
    }

    // ---- kullback_leibler_distance tests ----

    #[test]
    fn test_kl_distance_identical() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let kl = kullback_leibler_distance(&corr, &corr).unwrap();
        assert_abs_diff_eq!(kl, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_kl_distance_nonnegative() {
        let a = array![[1.0, 0.3], [0.3, 1.0]];
        let b = array![[1.0, 0.7], [0.7, 1.0]];
        let kl = kullback_leibler_distance(&a, &b).unwrap();
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_kl_distance_dimension_mismatch() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(kullback_leibler_distance(&a, &b).is_err());
    }

    #[test]
    fn test_kl_distance_identity() {
        let eye = Array2::eye(3);
        let kl = kullback_leibler_distance(&eye, &eye).unwrap();
        assert_abs_diff_eq!(kl, 0.0, epsilon = 0.1);
    }

    // ---- norm_distance tests ----

    #[test]
    fn test_norm_distance_identical() {
        let a = array![[1.0, 0.5], [0.5, 1.0]];
        let d = norm_distance(&a, &a).unwrap();
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_norm_distance_known() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[1.0, 1.0], [1.0, 1.0]];
        let d = norm_distance(&a, &b).unwrap();
        // diff = [[0, -1], [-1, 0]], Frobenius = sqrt(0+1+1+0) = sqrt(2), normalized = sqrt(2)/2
        assert_abs_diff_eq!(d, 2.0_f64.sqrt() / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_distance_nonnegative() {
        let a = array![[1.0, 0.3], [0.3, 1.0]];
        let b = array![[1.0, 0.7], [0.7, 1.0]];
        let d = norm_distance(&a, &b).unwrap();
        assert!(d >= 0.0);
    }

    #[test]
    fn test_norm_distance_symmetric() {
        let a = array![[1.0, 0.3], [0.3, 1.0]];
        let b = array![[1.0, 0.7], [0.7, 1.0]];
        let d1 = norm_distance(&a, &b).unwrap();
        let d2 = norm_distance(&b, &a).unwrap();
        assert_abs_diff_eq!(d1, d2, epsilon = 1e-15);
    }

    #[test]
    fn test_norm_distance_dimension_mismatch() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(norm_distance(&a, &b).is_err());
    }

    #[test]
    fn test_norm_distance_empty() {
        let a = Array2::<f64>::zeros((0, 0));
        let b = Array2::<f64>::zeros((0, 0));
        let d = norm_distance(&a, &b).unwrap();
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-15);
    }

    // ---- angular_distance_from_corr tests ----

    #[test]
    fn test_angular_from_corr_one() {
        assert_abs_diff_eq!(angular_distance_from_corr(1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_from_corr_neg_one() {
        assert_abs_diff_eq!(angular_distance_from_corr(-1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_from_corr_zero() {
        assert_abs_diff_eq!(angular_distance_from_corr(0.0), 0.5, epsilon = 1e-10);
    }
}
