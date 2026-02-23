//! PCA weights from risk distribution.
//!
//! Computes portfolio weights that allocate risk evenly across the principal
//! components of a covariance matrix. This is based on the approach in
//! "Advances in Financial Machine Learning" (Chapter 3).

use mlfinance_core::error::{MlFinanceError, Result};
use mlfinance_core::matrix::power_iteration_eig;
use ndarray::{Array1, Array2};

/// Compute PCA-based portfolio weights that distribute risk evenly across
/// principal components.
///
/// Given a covariance matrix, this function:
/// 1. Computes the eigendecomposition
/// 2. Allocates risk budget across components (equal by default)
/// 3. Computes weights in the original asset space
///
/// # Arguments
/// - `cov_matrix`: NxN covariance matrix of asset returns
/// - `risk_target`: optional total risk target (variance). If `None`, uses 1.0.
///
/// # Returns
/// An `Array1<f64>` of N portfolio weights.
///
/// # Errors
/// Returns an error if the matrix is not square or eigendecomposition fails.
pub fn pca_weights(cov_matrix: &Array2<f64>, risk_target: Option<f64>) -> Result<Array1<f64>> {
    let n = cov_matrix.nrows();
    if n != cov_matrix.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "covariance matrix must be square, got {}x{}",
                n,
                cov_matrix.ncols()
            ),
        });
    }
    if n == 0 {
        return Err(MlFinanceError::EmptySeries);
    }

    let risk_target = risk_target.unwrap_or(1.0);

    // Eigendecomposition
    let (eigenvalues, eigenvectors) = power_iteration_eig(cov_matrix, n, 1000, 1e-12)?;

    // Allocate risk budget equally across all components
    let risk_per_component = risk_target / n as f64;

    // Weights in PCA space: w_k = sqrt(risk_per_component / lambda_k)
    let mut pca_weights_vec = Array1::zeros(n);
    for k in 0..n {
        if eigenvalues[k] > 1e-15 {
            pca_weights_vec[k] = (risk_per_component / eigenvalues[k]).sqrt();
        }
    }

    // Transform back to original asset space: w = V * w_pca
    let weights = eigenvectors.dot(&pca_weights_vec);

    // Normalize so weights sum to 1
    let sum: f64 = weights.iter().map(|w| w.abs()).sum();
    if sum < 1e-15 {
        return Err(MlFinanceError::ComputationError {
            msg: "degenerate covariance matrix produces zero weights".into(),
        });
    }

    Ok(&weights / sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pca_weights_identity() {
        // Identity covariance -> all eigenvalues equal
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let w = pca_weights(&cov, None).unwrap();
        assert_eq!(w.len(), 2);
        // Weights should be normalized (absolute values sum to 1)
        let sum: f64 = w.iter().map(|x| x.abs()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Both weights should be non-zero
        assert!(w[0].abs() > 1e-10);
        assert!(w[1].abs() > 1e-10);
    }

    #[test]
    fn test_pca_weights_diagonal() {
        // Diagonal covariance with different variances
        let cov = array![[4.0, 0.0], [0.0, 1.0]];
        let w = pca_weights(&cov, None).unwrap();
        assert_eq!(w.len(), 2);
        // Sum of absolute weights should be 1
        let sum: f64 = w.iter().map(|x| x.abs()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pca_weights_with_risk_target() {
        let cov = array![[1.0, 0.5], [0.5, 1.0]];
        let w = pca_weights(&cov, Some(0.5)).unwrap();
        assert_eq!(w.len(), 2);
        let sum: f64 = w.iter().map(|x| x.abs()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pca_weights_non_square_error() {
        let cov = Array2::zeros((2, 3));
        let result = pca_weights(&cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pca_weights_empty_error() {
        let cov = Array2::zeros((0, 0));
        let result = pca_weights(&cov, None);
        assert!(result.is_err());
    }
}
