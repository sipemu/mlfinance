//! PCA orthogonal features (Snippet 8.5).
//!
//! Transforms features into orthogonal components using Principal Component Analysis.

use mlfinance_core::error::{MlFinanceError, Result};
use mlfinance_core::matrix::power_iteration_eig;
use ndarray::{Array1, Array2, Axis};

/// Validate PCA input parameters.
fn validate_pca_inputs(n_samples: usize, n_features: usize, n_components: usize) -> Result<()> {
    if n_samples == 0 || n_features == 0 {
        return Err(MlFinanceError::InsufficientData {
            expected: 1,
            actual: 0,
        });
    }
    if n_components == 0 || n_components > n_features {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!(
                "n_components must be in [1, {}], got {}",
                n_features, n_components
            ),
        });
    }
    Ok(())
}

/// Extract top components and explained variance ratios.
fn extract_components(
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    n_components: usize,
    total_variance: f64,
) -> (Array2<f64>, Array1<f64>) {
    let components = eigenvectors
        .slice(ndarray::s![.., ..n_components])
        .to_owned();
    let explained_variance: Array1<f64> = (0..n_components)
        .map(|k| eigenvalues[k].max(0.0) / total_variance)
        .collect();
    (components, explained_variance)
}

/// Apply PCA to create orthogonal features.
///
/// Centers the data, computes the covariance matrix, performs eigendecomposition,
/// and projects onto the top `n_components` principal components.
///
/// # Arguments
///
/// * `x` - Feature matrix of shape `(n_samples, n_features)`.
/// * `n_components` - Number of principal components to keep (must be in `[1, n_features]`).
///
/// # Returns
///
/// A tuple `(x_transformed, explained_variance_ratio)` where `x_transformed` has
/// shape `(n_samples, n_components)` and `explained_variance_ratio` contains the
/// fraction of total variance explained by each component.
///
/// # Errors
///
/// Returns an error if the input is empty, `n_components` is out of range, the
/// eigendecomposition fails, or the total variance is non-positive.
pub fn orthogonal_features(
    x: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, Array1<f64>)> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    validate_pca_inputs(n_samples, n_features, n_components)?;

    // Center the data (subtract column means)
    let means = x.mean_axis(Axis(0)).unwrap();
    let mut x_centered = x.clone();
    for mut row in x_centered.rows_mut() {
        row -= &means;
    }

    // Compute covariance matrix: (1/(n-1)) * X^T * X
    let denom = if n_samples > 1 {
        (n_samples - 1) as f64
    } else {
        1.0
    };
    let cov = x_centered.t().dot(&x_centered) / denom;

    let (eigenvalues, eigenvectors) = power_iteration_eig(&cov, n_features, 1000, 1e-10)?;

    let total_variance: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();
    if total_variance <= 0.0 {
        return Err(MlFinanceError::ComputationError {
            msg: "Total variance is zero or negative".to_string(),
        });
    }

    let (components, explained_variance) =
        extract_components(&eigenvalues, &eigenvectors, n_components, total_variance);
    let x_transformed = x_centered.dot(&components);

    Ok((x_transformed, explained_variance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonal_features_basic() {
        // Create data with clear principal component structure
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0,
            ],
        )
        .unwrap();

        let result = orthogonal_features(&x, 2);
        assert!(result.is_ok());
        let (transformed, variance_ratio) = result.unwrap();
        assert_eq!(transformed.nrows(), 5);
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(variance_ratio.len(), 2);

        // First component should explain most of the variance
        assert!(
            variance_ratio[0] > 0.9,
            "First PC should explain >90% variance"
        );
        // All variance ratios should be non-negative
        for &v in variance_ratio.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_orthogonal_features_full_components() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = orthogonal_features(&x, 2);
        assert!(result.is_ok());
        let (_, variance_ratio) = result.unwrap();
        let total: f64 = variance_ratio.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "All components should explain 100% variance, got {}",
            total
        );
    }

    #[test]
    fn test_orthogonal_features_invalid_components() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = orthogonal_features(&x, 0);
        assert!(result.is_err());

        let result = orthogonal_features(&x, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_orthogonal_features_empty() {
        let x = Array2::zeros((0, 3));
        let result = orthogonal_features(&x, 1);
        assert!(result.is_err());
    }
}
