use ndarray::{Array1, Array2};

use super::quasi_diag::quasi_diag;
use super::recursive_bisection::recursive_bisection;
use super::tree_clustering::{correlation_distance, single_linkage_clustering};

/// Full HRP pipeline: correlation -> distance -> clustering -> quasi-diag -> recursive bisection.
///
/// # Arguments
///
/// * `returns` - Asset returns matrix with shape (n_observations, n_assets).
///
/// # Returns
///
/// Portfolio weight vector of length n_assets that sums to 1.0.
///
/// # Errors
///
/// Returns an error if the input is empty, has fewer than 2 observations, or if
/// the correlation matrix computation fails.
pub fn hrp_weights(returns: &Array2<f64>) -> mlfinance_core::error::Result<Array1<f64>> {
    let n = returns.ncols();
    if n == 0 {
        return Err(mlfinance_core::MlFinanceError::EmptySeries);
    }
    if n == 1 {
        return Ok(Array1::ones(1));
    }
    if returns.nrows() < 2 {
        return Err(mlfinance_core::MlFinanceError::InsufficientData {
            expected: 2,
            actual: returns.nrows(),
        });
    }

    // Step 1: Compute correlation and covariance matrices
    let corr = mlfinance_core::stats::correlation_matrix(returns)?;
    let cov = mlfinance_core::stats::covariance_matrix(returns)?;

    // Step 2: Compute distance matrix from correlation
    let dist = correlation_distance(&corr);

    // Step 3: Hierarchical clustering
    let linkage = single_linkage_clustering(&dist);

    // Step 4: Quasi-diagonalization
    let sorted_indices = quasi_diag(&linkage, n);

    // Step 5: Recursive bisection
    let weights = recursive_bisection(&cov, &sorted_indices);

    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hrp_weights_basic() {
        // 3 assets, 5 observations
        let returns = array![
            [0.01, 0.02, -0.01],
            [0.02, 0.01, 0.005],
            [-0.005, 0.015, 0.01],
            [0.01, -0.01, 0.02],
            [0.005, 0.008, -0.005]
        ];
        let weights = hrp_weights(&returns).unwrap();
        assert_eq!(weights.len(), 3);
        // Weights should sum to 1
        assert!((weights.sum() - 1.0).abs() < 1e-10);
        // All weights should be positive
        for &w in weights.iter() {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_hrp_single_asset() {
        let returns = array![[0.01], [0.02], [-0.01]];
        let weights = hrp_weights(&returns).unwrap();
        assert_eq!(weights.len(), 1);
        assert!((weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hrp_insufficient_data() {
        let returns = array![[0.01, 0.02]];
        let result = hrp_weights(&returns);
        assert!(result.is_err());
    }

    #[test]
    fn test_hrp_empty() {
        let returns = Array2::<f64>::zeros((5, 0));
        let result = hrp_weights(&returns);
        assert!(result.is_err());
    }
}
