use ndarray::{Array1, Array2};

/// CLA: compute minimum variance portfolio weights.
///
/// Solves: min w' * Sigma * w  subject to  sum(w) = 1
/// Solution: w = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1)
///
/// # Arguments
///
/// * `cov` - Covariance matrix (n x n), must be square and invertible.
///
/// # Returns
///
/// Portfolio weight vector of length n that sums to 1.0.
///
/// # Errors
///
/// Returns an error if the matrix is empty, non-square, singular, or degenerate.
pub fn cla_min_variance(cov: &Array2<f64>) -> mlfinance_core::error::Result<Array1<f64>> {
    let n = cov.nrows();
    if n == 0 {
        return Err(mlfinance_core::MlFinanceError::EmptySeries);
    }
    if n != cov.ncols() {
        return Err(mlfinance_core::MlFinanceError::DimensionMismatch {
            msg: format!(
                "covariance matrix must be square, got {}x{}",
                n,
                cov.ncols()
            ),
        });
    }
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let inv_cov = mlfinance_core::matrix::matrix_inverse(cov)?;

    // w_raw = Sigma^{-1} * 1
    let ones = Array1::ones(n);
    let w_raw = inv_cov.dot(&ones);

    // Normalize so weights sum to 1
    let total: f64 = w_raw.sum();
    if total.abs() < 1e-15 {
        return Err(mlfinance_core::MlFinanceError::ComputationError {
            msg: "degenerate covariance matrix: weights sum to zero".into(),
        });
    }

    Ok(w_raw / total)
}

/// Validate inputs for max-Sharpe CLA.
fn validate_max_sharpe_inputs(
    n: usize,
    cov_ncols: usize,
    mu_len: usize,
) -> mlfinance_core::error::Result<()> {
    if n == 0 {
        return Err(mlfinance_core::MlFinanceError::EmptySeries);
    }
    if n != cov_ncols {
        return Err(mlfinance_core::MlFinanceError::DimensionMismatch {
            msg: format!("covariance matrix must be square, got {}x{}", n, cov_ncols),
        });
    }
    if mu_len != n {
        return Err(mlfinance_core::MlFinanceError::DimensionMismatch {
            msg: format!(
                "expected returns length {} != covariance dimension {}",
                mu_len, n
            ),
        });
    }
    Ok(())
}

/// CLA: maximum Sharpe ratio portfolio.
///
/// Computes the tangency portfolio weights that maximize the Sharpe ratio.
/// Falls back to minimum variance if the weights sum to zero.
///
/// # Arguments
///
/// * `expected_returns` - Expected return vector of length n.
/// * `cov` - Covariance matrix (n x n).
///
/// # Returns
///
/// Portfolio weight vector of length n that sums to 1.0.
///
/// # Errors
///
/// Returns an error if dimensions mismatch, matrix is empty, or the
/// covariance matrix is singular.
pub fn cla_max_sharpe(
    expected_returns: &Array1<f64>,
    cov: &Array2<f64>,
) -> mlfinance_core::error::Result<Array1<f64>> {
    let n = cov.nrows();
    validate_max_sharpe_inputs(n, cov.ncols(), expected_returns.len())?;
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let inv_cov = mlfinance_core::matrix::matrix_inverse(cov)?;
    let w_raw = inv_cov.dot(expected_returns);
    let total: f64 = w_raw.sum();
    if total.abs() < 1e-15 {
        return cla_min_variance(cov);
    }

    Ok(w_raw / total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_min_variance_diagonal() {
        // Diagonal covariance: analytical solution is inverse-variance weighted
        let cov = array![[0.04, 0.0], [0.0, 0.01]];
        let weights = cla_min_variance(&cov).unwrap();
        // w1 = (1/0.04) / (1/0.04 + 1/0.01) = 25 / (25 + 100) = 0.2
        // w2 = (1/0.01) / (1/0.04 + 1/0.01) = 100 / (25 + 100) = 0.8
        assert!((weights[0] - 0.2).abs() < 1e-10);
        assert!((weights[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_min_variance_sums_to_one() {
        let cov = array![
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.003],
            [0.005, 0.003, 0.01]
        ];
        let weights = cla_min_variance(&cov).unwrap();
        assert!((weights.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_sharpe() {
        let mu = array![0.05, 0.10];
        let cov = array![[0.04, 0.0], [0.0, 0.04]];
        let weights = cla_max_sharpe(&mu, &cov).unwrap();
        assert!((weights.sum() - 1.0).abs() < 1e-10);
        // Higher expected return asset should get more weight
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_max_sharpe_single() {
        let mu = array![0.05];
        let cov = array![[0.04]];
        let weights = cla_max_sharpe(&mu, &cov).unwrap();
        assert!((weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mu = array![0.05];
        let cov = array![[0.04, 0.01], [0.01, 0.09]];
        let result = cla_max_sharpe(&mu, &cov);
        assert!(result.is_err());
    }
}
