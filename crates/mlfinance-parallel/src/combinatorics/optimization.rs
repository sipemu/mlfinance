use ndarray::{Array1, Array2};

/// Static mean-variance portfolio optimization.
///
/// Solves `max(w'mu - lambda/2 * w'Sigma*w)` subject to `sum(w) = 1` using
/// the analytical solution `w* = (1/lambda) * Sigma^{-1} * mu`, then normalises
/// the weights to sum to one. Equivalent to Snippet 21.5.
///
/// # Arguments
///
/// * `expected_returns` - Vector of expected returns for each asset.
/// * `cov_matrix` - Covariance matrix of asset returns (must be square and
///   match the length of `expected_returns`).
/// * `risk_aversion` - Risk-aversion parameter (lambda). Larger values produce
///   more conservative portfolios.
///
/// # Returns
///
/// Normalised portfolio weight vector that sums to one.
///
/// # Errors
///
/// Returns an error if the covariance matrix is singular or cannot be inverted.
pub fn static_optimization(
    expected_returns: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    risk_aversion: f64,
) -> mlfinance_core::error::Result<Array1<f64>> {
    let n = expected_returns.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Simple equal-weight as fallback when no covariance inverse is available
    // For proper optimization, use with ndarray-linalg
    let inv = mlfinance_core::matrix::matrix_inverse(cov_matrix)?;
    let _ones: Array1<f64> = Array1::ones(n);

    // Unconstrained: w* = (1/lambda) * Sigma^-1 * mu
    let raw_weights: Array1<f64> = inv.dot(expected_returns) / risk_aversion;

    // Normalize to sum to 1
    let total: f64 = raw_weights.sum();
    if total.abs() < 1e-15 {
        return Ok(Array1::from_elem(n, 1.0 / n as f64));
    }

    Ok(&raw_weights / total)
}

/// Dynamic portfolio optimization with time-varying parameters.
///
/// At each time step, computes the mean-variance optimal weights using the
/// corresponding expected-return vector and covariance matrix.
/// Equivalent to Snippets 21.6-21.7.
///
/// # Arguments
///
/// * `expected_returns_series` - Expected return vectors, one per time step.
/// * `cov_matrices` - Covariance matrices, one per time step (must have the
///   same length as `expected_returns_series`).
/// * `risk_aversion` - Risk-aversion parameter applied at every step.
///
/// # Returns
///
/// A vector of portfolio weight vectors, one per time step.
///
/// # Errors
///
/// Returns an error if the two series have different lengths, or if any single-step
/// optimisation fails (e.g., singular covariance matrix).
pub fn dynamic_optimization(
    expected_returns_series: &[Array1<f64>],
    cov_matrices: &[Array2<f64>],
    risk_aversion: f64,
) -> mlfinance_core::error::Result<Vec<Array1<f64>>> {
    if expected_returns_series.len() != cov_matrices.len() {
        return Err(mlfinance_core::error::MlFinanceError::DimensionMismatch {
            msg: "expected returns and covariance series must have same length".into(),
        });
    }

    let mut weights_series = Vec::with_capacity(expected_returns_series.len());
    for (mu, sigma) in expected_returns_series.iter().zip(cov_matrices.iter()) {
        let w = static_optimization(mu, sigma, risk_aversion)?;
        weights_series.push(w);
    }
    Ok(weights_series)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_static_optimization() {
        let mu = array![0.1, 0.2];
        let sigma = array![[0.04, 0.01], [0.01, 0.09]];
        let w = static_optimization(&mu, &sigma, 1.0).unwrap();
        assert!((w.sum() - 1.0).abs() < 1e-10);
        assert_eq!(w.len(), 2);
    }
}
