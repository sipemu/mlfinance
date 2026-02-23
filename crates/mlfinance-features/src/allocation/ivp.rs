use ndarray::{Array1, Array2};

/// Compute inverse variance portfolio weights.
///
/// Each asset is weighted inversely proportional to its variance:
/// w_i = (1/sigma_i^2) / sum(1/sigma_j^2).
///
/// # Arguments
///
/// * `cov` - Covariance matrix (n x n); only diagonal entries are used.
///
/// # Returns
///
/// Weight vector of length n that sums to 1.0. Falls back to equal weights
/// if all diagonal entries are zero or negative.
pub fn inverse_variance_weights(cov: &Array2<f64>) -> Array1<f64> {
    let n = cov.nrows();
    if n == 0 {
        return Array1::zeros(0);
    }

    let mut inv_variances = Array1::zeros(n);
    let mut total_inv_var = 0.0;

    for i in 0..n {
        let var_i = cov[[i, i]];
        if var_i > 0.0 {
            inv_variances[i] = 1.0 / var_i;
            total_inv_var += 1.0 / var_i;
        }
    }

    if total_inv_var > 0.0 {
        inv_variances / total_inv_var
    } else {
        // Equal weights if all variances are zero or negative
        Array1::from_elem(n, 1.0 / n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_inverse_variance_diagonal() {
        let cov = array![[0.04, 0.0], [0.0, 0.01]];
        let weights = inverse_variance_weights(&cov);
        // w0 = (1/0.04) / (1/0.04 + 1/0.01) = 25/125 = 0.2
        // w1 = (1/0.01) / (1/0.04 + 1/0.01) = 100/125 = 0.8
        assert!((weights[0] - 0.2).abs() < 1e-10);
        assert!((weights[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_variance_equal() {
        let cov = array![[0.04, 0.0], [0.0, 0.04]];
        let weights = inverse_variance_weights(&cov);
        assert!((weights[0] - 0.5).abs() < 1e-10);
        assert!((weights[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_variance_sums_to_one() {
        let cov = array![
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.003],
            [0.005, 0.003, 0.01]
        ];
        let weights = inverse_variance_weights(&cov);
        assert!((weights.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_variance_single() {
        let cov = array![[0.04]];
        let weights = inverse_variance_weights(&cov);
        assert!((weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_variance_empty() {
        let cov = Array2::<f64>::zeros((0, 0));
        let weights = inverse_variance_weights(&cov);
        assert!(weights.is_empty());
    }
}
