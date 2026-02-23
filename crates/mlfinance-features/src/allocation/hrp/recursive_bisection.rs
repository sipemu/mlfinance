use ndarray::{Array1, Array2};

/// Recursive bisection: allocate weights by splitting sorted items.
///
/// At each step, split the sorted list in half. Compute inverse-variance
/// weight for each half. Recurse into each half to distribute weights.
///
/// # Arguments
///
/// * `cov` - Covariance matrix (n x n).
/// * `sorted_indices` - Quasi-diagonalized ordering of asset indices.
///
/// # Returns
///
/// Portfolio weight vector of length n that sums to 1.0.
pub fn recursive_bisection(cov: &Array2<f64>, sorted_indices: &[usize]) -> Array1<f64> {
    let n = cov.nrows();
    let mut weights = Array1::ones(n);

    recursive_bisection_inner(cov, sorted_indices, &mut weights);

    weights
}

fn recursive_bisection_inner(
    cov: &Array2<f64>,
    sorted_indices: &[usize],
    weights: &mut Array1<f64>,
) {
    if sorted_indices.len() <= 1 {
        return;
    }

    let mid = sorted_indices.len() / 2;
    let left = &sorted_indices[..mid];
    let right = &sorted_indices[mid..];

    // Compute inverse-variance allocation between the two halves
    let left_var = cluster_variance(cov, left);
    let right_var = cluster_variance(cov, right);

    // Allocation factor: weight given to left cluster
    let total_inv_var = 1.0 / left_var + 1.0 / right_var;
    let alpha = (1.0 / left_var) / total_inv_var;

    // Scale weights
    for &i in left {
        weights[i] *= alpha;
    }
    for &i in right {
        weights[i] *= 1.0 - alpha;
    }

    // Recurse
    recursive_bisection_inner(cov, left, weights);
    recursive_bisection_inner(cov, right, weights);
}

/// Compute the variance of an inverse-variance-weighted cluster.
///
/// For a cluster of assets, compute the IVP weights within the cluster,
/// then compute the portfolio variance.
fn cluster_variance(cov: &Array2<f64>, indices: &[usize]) -> f64 {
    let k = indices.len();
    if k == 1 {
        return cov[[indices[0], indices[0]]];
    }

    // Inverse-variance weights within the cluster
    let mut ivp_weights = Array1::zeros(k);
    let mut total_inv_var = 0.0;
    for (idx, &i) in indices.iter().enumerate() {
        let var_i = cov[[i, i]];
        if var_i > 0.0 {
            ivp_weights[idx] = 1.0 / var_i;
            total_inv_var += 1.0 / var_i;
        }
    }
    if total_inv_var > 0.0 {
        ivp_weights /= total_inv_var;
    } else {
        // Equal weights if all variances are zero
        ivp_weights.fill(1.0 / k as f64);
    }

    // Portfolio variance: w' * Sigma_cluster * w
    let mut portfolio_var = 0.0;
    for (a, &i) in indices.iter().enumerate() {
        for (b, &j) in indices.iter().enumerate() {
            portfolio_var += ivp_weights[a] * ivp_weights[b] * cov[[i, j]];
        }
    }

    portfolio_var.max(1e-15)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_recursive_bisection_two_assets() {
        // Two uncorrelated assets with different variances
        let cov = array![[0.04, 0.0], [0.0, 0.01]];
        let sorted = vec![0, 1];
        let weights = recursive_bisection(&cov, &sorted);
        assert_eq!(weights.len(), 2);
        // Asset with lower variance should get more weight
        assert!(weights[1] > weights[0]);
        // Weights should sum to 1
        assert!((weights.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recursive_bisection_equal_variance() {
        let cov = array![[0.04, 0.0], [0.0, 0.04]];
        let sorted = vec![0, 1];
        let weights = recursive_bisection(&cov, &sorted);
        // Equal variance -> equal weights
        assert!((weights[0] - 0.5).abs() < 1e-10);
        assert!((weights[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_recursive_bisection_single_asset() {
        let cov = array![[0.04]];
        let sorted = vec![0];
        let weights = recursive_bisection(&cov, &sorted);
        assert!((weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recursive_bisection_four_assets() {
        let cov = array![
            [0.04, 0.01, 0.005, 0.002],
            [0.01, 0.09, 0.003, 0.001],
            [0.005, 0.003, 0.01, 0.004],
            [0.002, 0.001, 0.004, 0.16]
        ];
        let sorted = vec![0, 2, 1, 3];
        let weights = recursive_bisection(&cov, &sorted);
        assert_eq!(weights.len(), 4);
        // Weights should sum to 1
        assert!((weights.sum() - 1.0).abs() < 1e-10);
        // All weights should be positive
        for &w in weights.iter() {
            assert!(w > 0.0);
        }
    }
}
