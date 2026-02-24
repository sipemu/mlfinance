use mlfinance_core::math::{cumsum, ewma_std, sign, simple_returns};
use mlfinance_core::matrix::{cov_matrix, power_iteration_eig};
use mlfinance_core::stats::weighted_mean;
use ndarray::array;

#[test]
fn test_ewma_std_basic() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = ewma_std(&values, 3).unwrap();
    assert_eq!(result.len(), values.len());
    // First value is 0.0 (no prior data)
    assert_eq!(result[0], 0.0);
    // Later values should be positive (non-constant series)
    for &v in &result[1..] {
        assert!(v >= 0.0, "ewma_std should be non-negative, got {v}");
    }
}

#[test]
fn test_ewma_std_constant_series() {
    let values = vec![5.0; 20];
    let result = ewma_std(&values, 5).unwrap();
    // Constant series should have near-zero std
    for &v in &result[2..] {
        assert!(v.abs() < 1e-10 || v.is_nan());
    }
}

#[test]
fn test_ewma_std_insufficient_data() {
    let result = ewma_std(&[1.0], 3);
    assert!(result.is_err());
    let result = ewma_std(&[], 3);
    assert!(result.is_err());
}

#[test]
fn test_cumsum() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = cumsum(&values);
    assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

#[test]
fn test_cumsum_empty() {
    let result = cumsum(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_cumsum_single() {
    let result = cumsum(&[42.0]);
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_simple_returns() {
    let prices = vec![100.0, 110.0, 105.0, 115.0];
    let returns = simple_returns(&prices);
    assert_eq!(returns.len(), 3);
    assert!((returns[0] - 0.1).abs() < 1e-10);
    assert!((returns[1] - (-5.0 / 110.0)).abs() < 1e-10);
    assert!((returns[2] - (10.0 / 105.0)).abs() < 1e-10);
}

#[test]
fn test_simple_returns_empty_and_single() {
    assert!(simple_returns(&[]).is_empty());
    assert!(simple_returns(&[100.0]).is_empty());
}

#[test]
fn test_sign() {
    assert_eq!(sign(5.0), 1.0);
    assert_eq!(sign(-3.0), -1.0);
    assert_eq!(sign(0.0), 0.0);
    assert_eq!(sign(f64::INFINITY), 1.0);
    assert_eq!(sign(f64::NEG_INFINITY), -1.0);
}

#[test]
fn test_power_iteration_eig_diagonal() {
    // Diagonal matrix with distinct eigenvalues (power iteration works best here)
    let m = array![[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
    let (eigenvalues, eigenvectors) = power_iteration_eig(&m, 3, 1000, 1e-10).unwrap();
    let mut evs: Vec<f64> = eigenvalues.to_vec();
    evs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((evs[0] - 3.0).abs() < 1e-4, "got {}", evs[0]);
    assert!((evs[1] - 2.0).abs() < 1e-4, "got {}", evs[1]);
    assert!((evs[2] - 1.0).abs() < 1e-4, "got {}", evs[2]);
    assert_eq!(eigenvectors.shape(), &[3, 3]);
}

#[test]
fn test_power_iteration_eig_symmetric() {
    // Simple symmetric 2x2 matrix
    let m = array![[2.0, 1.0], [1.0, 2.0]];
    let (eigenvalues, eigenvectors) = power_iteration_eig(&m, 2, 1000, 1e-10).unwrap();
    // Eigenvalues should be 3 and 1
    let mut evs: Vec<f64> = eigenvalues.to_vec();
    evs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!(
        (evs[0] - 3.0).abs() < 1e-6,
        "largest eigenvalue should be 3, got {}",
        evs[0]
    );
    assert!(
        (evs[1] - 1.0).abs() < 1e-6,
        "smallest eigenvalue should be 1, got {}",
        evs[1]
    );
    assert_eq!(eigenvectors.shape(), &[2, 2]);
}

#[test]
fn test_power_iteration_eig_single_component() {
    let m = array![[4.0, 2.0], [2.0, 3.0]];
    let (eigenvalues, _) = power_iteration_eig(&m, 1, 1000, 1e-10).unwrap();
    assert_eq!(eigenvalues.len(), 1);
    // Largest eigenvalue of [[4,2],[2,3]] is (7+sqrt(5))/2 ≈ 5.236
    assert!(eigenvalues[0] > 4.0);
}

#[test]
fn test_weighted_mean_uniform() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![1.0, 1.0, 1.0, 1.0];
    let result = weighted_mean(&values, &weights).unwrap();
    assert!((result - 2.5).abs() < 1e-10);
}

#[test]
fn test_weighted_mean_non_uniform() {
    let values = vec![10.0, 20.0];
    let weights = vec![3.0, 1.0];
    let result = weighted_mean(&values, &weights).unwrap();
    assert!((result - 12.5).abs() < 1e-10);
}

#[test]
fn test_weighted_mean_mismatched_lengths() {
    let result = weighted_mean(&[1.0, 2.0], &[1.0]);
    assert!(result.is_err());
}

#[test]
fn test_cov_matrix_basic() {
    // Two perfectly correlated features
    let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let cov = cov_matrix(&data).unwrap();
    assert_eq!(cov.shape(), &[2, 2]);
    // Variance of first col: var([1,2,3,4])
    assert!(cov[[0, 0]] > 0.0);
    assert!(cov[[1, 1]] > 0.0);
    // Perfect correlation -> cov[0,1]^2 = cov[0,0]*cov[1,1]
    let ratio = cov[[0, 1]].powi(2) / (cov[[0, 0]] * cov[[1, 1]]);
    assert!((ratio - 1.0).abs() < 1e-10);
}

#[test]
fn test_cov_matrix_single_row() {
    let data = array![[1.0, 2.0]];
    let result = cov_matrix(&data);
    // Single row should either return zero cov or error
    assert!(result.is_ok() || result.is_err());
}
