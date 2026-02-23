use mlfinance_core::math::{ewma, log_returns};
use mlfinance_core::matrix::matrix_inverse;
use mlfinance_core::stats::{correlation_matrix, covariance_matrix, mean, std_dev, variance};
use ndarray::Array2;

#[test]
fn correlation_of_single_column() {
    let data = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
    if let Ok(corr) = correlation_matrix(&data) {
        assert_eq!(corr.shape(), &[1, 1]);
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn covariance_of_constant_columns() {
    let data = Array2::from_shape_fn((20, 3), |(_i, j)| (j + 1) as f64);
    if let Ok(cov) = covariance_matrix(&data) {
        // All entries should be 0 (no variance in constant data)
        for &val in cov.iter() {
            assert!(
                val.abs() < 1e-10,
                "Expected 0 covariance for constant data, got {}",
                val
            );
        }
    }
}

#[test]
fn matrix_inverse_singular_returns_error() {
    let m = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
    assert!(matrix_inverse(&m).is_err());
}

#[test]
fn matrix_inverse_identity() {
    let eye = Array2::<f64>::eye(3);
    let inv = matrix_inverse(&eye).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (inv[[i, j]] - expected).abs() < 1e-10,
                "inv[{},{}] = {}, expected {}",
                i,
                j,
                inv[[i, j]],
                expected
            );
        }
    }
}

#[test]
fn ewma_empty_input() {
    let result = ewma(&[], 10);
    assert!(result.is_err() || result.unwrap().is_empty());
}

#[test]
fn ewma_single_value() {
    if let Ok(result) = ewma(&[5.0], 10) {
        assert_eq!(result.len(), 1);
        assert!((result[0] - 5.0).abs() < 1e-10);
    }
}

#[test]
fn log_returns_empty() {
    let result = log_returns(&[]);
    assert!(result.is_empty());
}

#[test]
fn log_returns_single_price() {
    let result = log_returns(&[100.0]);
    assert!(result.is_empty());
}

#[test]
fn mean_empty() {
    assert!(mean(&[]).is_err());
}

#[test]
fn variance_single_value() {
    // Variance of single value with ddof=0 should be 0
    if let Ok(v) = variance(&[5.0], 0) {
        assert!(v.abs() < 1e-10);
    }
}

#[test]
fn std_dev_empty() {
    assert!(std_dev(&[], 0).is_err());
}

#[test]
fn ewma_nan_input() {
    let values = vec![1.0, f64::NAN, 3.0, 4.0];
    let result = ewma(&values, 3);
    // Should not panic; NaN propagates
    if let Ok(r) = result {
        // At least some values should be NaN due to propagation
        assert!(r.iter().any(|v| v.is_nan()));
    }
}

#[test]
fn log_returns_inf_price() {
    let prices = vec![100.0, f64::INFINITY, 200.0];
    let returns = log_returns(&prices);
    // Should not panic
    assert_eq!(returns.len(), 2);
}
