use mlfinance_core::math::{ewma, log_returns};
use mlfinance_core::matrix::matrix_inverse;
use mlfinance_core::stats::{correlation_matrix, covariance_matrix};
use ndarray::Array2;
use proptest::prelude::*;

fn generate_data_matrix(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((i * 7 + j * 13 + 1) as f64 * 0.7).sin() * 10.0
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn correlation_matrix_symmetric(rows in 10usize..50, cols in 2usize..8) {
        let data = generate_data_matrix(rows, cols);
        if let Ok(corr) = correlation_matrix(&data) {
            let n = corr.nrows();
            for i in 0..n {
                for j in 0..n {
                    prop_assert!(
                        (corr[[i, j]] - corr[[j, i]]).abs() < 1e-10,
                        "Not symmetric at [{},{}]",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn correlation_matrix_diagonal_is_one(rows in 10usize..50, cols in 2usize..8) {
        let data = generate_data_matrix(rows, cols);
        if let Ok(corr) = correlation_matrix(&data) {
            let n = corr.nrows();
            for i in 0..n {
                prop_assert!(
                    (corr[[i, i]] - 1.0).abs() < 1e-6,
                    "Diagonal [{0},{0}] = {1}, expected 1.0",
                    i,
                    corr[[i, i]]
                );
            }
        }
    }

    #[test]
    fn correlation_values_in_range(rows in 10usize..50, cols in 2usize..8) {
        let data = generate_data_matrix(rows, cols);
        if let Ok(corr) = correlation_matrix(&data) {
            for &val in corr.iter() {
                prop_assert!(
                    (-1.0 - 1e-6..=1.0 + 1e-6).contains(&val),
                    "Correlation {} out of [-1, 1]",
                    val
                );
            }
        }
    }

    #[test]
    fn covariance_matrix_symmetric(rows in 10usize..50, cols in 2usize..8) {
        let data = generate_data_matrix(rows, cols);
        if let Ok(cov) = covariance_matrix(&data) {
            let n = cov.nrows();
            for i in 0..n {
                for j in 0..n {
                    prop_assert!(
                        (cov[[i, j]] - cov[[j, i]]).abs() < 1e-10,
                        "Not symmetric at [{},{}]",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn covariance_diagonal_nonneg(rows in 10usize..50, cols in 2usize..8) {
        let data = generate_data_matrix(rows, cols);
        if let Ok(cov) = covariance_matrix(&data) {
            for i in 0..cov.nrows() {
                prop_assert!(
                    cov[[i, i]] >= -1e-10,
                    "Variance [{0},{0}] = {1} is negative",
                    i,
                    cov[[i, i]]
                );
            }
        }
    }

    #[test]
    fn ewma_output_length(values in proptest::collection::vec(-100.0f64..100.0, 2..200),
                          span in 2usize..50) {
        if let Ok(result) = ewma(&values, span) {
            prop_assert_eq!(result.len(), values.len());
        }
    }

    #[test]
    fn log_returns_of_constant_is_zero(c in 1.0f64..1000.0, n in 2usize..100) {
        let prices = vec![c; n];
        let returns = log_returns(&prices);
        for &r in &returns {
            prop_assert!(
                r.abs() < 1e-10,
                "Log return of constant {} is {}, expected 0",
                c,
                r
            );
        }
    }

    #[test]
    fn matrix_inverse_times_original_is_identity(n in 2usize..5) {
        // Build a well-conditioned matrix
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] = (i + 1) as f64 * 10.0;
            for j in 0..n {
                if i != j {
                    m[[i, j]] = ((i * 3 + j * 7) as f64 * 0.1).sin();
                }
            }
        }

        if let Ok(inv) = matrix_inverse(&m) {
            let product = m.dot(&inv);
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    prop_assert!(
                        (product[[i, j]] - expected).abs() < 1e-6,
                        "Product[{},{}] = {}, expected {}",
                        i,
                        j,
                        product[[i, j]],
                        expected
                    );
                }
            }
        }
    }
}
