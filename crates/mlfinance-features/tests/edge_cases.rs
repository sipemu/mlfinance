use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance_features::entropy::plugin::plugin_entropy;
use ndarray::Array2;
use proptest::prelude::*;

#[test]
fn hrp_two_assets() {
    let returns = Array2::from_shape_fn((50, 2), |(i, j)| {
        ((i * 7 + j * 13) as f64 * 0.7).sin() * 0.01
    });
    let weights = hrp_weights(&returns).unwrap();
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert_eq!(weights.len(), 2);
    assert!(weights[0] > 0.0);
    assert!(weights[1] > 0.0);
}

#[test]
fn hrp_perfectly_correlated_assets_no_crash() {
    let mut returns = Array2::zeros((50, 3));
    for i in 0..50 {
        let val = (i as f64 * 0.1).sin() * 0.01;
        returns[[i, 0]] = val;
        returns[[i, 1]] = val; // perfectly correlated with col 0
        returns[[i, 2]] = val * 0.5 + 0.001; // nearly correlated
    }
    // Should not crash even with perfectly correlated assets
    let _ = hrp_weights(&returns);
}

#[test]
fn hrp_zero_variance_column() {
    let mut returns = Array2::zeros((50, 3));
    for i in 0..50 {
        returns[[i, 0]] = (i as f64 * 0.1).sin() * 0.01;
        // Column 1 has zero variance
        returns[[i, 1]] = 0.0;
        returns[[i, 2]] = (i as f64 * 0.2).cos() * 0.01;
    }
    // Should either succeed or return an error, not panic
    let _ = hrp_weights(&returns);
}

#[test]
fn plugin_entropy_uniform_distribution() {
    // Uniform distribution over n symbols has entropy = log2(n)
    let n = 8;
    let sequence: Vec<usize> = (0..800).map(|i| i % n).collect();
    let entropy = plugin_entropy(&sequence, n);
    let expected = (n as f64).log2();
    assert!(
        (entropy - expected).abs() < 0.01,
        "Uniform entropy: got {}, expected {}",
        entropy,
        expected
    );
}

#[test]
fn plugin_entropy_single_symbol() {
    let sequence = vec![0; 100];
    let entropy = plugin_entropy(&sequence, 2);
    assert!(entropy.abs() < 1e-10, "Single symbol entropy should be 0");
}

#[test]
fn lempel_ziv_all_same() {
    let binary = vec![true; 20];
    let complexity = lempel_ziv_complexity(&binary);
    // Constant string should have low complexity
    assert!(
        complexity <= 3,
        "Constant string should have low LZ complexity, got {}",
        complexity
    );
}

#[test]
fn lempel_ziv_empty() {
    let binary: Vec<bool> = vec![];
    let complexity = lempel_ziv_complexity(&binary);
    assert_eq!(complexity, 0);
}

#[test]
fn lempel_ziv_single_element() {
    let binary = vec![true];
    let complexity = lempel_ziv_complexity(&binary);
    assert_eq!(complexity, 1);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn hrp_never_panics(rows in 2usize..30, cols in 1usize..10) {
        let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64 * 0.7).sin() * 0.01).collect();
        if let Ok(returns) = Array2::from_shape_vec((rows, cols), data) {
            let _ = hrp_weights(&returns);
        }
    }
}
