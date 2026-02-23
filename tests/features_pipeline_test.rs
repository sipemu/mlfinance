//! Feature pipeline test: synthetic returns → HRP weights → correlation matrix
//! → entropy of discretized returns → SADF on cumulated returns

use mlfinance::core::math::cumsum;
use mlfinance::core::stats::correlation_matrix;
use mlfinance::features::allocation::hrp::hrp::hrp_weights;
use mlfinance::features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance::features::entropy::plugin::plugin_entropy;
use mlfinance::features::structural_breaks::sadf::{sadf, sadf_stat};
use ndarray::Array2;

fn generate_correlated_returns(n_obs: usize, n_assets: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_obs, n_assets), |(i, j)| {
        let base = ((i * 7 + 1) as f64 * 0.3).sin() * 0.01;
        let idio = ((i * 13 + j * 31 + 1) as f64 * 0.7).sin() * 0.005;
        base + idio
    })
}

#[test]
fn test_features_pipeline() {
    let n_obs = 200;
    let n_assets = 5;

    // Stage 1: Generate correlated returns
    let returns = generate_correlated_returns(n_obs, n_assets);
    assert_eq!(returns.shape(), &[n_obs, n_assets]);

    // Stage 2: HRP portfolio allocation
    let weights = hrp_weights(&returns).expect("HRP should succeed");

    // Weights must sum to 1.0
    let weight_sum: f64 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-4,
        "HRP weights sum = {}, expected 1.0",
        weight_sum
    );

    // All weights must be positive
    for (i, &w) in weights.iter().enumerate() {
        assert!(w > 0.0, "Weight[{}] = {} should be positive", i, w);
    }

    // Stage 3: Correlation matrix
    let corr = correlation_matrix(&returns).expect("Correlation should succeed");
    assert_eq!(corr.shape(), &[n_assets, n_assets]);

    // Correlation bounds
    for &val in corr.iter() {
        assert!(
            val >= -1.0 - 1e-6 && val <= 1.0 + 1e-6,
            "Correlation {} out of [-1, 1]",
            val
        );
    }
    // Diagonal should be 1
    for i in 0..n_assets {
        assert!((corr[[i, i]] - 1.0).abs() < 1e-6);
    }

    // Stage 4: Discretize returns for entropy
    // Convert first asset's returns to discrete symbols (positive/negative)
    let first_col: Vec<f64> = (0..n_obs).map(|i| returns[[i, 0]]).collect();
    let discrete: Vec<usize> = first_col
        .iter()
        .map(|&r| if r >= 0.0 { 1 } else { 0 })
        .collect();

    let entropy = plugin_entropy(&discrete, 2);
    assert!(
        entropy >= 0.0,
        "Entropy should be non-negative, got {}",
        entropy
    );
    assert!(
        entropy <= 1.0 + 1e-6,
        "Binary entropy should be <= 1.0, got {}",
        entropy
    );

    // Stage 5: LZ complexity
    let binary: Vec<bool> = first_col.iter().map(|&r| r >= 0.0).collect();
    let lz = lempel_ziv_complexity(&binary);
    assert!(lz >= 1, "LZ complexity should be >= 1");
    assert!(lz <= n_obs, "LZ complexity should be <= n");

    // Stage 6: SADF on cumulated returns
    let cum_returns = cumsum(&first_col);
    // Add a base level to avoid near-zero values
    let price_like: Vec<f64> = cum_returns.iter().map(|&v| 100.0 + v * 1000.0).collect();

    let min_window = 30;
    let max_lags = 3;
    let stats = sadf(&price_like, min_window, max_lags);
    assert!(
        !stats.is_empty(),
        "SADF should produce statistics"
    );
    for &s in &stats {
        assert!(s.is_finite(), "SADF stat should be finite, got {}", s);
    }

    let sup = sadf_stat(&price_like, min_window, max_lags);
    assert!(sup.is_finite(), "SADF supremum should be finite");
}
