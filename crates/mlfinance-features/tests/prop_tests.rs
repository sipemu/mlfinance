use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance_features::entropy::plugin::plugin_entropy;
use mlfinance_features::structural_breaks::sadf::{sadf, sadf_stat};
use ndarray::Array2;
use proptest::prelude::*;

fn generate_returns(n_obs: usize, n_assets: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_obs, n_assets), |(i, j)| {
        ((i * 7 + j * 13 + 1) as f64 * 0.7).sin() * 0.01
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn hrp_weights_sum_to_one(n_assets in 2usize..10) {
        let returns = generate_returns(100, n_assets);
        if let Ok(weights) = hrp_weights(&returns) {
            let sum: f64 = weights.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "Weights sum = {}, expected 1.0",
                sum
            );
        }
    }

    #[test]
    fn hrp_weights_all_positive(n_assets in 2usize..10) {
        let returns = generate_returns(100, n_assets);
        if let Ok(weights) = hrp_weights(&returns) {
            for (i, &w) in weights.iter().enumerate() {
                prop_assert!(w > -1e-10, "Weight[{}] = {} should be positive", i, w);
            }
        }
    }

    #[test]
    fn plugin_entropy_in_range(n_symbols in 2usize..20, n in 10usize..200) {
        let sequence: Vec<usize> = (0..n).map(|i| i % n_symbols).collect();
        let entropy = plugin_entropy(&sequence, n_symbols);
        let max_entropy = (n_symbols as f64).log2();
        prop_assert!(
            entropy >= -1e-10 && entropy <= max_entropy + 1e-10,
            "Entropy {} out of [0, {}]",
            entropy,
            max_entropy
        );
    }

    #[test]
    fn lempel_ziv_in_range(n in 1usize..200) {
        let binary: Vec<bool> = (0..n).map(|i| (i * 7 + 3) % 5 < 3).collect();
        let complexity = lempel_ziv_complexity(&binary);
        prop_assert!(complexity >= 1, "LZ complexity {} < 1", complexity);
        prop_assert!(complexity <= n, "LZ complexity {} > n={}", complexity, n);
    }

    #[test]
    fn sadf_supremum_is_max(n in 40usize..80) {
        let series: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.05).sin() * 10.0 + 100.0)
            .collect();
        let min_window = 20;
        let max_lags = 3;
        let stats = sadf(&series, min_window, max_lags);
        if !stats.is_empty() {
            let stat = sadf_stat(&series, min_window, max_lags);
            let max_val = stats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!(
                (stat - max_val).abs() < 1e-10,
                "sadf_stat {} != max of stats {}",
                stat,
                max_val
            );
        }
    }
}
