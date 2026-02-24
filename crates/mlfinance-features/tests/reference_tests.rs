use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::allocation::ivp::inverse_variance_weights;
use mlfinance_features::codependence::correlation::{
    absolute_angular_distance, angular_distance, distance_correlation, squared_angular_distance,
};
use mlfinance_features::codependence::information::{mutual_information, variation_of_information};
use mlfinance_features::entropy::gaussian_entropy::{entropy_implied_vol, gaussian_entropy};
use mlfinance_features::entropy::kontoyiannis::kontoyiannis_entropy;
use mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance_features::entropy::plugin::plugin_entropy;
use mlfinance_features::entropy::shannon::shannon_entropy;
use mlfinance_features::microstructure::amihud_lambda::amihud_lambda;
use mlfinance_features::microstructure::corwin_schultz::corwin_schultz_spread;
use mlfinance_features::microstructure::kyle_lambda::kyle_lambda;
use mlfinance_features::microstructure::roll_model::roll_spread;
use mlfinance_features::microstructure::tick_rule::tick_rule_classify;
use mlfinance_features::microstructure::vpin::vpin;
use mlfinance_features::structural_breaks::adf::adf_test;
use mlfinance_features::structural_breaks::cusum_tests::{
    brown_durbin_evans, chu_stinchcombe_white,
};
use mlfinance_features::structural_breaks::gsadf::{gsadf, gsadf_stat};
use mlfinance_features::structural_breaks::sadf::{sadf, sadf_stat};
use ndarray::Array2;
use serde_json::Value;
use std::fs;

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
}

fn load_fixture(name: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(fixture_path(name)).unwrap()).unwrap()
}

fn parse_f64_array(val: &Value) -> Vec<f64> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[test]
fn test_plugin_entropy_match_python() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("entropy.json")).unwrap()).unwrap();

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "plugin" {
            continue;
        }
        let sequence: Vec<usize> = case["sequence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let num_symbols = case["num_symbols"].as_u64().unwrap() as usize;
        let expected = case["result"].as_f64().unwrap();

        let actual = plugin_entropy(&sequence, num_symbols);

        assert!(
            (actual - expected).abs() < 1e-10,
            "Plugin entropy mismatch: seq_len={}, num_sym={}, got={}, expected={}",
            sequence.len(),
            num_symbols,
            actual,
            expected
        );
    }
}

#[test]
fn test_lempel_ziv_match_python() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("entropy.json")).unwrap()).unwrap();

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "lempel_ziv" {
            continue;
        }
        let binary_string: Vec<bool> = case["binary_string"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let expected = case["result"].as_u64().unwrap() as usize;

        let actual = lempel_ziv_complexity(&binary_string);

        assert_eq!(
            actual,
            expected,
            "LZ complexity mismatch: input_len={}, got={}, expected={}",
            binary_string.len(),
            actual,
            expected
        );
    }
}

#[test]
fn test_hrp_weights_properties() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("hrp_weights.json")).unwrap())
            .unwrap();

    let n_obs = data["n_obs"].as_u64().unwrap() as usize;
    let n_assets = data["n_assets"].as_u64().unwrap() as usize;
    let returns_flat: Vec<f64> = data["returns"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect();

    let returns = Array2::from_shape_vec((n_obs, n_assets), returns_flat).unwrap();

    let weights = hrp_weights(&returns).expect("HRP should succeed");

    // Weights must sum to 1.0 (within tolerance)
    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "HRP weights sum = {}, expected 1.0",
        sum
    );

    // All weights must be positive
    for (i, &w) in weights.iter().enumerate() {
        assert!(w > 0.0, "HRP weight[{}] = {} should be positive", i, w);
    }

    // Correct number of weights
    assert_eq!(weights.len(), n_assets);
}

// --- Structural breaks reference tests ---

#[test]
fn test_sadf_match_python() {
    let data = load_fixture("sadf_stats.json");
    let series = parse_f64_array(&data["series"]);
    let min_window = data["min_window"].as_u64().unwrap() as usize;
    let max_lags = data["max_lags"].as_u64().unwrap() as usize;
    let expected_stats = parse_f64_array(&data["stats"]);

    let actual_stats = sadf(&series, min_window, max_lags);

    assert_eq!(
        actual_stats.len(),
        expected_stats.len(),
        "SADF stats length mismatch: got {}, expected {}",
        actual_stats.len(),
        expected_stats.len()
    );
    for (i, (actual, expected)) in actual_stats.iter().zip(expected_stats.iter()).enumerate() {
        if expected.is_nan() {
            continue;
        }
        assert!(
            (actual - expected).abs() < 1e-4 || actual.is_nan(),
            "SADF stats[{}] mismatch: got={}, expected={}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_sadf_stat_match_python() {
    let data = load_fixture("sadf_stats.json");
    let series = parse_f64_array(&data["series"]);
    let min_window = data["min_window"].as_u64().unwrap() as usize;
    let max_lags = data["max_lags"].as_u64().unwrap() as usize;
    let expected = data["sadf_stat"].as_f64().unwrap();

    let actual = sadf_stat(&series, min_window, max_lags);
    assert!(
        (actual - expected).abs() < 1e-4,
        "SADF stat mismatch: got={}, expected={}",
        actual,
        expected
    );
}

#[test]
fn test_adf_match_python() {
    let data = load_fixture("structural_breaks.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "adf" {
            continue;
        }
        let series = parse_f64_array(&case["series"]);
        let max_lags = case["max_lags"].as_u64().unwrap() as usize;
        let expected_stat = case["adf_stat"].as_f64().unwrap();
        let expected_n_betas = case["n_betas"].as_u64().unwrap() as usize;

        let (actual_stat, betas) = adf_test(&series, max_lags);
        assert!(
            (actual_stat - expected_stat).abs() < 1e-4,
            "ADF stat mismatch: got={}, expected={}",
            actual_stat,
            expected_stat
        );
        assert_eq!(
            betas.len(),
            expected_n_betas,
            "ADF betas length mismatch: got={}, expected={}",
            betas.len(),
            expected_n_betas
        );
    }
}

#[test]
fn test_brown_durbin_evans_match_python() {
    let data = load_fixture("structural_breaks.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "brown_durbin_evans" {
            continue;
        }
        let residuals = parse_f64_array(&case["residuals"]);
        let expected_cusum = parse_f64_array(&case["cusum"]);
        let expected_critical = case["critical"].as_f64().unwrap();

        let (actual_cusum, actual_critical) = brown_durbin_evans(&residuals);
        assert_eq!(actual_cusum.len(), expected_cusum.len());
        for (i, (a, e)) in actual_cusum.iter().zip(expected_cusum.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "BDE cusum[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
        assert!(
            (actual_critical - expected_critical).abs() < 1e-10,
            "BDE critical mismatch: got={}, expected={}",
            actual_critical,
            expected_critical
        );
    }
}

#[test]
fn test_chu_stinchcombe_white_match_python() {
    let data = load_fixture("structural_breaks.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "chu_stinchcombe_white" {
            continue;
        }
        let log_prices = parse_f64_array(&case["log_prices"]);
        let critical_value = case["critical_value"].as_f64().unwrap();
        let expected_stats = parse_f64_array(&case["stats"]);

        let actual_stats = chu_stinchcombe_white(&log_prices, critical_value);
        assert_eq!(
            actual_stats.len(),
            expected_stats.len(),
            "CSW stats length mismatch: got {}, expected {}",
            actual_stats.len(),
            expected_stats.len()
        );
        for (i, (a, e)) in actual_stats.iter().zip(expected_stats.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "CSW stats[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}

#[test]
fn test_gsadf_stat_match_python() {
    let data = load_fixture("structural_breaks.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "gsadf" {
            continue;
        }
        let series = parse_f64_array(&case["series"]);
        let min_window = case["min_window"].as_u64().unwrap() as usize;
        let max_lags = case["max_lags"].as_u64().unwrap() as usize;
        let expected_stat = case["gsadf_stat"].as_f64().unwrap();

        let actual_stat = gsadf_stat(&series, min_window, max_lags);
        assert!(
            (actual_stat - expected_stat).abs() < 1e-4,
            "GSADF stat mismatch: got={}, expected={}",
            actual_stat,
            expected_stat
        );
    }
}

#[test]
fn test_gsadf_match_python() {
    let data = load_fixture("structural_breaks.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "gsadf" {
            continue;
        }
        let series = parse_f64_array(&case["series"]);
        let min_window = case["min_window"].as_u64().unwrap() as usize;
        let max_lags = case["max_lags"].as_u64().unwrap() as usize;
        let expected_n = case["n_stats"].as_u64().unwrap() as usize;

        let stats = gsadf(&series, min_window, max_lags);
        // The number of finite stats should match (both filter out non-finite)
        assert_eq!(
            stats.len(),
            expected_n,
            "GSADF stats count mismatch: got={}, expected={}",
            stats.len(),
            expected_n
        );
    }
}

// =====================================================================
// P1 — Codependence reference tests
// =====================================================================

#[test]
fn test_angular_distance_match_python() {
    let data = load_fixture("codependence.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let x = parse_f64_array(&case["x"]);
        let y = parse_f64_array(&case["y"]);
        let expected = case["angular_distance"].as_f64().unwrap();

        let actual = angular_distance(&x, &y).unwrap();
        assert!(
            (actual - expected).abs() < 1e-8,
            "Angular distance mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_absolute_angular_distance_match_python() {
    let data = load_fixture("codependence.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let x = parse_f64_array(&case["x"]);
        let y = parse_f64_array(&case["y"]);
        let expected = case["absolute_angular_distance"].as_f64().unwrap();

        let actual = absolute_angular_distance(&x, &y).unwrap();
        assert!(
            (actual - expected).abs() < 1e-8,
            "Absolute angular distance mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_squared_angular_distance_match_python() {
    let data = load_fixture("codependence.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let x = parse_f64_array(&case["x"]);
        let y = parse_f64_array(&case["y"]);
        let expected = case["squared_angular_distance"].as_f64().unwrap();

        let actual = squared_angular_distance(&x, &y).unwrap();
        assert!(
            (actual - expected).abs() < 1e-8,
            "Squared angular distance mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_distance_correlation_match_python() {
    let data = load_fixture("codependence.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let x = parse_f64_array(&case["x"]);
        let y = parse_f64_array(&case["y"]);
        let expected = case["distance_correlation"].as_f64().unwrap();

        let actual = distance_correlation(&x, &y).unwrap();
        assert!(
            (actual - expected).abs() < 1e-8,
            "Distance correlation mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_mutual_information_match_python() {
    let data = load_fixture("codependence.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let x = parse_f64_array(&case["x"]);
        let y = parse_f64_array(&case["y"]);
        let n_bins = case["n_bins"].as_u64().unwrap() as usize;
        let expected = case["mutual_information"].as_f64().unwrap();

        let actual = mutual_information(&x, &y, Some(n_bins), false).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mutual information mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_variation_of_information_match_python() {
    let data = load_fixture("codependence.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let x = parse_f64_array(&case["x"]);
        let y = parse_f64_array(&case["y"]);
        let n_bins = case["n_bins"].as_u64().unwrap() as usize;
        let expected = case["variation_of_information"].as_f64().unwrap();

        let actual = variation_of_information(&x, &y, Some(n_bins), false).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "Variation of information mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

// =====================================================================
// P1 — Microstructure reference tests
// =====================================================================

#[test]
fn test_amihud_lambda_match_python() {
    let data = load_fixture("microstructure.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "amihud_lambda" {
            continue;
        }
        let returns = parse_f64_array(&case["returns"]);
        let dollar_volumes = parse_f64_array(&case["dollar_volumes"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = amihud_lambda(&returns, &dollar_volumes);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Amihud lambda mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_kyle_lambda_match_python() {
    let data = load_fixture("microstructure.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "kyle_lambda" {
            continue;
        }
        let returns = parse_f64_array(&case["returns"]);
        let signed_volume = parse_f64_array(&case["signed_volume"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = kyle_lambda(&returns, &signed_volume);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Kyle lambda mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_roll_spread_match_python() {
    let data = load_fixture("microstructure.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "roll_spread" {
            continue;
        }
        let prices = parse_f64_array(&case["prices"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = roll_spread(&prices);
        assert!(
            (actual - expected).abs() < 1e-6,
            "Roll spread mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_tick_rule_match_python() {
    let data = load_fixture("microstructure.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "tick_rule" {
            continue;
        }
        let prices = parse_f64_array(&case["prices"]);
        let expected = parse_f64_array(&case["result"]);

        let actual = tick_rule_classify(&prices);
        assert_eq!(
            actual.len(),
            expected.len(),
            "Tick rule length mismatch: got={}, expected={}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "Tick rule[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}

#[test]
fn test_corwin_schultz_match_python() {
    let data = load_fixture("microstructure.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "corwin_schultz" {
            continue;
        }
        let highs = parse_f64_array(&case["highs"]);
        let lows = parse_f64_array(&case["lows"]);
        let expected = parse_f64_array(&case["result"]);

        let actual = corwin_schultz_spread(&highs, &lows);
        assert_eq!(
            actual.len(),
            expected.len(),
            "Corwin-Schultz length mismatch: got={}, expected={}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "Corwin-Schultz[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}

#[test]
fn test_vpin_match_python() {
    let data = load_fixture("microstructure.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "vpin" {
            continue;
        }
        let prices = parse_f64_array(&case["prices"]);
        let volumes = parse_f64_array(&case["volumes"]);
        let bucket_size = case["bucket_size"].as_f64().unwrap();
        let n_buckets = case["n_buckets"].as_u64().unwrap() as usize;
        let expected = parse_f64_array(&case["result"]);

        let actual = vpin(&volumes, &prices, bucket_size, n_buckets);
        assert_eq!(
            actual.len(),
            expected.len(),
            "VPIN length mismatch: got={}, expected={}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "VPIN[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}

// =====================================================================
// P1 — Entropy extended reference tests
// =====================================================================

#[test]
fn test_shannon_entropy_match_python() {
    let data = load_fixture("entropy_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "shannon_entropy" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let probs = parse_f64_array(&case["probs"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = shannon_entropy(&probs);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Shannon entropy mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_kontoyiannis_entropy_match_python() {
    let data = load_fixture("entropy_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "kontoyiannis_entropy" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let sequence: Vec<usize> = case["sequence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let window = case["window"].as_u64().unwrap() as usize;
        let expected = case["result"].as_f64().unwrap();

        let actual = kontoyiannis_entropy(&sequence, window);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Kontoyiannis entropy mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_gaussian_entropy_match_python() {
    let data = load_fixture("entropy_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "gaussian_entropy" {
            continue;
        }
        let variance = case["variance"].as_f64().unwrap();
        let expected = case["entropy"].as_f64().unwrap();

        let actual = gaussian_entropy(variance);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Gaussian entropy mismatch for var={}: got={}, expected={}",
            variance,
            actual,
            expected
        );
    }
}

#[test]
fn test_entropy_implied_vol_match_python() {
    let data = load_fixture("entropy_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "gaussian_entropy" {
            continue;
        }
        let entropy = case["entropy"].as_f64().unwrap();
        let expected = case["implied_vol"].as_f64().unwrap();

        let actual = entropy_implied_vol(entropy);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Entropy implied vol mismatch for H={}: got={}, expected={}",
            entropy,
            actual,
            expected
        );
    }
}

// =====================================================================
// P1 — Portfolio allocation reference tests
// =====================================================================

#[test]
fn test_inverse_variance_weights_match_python() {
    let data = load_fixture("portfolio_allocation.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let cov_flat: Vec<f64> = case["covariance"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let cov = Array2::from_shape_vec((n, n), cov_flat).unwrap();
        let expected = parse_f64_array(&case["weights"]);

        let actual = inverse_variance_weights(&cov);
        assert_eq!(actual.len(), n);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "IVP weight[{}] mismatch for '{}': got={}, expected={}",
                i,
                label,
                a,
                e
            );
        }
    }
}
