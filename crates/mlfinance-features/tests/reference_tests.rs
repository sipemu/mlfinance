use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance_features::entropy::plugin::plugin_entropy;
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
