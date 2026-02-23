use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance_features::entropy::plugin::plugin_entropy;
use ndarray::Array2;
use serde_json::Value;
use std::fs;

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
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
