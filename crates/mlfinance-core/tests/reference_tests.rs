use mlfinance_core::stats::{correlation_matrix, covariance_matrix, kurtosis, skewness};
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
fn test_skewness_match_python() {
    let data = load_fixture("core_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "univariate" {
            continue;
        }
        let values = parse_f64_array(&case["data"]);
        let expected = case["skewness"].as_f64().unwrap();
        let label = case["label"].as_str().unwrap();

        let actual = skewness(&values).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "Skewness mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_kurtosis_match_python() {
    let data = load_fixture("core_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "univariate" {
            continue;
        }
        let values = parse_f64_array(&case["data"]);
        let expected = case["kurtosis"].as_f64().unwrap();
        let label = case["label"].as_str().unwrap();

        // Need at least 4 elements for kurtosis
        if values.len() < 4 {
            continue;
        }

        let actual = kurtosis(&values).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "Kurtosis mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_correlation_matrix_match_python() {
    let data = load_fixture("core_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "matrix" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let n_rows = case["n_rows"].as_u64().unwrap() as usize;
        let n_cols = case["n_cols"].as_u64().unwrap() as usize;
        let flat: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let matrix = Array2::from_shape_vec((n_rows, n_cols), flat).unwrap();

        let expected: Vec<Vec<f64>> = case["correlation_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();

        let actual = correlation_matrix(&matrix).unwrap();

        for i in 0..n_cols {
            for j in 0..n_cols {
                assert!(
                    (actual[[i, j]] - expected[i][j]).abs() < 1e-10,
                    "Correlation[{},{}] mismatch for '{}': got={}, expected={}",
                    i,
                    j,
                    label,
                    actual[[i, j]],
                    expected[i][j]
                );
            }
        }
    }
}

#[test]
fn test_covariance_matrix_match_python() {
    let data = load_fixture("core_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "matrix" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let n_rows = case["n_rows"].as_u64().unwrap() as usize;
        let n_cols = case["n_cols"].as_u64().unwrap() as usize;
        let flat: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let matrix = Array2::from_shape_vec((n_rows, n_cols), flat).unwrap();

        let expected: Vec<Vec<f64>> = case["covariance_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();

        let actual = covariance_matrix(&matrix).unwrap();

        for i in 0..n_cols {
            for j in 0..n_cols {
                assert!(
                    (actual[[i, j]] - expected[i][j]).abs() < 1e-10,
                    "Covariance[{},{}] mismatch for '{}': got={}, expected={}",
                    i,
                    j,
                    label,
                    actual[[i, j]],
                    expected[i][j]
                );
            }
        }
    }
}
