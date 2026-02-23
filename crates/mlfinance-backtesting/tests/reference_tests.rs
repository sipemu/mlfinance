use mlfinance_backtesting::bet_sizing::probability_to_size::{power_bet_size, sigmoid_bet_size};
use serde_json::Value;
use std::fs;

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
}

#[test]
fn test_bet_sizing_match_python() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("bet_sizing.json")).unwrap())
            .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let case_type = case["type"].as_str().unwrap();
        let prob = case["prob"].as_f64().unwrap();
        let num_classes = case["num_classes"].as_u64().unwrap() as usize;
        let expected = case["result"].as_f64().unwrap();

        match case_type {
            "sigmoid" => {
                let actual = sigmoid_bet_size(prob, num_classes);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Sigmoid mismatch: prob={}, classes={}, got={}, expected={}",
                    prob,
                    num_classes,
                    actual,
                    expected
                );
            }
            "power" => {
                let exponent = case["exponent"].as_f64().unwrap();
                let actual = power_bet_size(prob, num_classes, exponent);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Power mismatch: prob={}, classes={}, exp={}, got={}, expected={}",
                    prob,
                    num_classes,
                    exponent,
                    actual,
                    expected
                );
            }
            _ => panic!("Unknown case type: {}", case_type),
        }
    }
}
