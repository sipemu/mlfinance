use mlfinance_data::sampling::cusum_filter::cusum_filter;
use serde_json::Value;
use std::fs;

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
}

#[test]
fn test_cusum_filter_match_python() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("cusum_events.json")).unwrap())
            .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let values: Vec<f64> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let threshold = case["threshold"].as_f64().unwrap();
        let expected: Vec<usize> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let actual = cusum_filter(&values, threshold);

        assert_eq!(
            actual, expected,
            "CUSUM events mismatch for threshold={}. Got {:?}, expected {:?}",
            threshold, actual, expected
        );
    }
}

#[test]
fn test_cusum_events_strictly_increasing() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("cusum_events.json")).unwrap())
            .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let events: Vec<usize> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        for window in events.windows(2) {
            assert!(
                window[0] < window[1],
                "CUSUM events not strictly increasing: {} >= {}",
                window[0],
                window[1]
            );
        }
    }
}
