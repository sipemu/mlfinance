use mlfinance_sampling::concurrency::average_uniqueness::average_uniqueness;
use mlfinance_sampling::concurrency::indicator_matrix::get_indicator_matrix;
use mlfinance_sampling::fracdiff::ffd::frac_diff_ffd;
use mlfinance_sampling::fracdiff::weights::get_weights_ffd;
use mlfinance_sampling::weights::class_weights::balanced_class_weights;
use mlfinance_sampling::weights::time_decay::time_decay;
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
fn test_ffd_weights_match_python() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("ffd_weights.json")).unwrap())
            .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let d = case["d"].as_f64().unwrap();
        let threshold = case["threshold"].as_f64().unwrap();
        let expected: Vec<f64> = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let actual = get_weights_ffd(d, threshold);

        assert_eq!(
            actual.len(),
            expected.len(),
            "Weight count mismatch for d={}, threshold={}",
            d,
            threshold
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "Weight[{}] mismatch for d={}, threshold={}: got {}, expected {}",
                i,
                d,
                threshold,
                a,
                e
            );
        }
    }
}

#[test]
fn test_ffd_series_match_python() {
    let data: Value =
        serde_json::from_str(&fs::read_to_string(fixture_path("ffd_series.json")).unwrap())
            .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let d = case["d"].as_f64().unwrap();
        let threshold = case["threshold"].as_f64().unwrap();
        let series: Vec<f64> = case["series"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected: Vec<Option<f64>> = case["result"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                if v.is_null() {
                    None
                } else {
                    Some(v.as_f64().unwrap())
                }
            })
            .collect();

        let actual = frac_diff_ffd(&series, d, threshold);

        assert_eq!(
            actual.len(),
            expected.len(),
            "Length mismatch for d={}, threshold={}",
            d,
            threshold
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            match e {
                None => {
                    assert!(
                        a.is_nan(),
                        "Expected NaN at index {} for d={}, threshold={}, got {}",
                        i,
                        d,
                        threshold,
                        a
                    );
                }
                Some(ev) => {
                    assert!(
                        (a - ev).abs() < 1e-10,
                        "Value mismatch at index {} for d={}, threshold={}: got {}, expected {}",
                        i,
                        d,
                        threshold,
                        a,
                        ev
                    );
                }
            }
        }
    }
}

#[test]
fn test_seq_bootstrap_uniqueness_match_python() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("seq_bootstrap_draws.json")).unwrap(),
    )
    .unwrap();

    let events: Vec<(usize, usize)> = data["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let arr = v.as_array().unwrap();
            (
                arr[0].as_u64().unwrap() as usize,
                arr[1].as_u64().unwrap() as usize,
            )
        })
        .collect();

    let num_bars = data["num_bars"].as_u64().unwrap() as usize;

    let expected_uniq: Vec<f64> = data["avg_uniqueness"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let actual_uniq = average_uniqueness(&events, num_bars);

    assert_eq!(actual_uniq.len(), expected_uniq.len());
    for (i, (a, e)) in actual_uniq.iter().zip(expected_uniq.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "Uniqueness[{}] mismatch: got {}, expected {}",
            i,
            a,
            e
        );
    }
}

#[test]
fn test_indicator_matrix_structure() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("seq_bootstrap_draws.json")).unwrap(),
    )
    .unwrap();

    let events: Vec<(usize, usize)> = data["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let arr = v.as_array().unwrap();
            (
                arr[0].as_u64().unwrap() as usize,
                arr[1].as_u64().unwrap() as usize,
            )
        })
        .collect();

    let num_bars = data["num_bars"].as_u64().unwrap() as usize;

    let matrix = get_indicator_matrix(&events, num_bars);
    assert_eq!(matrix.shape(), &[num_bars, events.len()]);

    // Verify each event column has 1s in the correct range
    for (j, &(start, end)) in events.iter().enumerate() {
        for t in 0..num_bars {
            let expected = if t >= start && t <= end { 1.0 } else { 0.0 };
            assert_eq!(
                matrix[[t, j]],
                expected,
                "Matrix[{}, {}] mismatch for event ({}, {})",
                t,
                j,
                start,
                end
            );
        }
    }
}

// =====================================================================
// P1 — Sampling extended reference tests
// =====================================================================

#[test]
fn test_time_decay_match_python() {
    let data = load_fixture("sampling_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "time_decay" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let weights = parse_f64_array(&case["weights"]);
        let oldest_weight = case["oldest_weight"].as_f64().unwrap();
        let expected = parse_f64_array(&case["result"]);

        let actual = time_decay(&weights, oldest_weight);
        assert_eq!(
            actual.len(),
            expected.len(),
            "Time decay length mismatch for '{}': got={}, expected={}",
            label,
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "Time decay[{}] mismatch for '{}': got={}, expected={}",
                i,
                label,
                a,
                e
            );
        }
    }
}

#[test]
fn test_balanced_class_weights_match_python() {
    let data = load_fixture("sampling_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "balanced_class_weights" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let labels: Vec<i32> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let expected_weights = case["weights"].as_object().unwrap();

        let actual = balanced_class_weights(&labels);

        assert_eq!(
            actual.len(),
            expected_weights.len(),
            "Class weights count mismatch for '{}': got={}, expected={}",
            label,
            actual.len(),
            expected_weights.len()
        );

        for (key, val) in expected_weights {
            let class_label: i32 = key.parse().unwrap();
            let expected_w = val.as_f64().unwrap();
            let actual_w = actual.get(&class_label).unwrap_or_else(|| {
                panic!("Missing class {} in weights for '{}'", class_label, label)
            });
            assert!(
                (actual_w - expected_w).abs() < 1e-10,
                "Class weight[{}] mismatch for '{}': got={}, expected={}",
                class_label,
                label,
                actual_w,
                expected_w
            );
        }
    }
}
