use chrono::Utc;
use mlfinance_core::types::{OhlcvBar, Timestamp};
use mlfinance_labeling::barriers::{find_first_touch, BarrierTouchType, TripleBarrierConfig};
use mlfinance_labeling::events::get_events;
use mlfinance_labeling::volatility::{
    daily_volatility, garman_klass_volatility, parkinson_volatility, yang_zhang_volatility,
};
use serde_json::Value;
use std::fs;

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
}

#[test]
fn test_triple_barrier_events_match_python() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("triple_barrier_events.json")).unwrap(),
    )
    .unwrap();

    let prices: Vec<f64> = data["prices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let entry_indices: Vec<usize> = data["entry_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let upper = data["upper_barrier"].as_f64().unwrap();
    let lower = data["lower_barrier"].as_f64().unwrap();
    let max_hold = data["max_holding_period"].as_u64().unwrap() as usize;
    let daily_vols: Vec<f64> = data["daily_vols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected_events = data["events"].as_array().unwrap();

    let config = TripleBarrierConfig {
        upper_barrier: Some(upper),
        lower_barrier: Some(lower),
        max_holding_period: Some(max_hold),
    };

    let actual_events = get_events(&prices, &entry_indices, &config, &daily_vols);

    assert_eq!(
        actual_events.len(),
        expected_events.len(),
        "Event count mismatch"
    );

    for (i, (actual, expected)) in actual_events.iter().zip(expected_events.iter()).enumerate() {
        let exp_entry = expected["entry_idx"].as_u64().unwrap() as usize;
        let exp_exit = expected["exit_idx"].as_u64().unwrap() as usize;
        let exp_touch = expected["touch_type"].as_str().unwrap();
        let exp_return = expected["return_value"].as_f64().unwrap();

        assert_eq!(
            actual.entry_idx, exp_entry,
            "Event {} entry_idx mismatch",
            i
        );
        assert_eq!(
            actual.exit_idx, exp_exit,
            "Event {} exit_idx mismatch: got {}, expected {}",
            i, actual.exit_idx, exp_exit
        );

        let actual_touch = match actual.touch_type {
            BarrierTouchType::Upper => "Upper",
            BarrierTouchType::Lower => "Lower",
            BarrierTouchType::Vertical => "Vertical",
        };
        assert_eq!(actual_touch, exp_touch, "Event {} touch_type mismatch", i);

        assert!(
            (actual.return_value - exp_return).abs() < 1e-6,
            "Event {} return mismatch: got {}, expected {}",
            i,
            actual.return_value,
            exp_return
        );
    }
}

#[test]
fn test_find_first_touch_individual() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("triple_barrier_events.json")).unwrap(),
    )
    .unwrap();

    let prices: Vec<f64> = data["prices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let upper = data["upper_barrier"].as_f64().unwrap();
    let lower = data["lower_barrier"].as_f64().unwrap();
    let max_hold = data["max_holding_period"].as_u64().unwrap() as usize;
    let daily_vols: Vec<f64> = data["daily_vols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected_events = data["events"].as_array().unwrap();

    let config = TripleBarrierConfig {
        upper_barrier: Some(upper),
        lower_barrier: Some(lower),
        max_holding_period: Some(max_hold),
    };

    for expected in expected_events {
        let entry_idx = expected["entry_idx"].as_u64().unwrap() as usize;
        let exp_exit = expected["exit_idx"].as_u64().unwrap() as usize;
        let exp_touch = expected["touch_type"].as_str().unwrap();

        let touch = find_first_touch(&prices, entry_idx, &config, daily_vols[entry_idx]);
        assert!(
            touch.is_some(),
            "No touch found for entry_idx={}",
            entry_idx
        );

        let touch = touch.unwrap();
        assert_eq!(
            touch.timestamp_index, exp_exit,
            "Exit mismatch for entry_idx={}",
            entry_idx
        );

        let actual_touch = match touch.touch_type {
            BarrierTouchType::Upper => "Upper",
            BarrierTouchType::Lower => "Lower",
            BarrierTouchType::Vertical => "Vertical",
        };
        assert_eq!(
            actual_touch, exp_touch,
            "Touch type mismatch for entry_idx={}",
            entry_idx
        );
    }
}

// --- Volatility reference tests ---

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

fn make_timestamps(n: usize) -> Vec<Timestamp> {
    (0..n).map(|_| Utc::now()).collect()
}

fn bars_from_fixture(val: &Value) -> Vec<OhlcvBar> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|b| OhlcvBar {
            timestamp: Utc::now(),
            open: b["open"].as_f64().unwrap(),
            high: b["high"].as_f64().unwrap(),
            low: b["low"].as_f64().unwrap(),
            close: b["close"].as_f64().unwrap(),
            volume: 1000.0,
            vwap: (b["high"].as_f64().unwrap()
                + b["low"].as_f64().unwrap()
                + b["close"].as_f64().unwrap())
                / 3.0,
        })
        .collect()
}

#[test]
fn test_daily_volatility_match_python() {
    let data = load_fixture("daily_volatility.json");

    for case in data["cases"].as_array().unwrap() {
        let prices = parse_f64_array(&case["prices"]);
        let span = case["span"].as_u64().unwrap() as usize;
        let expected = parse_f64_array(&case["result"]);
        let ts = make_timestamps(prices.len());

        let actual = daily_volatility(&prices, &ts, span);
        assert_eq!(
            actual.len(),
            expected.len(),
            "Daily vol length mismatch for span={}",
            span
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "Daily vol[{}] mismatch for span={}: got={}, expected={}",
                i,
                span,
                a,
                e
            );
        }
    }
}

#[test]
fn test_parkinson_volatility_match_python() {
    let data = load_fixture("volatility_estimators.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "parkinson" {
            continue;
        }
        let bars = bars_from_fixture(&case["bars"]);
        let window = case["window"].as_u64().unwrap() as usize;
        let expected_raw = case["result"].as_array().unwrap();

        let actual = parkinson_volatility(&bars, window).unwrap();
        assert_eq!(actual.len(), expected_raw.len());
        for (i, (a, e_val)) in actual.iter().zip(expected_raw.iter()).enumerate() {
            if e_val.is_null() {
                assert!(
                    a.is_nan(),
                    "Parkinson[{}] should be NaN for window={}",
                    i,
                    window
                );
            } else {
                let e = e_val.as_f64().unwrap();
                assert!(
                    (a - e).abs() < 1e-10,
                    "Parkinson[{}] mismatch for window={}: got={}, expected={}",
                    i,
                    window,
                    a,
                    e
                );
            }
        }
    }
}

#[test]
fn test_garman_klass_volatility_match_python() {
    let data = load_fixture("volatility_estimators.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "garman_klass" {
            continue;
        }
        let bars = bars_from_fixture(&case["bars"]);
        let window = case["window"].as_u64().unwrap() as usize;
        let expected_raw = case["result"].as_array().unwrap();

        let actual = garman_klass_volatility(&bars, window).unwrap();
        assert_eq!(actual.len(), expected_raw.len());
        for (i, (a, e_val)) in actual.iter().zip(expected_raw.iter()).enumerate() {
            if e_val.is_null() {
                assert!(a.is_nan(), "GK[{}] should be NaN for window={}", i, window);
            } else {
                let e = e_val.as_f64().unwrap();
                assert!(
                    (a - e).abs() < 1e-10,
                    "GK[{}] mismatch for window={}: got={}, expected={}",
                    i,
                    window,
                    a,
                    e
                );
            }
        }
    }
}

#[test]
fn test_yang_zhang_volatility_match_python() {
    let data = load_fixture("volatility_estimators.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "yang_zhang" {
            continue;
        }
        let bars = bars_from_fixture(&case["bars"]);
        let window = case["window"].as_u64().unwrap() as usize;
        let expected_raw = case["result"].as_array().unwrap();

        let actual = yang_zhang_volatility(&bars, window).unwrap();
        assert_eq!(actual.len(), expected_raw.len());
        for (i, (a, e_val)) in actual.iter().zip(expected_raw.iter()).enumerate() {
            if e_val.is_null() {
                assert!(a.is_nan(), "YZ[{}] should be NaN for window={}", i, window);
            } else {
                let e = e_val.as_f64().unwrap();
                assert!(
                    (a - e).abs() < 1e-10,
                    "YZ[{}] mismatch for window={}: got={}, expected={}",
                    i,
                    window,
                    a,
                    e
                );
            }
        }
    }
}
