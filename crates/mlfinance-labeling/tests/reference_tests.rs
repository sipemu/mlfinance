use mlfinance_labeling::barriers::{find_first_touch, BarrierTouchType, TripleBarrierConfig};
use mlfinance_labeling::events::get_events;
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
