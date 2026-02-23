use mlfinance_data::sampling::cusum_filter::cusum_filter;
use proptest::prelude::*;

#[test]
fn cusum_empty_input() {
    let events = cusum_filter(&[], 1.0);
    assert!(events.is_empty());
}

#[test]
fn cusum_single_value() {
    let events = cusum_filter(&[100.0], 1.0);
    assert!(events.is_empty());
}

#[test]
fn cusum_threshold_zero_every_tick() {
    let values = vec![100.0, 100.001, 100.002, 100.003, 100.004];
    let events = cusum_filter(&values, 0.0);
    // With threshold=0, every positive change triggers an event
    // diff[1..4] are all positive (0.001), so s_pos accumulates to >= 0 after each
    assert!(!events.is_empty());
}

#[test]
fn cusum_threshold_very_large() {
    let values = vec![100.0, 101.0, 102.0, 103.0];
    let events = cusum_filter(&values, f64::MAX);
    assert!(events.is_empty(), "Infinite threshold should never trigger");
}

#[test]
fn cusum_nan_input_no_panic() {
    let values = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let _ = cusum_filter(&values, 1.0); // Should not panic
}

#[test]
fn cusum_inf_input_no_panic() {
    let values = vec![1.0, f64::INFINITY, 3.0, 4.0];
    let _ = cusum_filter(&values, 1.0); // Should not panic
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn cusum_never_panics(
        values in proptest::collection::vec(-1e6f64..1e6, 0..1000),
        threshold in 0.0f64..1e6
    ) {
        let _ = cusum_filter(&values, threshold);
    }
}
