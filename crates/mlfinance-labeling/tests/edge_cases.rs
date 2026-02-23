use mlfinance_labeling::barriers::{find_first_touch, BarrierTouchType, TripleBarrierConfig};
use mlfinance_labeling::labels::drop_rare_labels;

#[test]
fn find_first_touch_zero_volatility() {
    let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),
        lower_barrier: Some(2.0),
        max_holding_period: Some(3),
    };
    // With vol=0, barriers are at 0: any move triggers immediately
    let touch = find_first_touch(&prices, 0, &config, 0.0);
    if let Some(t) = touch {
        // Upper threshold = 2.0 * 0.0 = 0.0, so any positive return triggers
        assert_eq!(t.touch_type, BarrierTouchType::Upper);
        assert_eq!(t.timestamp_index, 1);
    }
}

#[test]
fn find_first_touch_entry_at_last_index() {
    let prices = vec![100.0, 101.0, 102.0];
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),
        lower_barrier: Some(2.0),
        max_holding_period: Some(5),
    };
    // Entry at last index: no room for any barrier
    let touch = find_first_touch(&prices, 2, &config, 0.01);
    assert!(touch.is_none());
}

#[test]
fn find_first_touch_entry_out_of_bounds() {
    let prices = vec![100.0, 101.0];
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),
        lower_barrier: Some(2.0),
        max_holding_period: Some(5),
    };
    let touch = find_first_touch(&prices, 100, &config, 0.01);
    assert!(touch.is_none());
}

#[test]
fn find_first_touch_flat_prices_vertical_barrier() {
    let prices = vec![100.0; 20];
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),
        lower_barrier: Some(2.0),
        max_holding_period: Some(10),
    };
    let touch = find_first_touch(&prices, 0, &config, 0.02).unwrap();
    assert_eq!(touch.touch_type, BarrierTouchType::Vertical);
    assert_eq!(touch.timestamp_index, 10);
    assert!((touch.return_value - 0.0).abs() < 1e-10);
}

#[test]
fn drop_rare_labels_empty() {
    let mask = drop_rare_labels(&[], 0.1);
    assert!(mask.is_empty());
}

#[test]
fn drop_rare_labels_all_same() {
    let labels = vec![1; 10];
    let mask = drop_rare_labels(&labels, 0.5);
    assert!(mask.iter().all(|&m| m));
}

#[test]
fn drop_rare_labels_min_pct_zero_keeps_all() {
    let labels = vec![1, -1, 0, 1, -1];
    let mask = drop_rare_labels(&labels, 0.0);
    assert!(mask.iter().all(|&m| m));
}

#[test]
fn drop_rare_labels_min_pct_one_drops_all_but_majority() {
    let labels = vec![1, 1, 1, -1, 0];
    let mask = drop_rare_labels(&labels, 1.0);
    // Only labels appearing 100% of the time survive - none do
    // 1 appears 60%, -1 appears 20%, 0 appears 20%
    assert!(mask.iter().all(|&m| !m));
}
