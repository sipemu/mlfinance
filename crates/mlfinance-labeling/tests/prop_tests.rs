use mlfinance_labeling::barriers::{BarrierTouchType, TripleBarrierConfig};
use mlfinance_labeling::events::{get_events, Event};
use mlfinance_labeling::labels::{drop_rare_labels, get_bins, get_meta_bins};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn labels_in_valid_range(n in 1usize..50) {
        let events: Vec<Event> = (0..n).map(|i| Event {
            entry_idx: i,
            exit_idx: i + 1,
            touch_type: if i % 3 == 0 { BarrierTouchType::Upper }
                       else if i % 3 == 1 { BarrierTouchType::Lower }
                       else { BarrierTouchType::Vertical },
            return_value: (i as f64 - n as f64 / 2.0) * 0.01,
        }).collect();

        let bins = get_bins(&events);
        for &b in &bins {
            prop_assert!(b == -1 || b == 0 || b == 1, "Invalid label: {}", b);
        }
    }

    #[test]
    fn meta_labels_in_valid_range(n in 1usize..50) {
        let events: Vec<Event> = (0..n).map(|i| Event {
            entry_idx: i,
            exit_idx: i + 1,
            touch_type: BarrierTouchType::Upper,
            return_value: (i as f64 - n as f64 / 2.0) * 0.01,
        }).collect();

        let predictions = vec![1i32; n];
        let meta = get_meta_bins(&events, &predictions);
        for &m in &meta {
            prop_assert!(m == 0 || m == 1, "Invalid meta-label: {}", m);
        }
    }

    #[test]
    fn drop_rare_labels_mask_length(
        labels in proptest::collection::vec(-1i32..2, 1..100),
        min_pct in 0.0f64..1.0
    ) {
        let mask = drop_rare_labels(&labels, min_pct);
        prop_assert_eq!(mask.len(), labels.len());
    }

    #[test]
    fn vertical_barrier_exit_equals_entry_plus_holding(
        n in 20usize..100,
        holding_period in 5usize..15
    ) {
        // Flat prices: neither upper nor lower barrier should trigger
        let prices = vec![100.0; n];
        let entry_idx = 0;
        let daily_vols = vec![0.02; n];

        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0),
            max_holding_period: Some(holding_period),
        };

        let events = get_events(
            &prices,
            &[entry_idx],
            &config,
            &daily_vols,
        );

        if !events.is_empty() {
            let event = &events[0];
            prop_assert_eq!(
                event.exit_idx,
                entry_idx + holding_period,
                "For flat prices, exit should be at vertical barrier"
            );
            prop_assert_eq!(event.touch_type, BarrierTouchType::Vertical);
        }
    }
}
