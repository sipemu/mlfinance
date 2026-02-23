use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::TickData;
use mlfinance_data::bars::tick_bars::TickBarAggregator;
use mlfinance_data::bars::volume_bars::VolumeBarAggregator;
use mlfinance_data::sampling::cusum_filter::cusum_filter;
use proptest::prelude::*;

fn make_tick(price: f64, volume: f64) -> TickData {
    TickData {
        timestamp: chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
        price,
        volume,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn cusum_events_strictly_increasing(
        values in proptest::collection::vec(-1.0f64..1.0, 2..500),
        threshold in 0.01f64..10.0
    ) {
        let events = cusum_filter(&values, threshold);
        for window in events.windows(2) {
            prop_assert!(window[0] < window[1]);
        }
    }

    #[test]
    fn cusum_events_within_bounds(
        values in proptest::collection::vec(-1.0f64..1.0, 2..500),
        threshold in 0.01f64..10.0
    ) {
        let events = cusum_filter(&values, threshold);
        for &idx in &events {
            prop_assert!(idx > 0 && idx < values.len());
        }
    }

    #[test]
    fn cusum_constant_series_no_events(c in -100.0f64..100.0, n in 2usize..200) {
        let values = vec![c; n];
        let events = cusum_filter(&values, 0.01);
        prop_assert!(events.is_empty(), "Constant series should produce no CUSUM events");
    }

    #[test]
    fn tick_bars_correct_count(n_ticks in 1usize..200, bar_size in 1usize..50) {
        let ticks: Vec<TickData> = (0..n_ticks)
            .map(|i| make_tick(100.0 + i as f64 * 0.1, 10.0))
            .collect();

        let mut agg = TickBarAggregator::new(bar_size);
        let bars = agg.process_ticks(&ticks);

        prop_assert_eq!(bars.len(), n_ticks / bar_size);
    }

    #[test]
    fn volume_bars_volume_exceeds_threshold(
        n_ticks in 10usize..200,
        volume_per_tick in 1.0f64..100.0,
        threshold_mult in 2.0f64..20.0
    ) {
        let threshold = volume_per_tick * threshold_mult;
        let ticks: Vec<TickData> = (0..n_ticks)
            .map(|i| make_tick(100.0 + (i as f64 * 0.01).sin(), volume_per_tick))
            .collect();

        let mut agg = VolumeBarAggregator::new(threshold);
        let bars = agg.process_ticks(&ticks);

        for bar in &bars {
            prop_assert!(
                bar.volume >= threshold - 1e-6,
                "Bar volume {} < threshold {}",
                bar.volume,
                threshold
            );
        }
    }
}
