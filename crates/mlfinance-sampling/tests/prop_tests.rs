use mlfinance_sampling::bootstrap::sequential::seq_bootstrap;
use mlfinance_sampling::concurrency::average_uniqueness::average_uniqueness;
use mlfinance_sampling::concurrency::indicator_matrix::get_indicator_matrix;
use mlfinance_sampling::fracdiff::ffd::frac_diff_ffd;
use mlfinance_sampling::fracdiff::weights::get_weights_ffd;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn ffd_first_weight_is_one(d in 0.01f64..2.0, threshold in 1e-6f64..0.5) {
        let w = get_weights_ffd(d, threshold);
        prop_assert!(!w.is_empty());
        prop_assert!((w[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ffd_weights_decrease_for_d_in_01(d in 0.01f64..0.99, threshold in 1e-6f64..0.1) {
        let w = get_weights_ffd(d, threshold);
        for i in 1..w.len() {
            prop_assert!(
                w[i].abs() <= w[i - 1].abs() + 1e-10,
                "|w[{}]| = {} > |w[{}]| = {} for d={}",
                i,
                w[i].abs(),
                i - 1,
                w[i - 1].abs(),
                d
            );
        }
    }

    #[test]
    fn ffd_output_length_matches_input(
        n in 5usize..100,
        d in 0.1f64..1.5,
        threshold in 1e-4f64..0.5
    ) {
        let series: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 10.0 + 100.0).collect();
        let result = frac_diff_ffd(&series, d, threshold);
        prop_assert_eq!(result.len(), series.len());
    }

    #[test]
    fn seq_bootstrap_output_length(
        n_events in 2usize..10,
        n_bars in 10usize..30,
        n_samples in 1usize..20,
        seed in 0u64..10000
    ) {
        let events: Vec<(usize, usize)> = (0..n_events)
            .map(|i| {
                let start = i * n_bars / n_events;
                let end = (start + n_bars / n_events).min(n_bars - 1);
                (start, end)
            })
            .collect();
        let matrix = get_indicator_matrix(&events, n_bars);
        let result = seq_bootstrap(&matrix, n_samples, seed);
        prop_assert_eq!(result.len(), n_samples);
    }

    #[test]
    fn seq_bootstrap_indices_in_range(
        n_events in 2usize..10,
        n_bars in 10usize..30,
        n_samples in 1usize..20,
        seed in 0u64..10000
    ) {
        let events: Vec<(usize, usize)> = (0..n_events)
            .map(|i| {
                let start = i * n_bars / n_events;
                let end = (start + n_bars / n_events).min(n_bars - 1);
                (start, end)
            })
            .collect();
        let matrix = get_indicator_matrix(&events, n_bars);
        let result = seq_bootstrap(&matrix, n_samples, seed);
        for &idx in &result {
            prop_assert!(idx < n_events, "Index {} >= n_events {}", idx, n_events);
        }
    }

    #[test]
    fn seq_bootstrap_deterministic(
        n_events in 2usize..10,
        n_bars in 10usize..30,
        n_samples in 1usize..20,
        seed in 0u64..10000
    ) {
        let events: Vec<(usize, usize)> = (0..n_events)
            .map(|i| {
                let start = i * n_bars / n_events;
                let end = (start + n_bars / n_events).min(n_bars - 1);
                (start, end)
            })
            .collect();
        let matrix = get_indicator_matrix(&events, n_bars);
        let r1 = seq_bootstrap(&matrix, n_samples, seed);
        let r2 = seq_bootstrap(&matrix, n_samples, seed);
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn average_uniqueness_in_range(
        n_events in 1usize..10,
        n_bars in 5usize..30
    ) {
        let events: Vec<(usize, usize)> = (0..n_events)
            .map(|i| {
                let start = i * n_bars / n_events;
                let end = (start + n_bars / n_events).min(n_bars - 1);
                (start, end)
            })
            .collect();
        let uniq = average_uniqueness(&events, n_bars);
        for &u in &uniq {
            prop_assert!(u >= 0.0 && u <= 1.0 + 1e-10, "Uniqueness {} out of range", u);
        }
    }
}
