use mlfinance_sampling::bootstrap::monte_carlo::compare_bootstraps;
use mlfinance_sampling::bootstrap::standard::standard_bootstrap;
use mlfinance_sampling::concurrency::num_co_events::num_co_events;
use mlfinance_sampling::fracdiff::expanding::frac_diff_expanding;
use mlfinance_sampling::fracdiff::min_d::find_min_d;
use mlfinance_sampling::weights::return_attribution::return_attribution_weights;
use ndarray::Array2;

// ── num_co_events ──

#[test]
fn test_num_co_events_basic() {
    // Events: (start, end) pairs
    let events = vec![(0, 3), (1, 4), (2, 5)];
    let co = num_co_events(&events, 6);
    assert_eq!(co.len(), 6);
    // bar 0: only event 0 active => 1
    assert_eq!(co[0], 1);
    // bar 2: all three events active => 3
    assert_eq!(co[2], 3);
    // bar 5: only event 2 active => 1
    assert_eq!(co[5], 1);
}

#[test]
fn test_num_co_events_no_overlap() {
    let events = vec![(0, 0), (2, 2), (4, 4)];
    let co = num_co_events(&events, 5);
    assert_eq!(co[0], 1);
    assert_eq!(co[1], 0);
    assert_eq!(co[2], 1);
    assert_eq!(co[3], 0);
    assert_eq!(co[4], 1);
}

#[test]
fn test_num_co_events_empty() {
    let co = num_co_events(&[], 5);
    assert_eq!(co, vec![0; 5]);
}

// ── standard_bootstrap ──

#[test]
fn test_standard_bootstrap_size() {
    let samples = standard_bootstrap(100, 50, 42);
    assert_eq!(samples.len(), 50);
    for &s in &samples {
        assert!(s < 100);
    }
}

#[test]
fn test_standard_bootstrap_deterministic() {
    let s1 = standard_bootstrap(100, 20, 42);
    let s2 = standard_bootstrap(100, 20, 42);
    assert_eq!(s1, s2);
}

// ── compare_bootstraps ──

#[test]
fn test_compare_bootstraps() {
    // Simple indicator matrix: 3 events, 5 bars
    let ind_matrix = Array2::from_shape_vec(
        (3, 5),
        vec![
            1.0, 1.0, 0.0, 0.0, 0.0, // event 0 covers bars 0-1
            0.0, 1.0, 1.0, 0.0, 0.0, // event 1 covers bars 1-2
            0.0, 0.0, 0.0, 1.0, 1.0, // event 2 covers bars 3-4
        ],
    )
    .unwrap();
    let comparison = compare_bootstraps(&ind_matrix, 3, 10, 42);
    // Sequential bootstrap should have higher uniqueness than standard
    assert!(comparison.seq_uniqueness > 0.0);
    assert!(comparison.std_uniqueness > 0.0);
    assert!(comparison.seq_uniqueness >= comparison.std_uniqueness - 0.1);
}

// ── return_attribution_weights ──

#[test]
fn test_return_attribution_weights() {
    let events = vec![(0, 2), (1, 3), (3, 4)];
    // One return per event
    let returns = vec![0.01, -0.02, 0.03];
    let weights = return_attribution_weights(&events, &returns, 5);
    assert_eq!(weights.len(), events.len());
    for &w in &weights {
        assert!(w.is_finite());
    }
    // Weights should sum to num_events = 3
    let sum: f64 = weights.iter().sum();
    assert!((sum - 3.0).abs() < 1e-10);
}

#[test]
fn test_return_attribution_weights_single_event() {
    let events = vec![(0, 4)];
    let returns = vec![0.05];
    let weights = return_attribution_weights(&events, &returns, 5);
    assert_eq!(weights.len(), 1);
    assert!((weights[0] - 1.0).abs() < 1e-10);
}

// ── frac_diff_expanding ──

#[test]
fn test_frac_diff_expanding_d_zero() {
    let series = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let result = frac_diff_expanding(&series, 0.0, 1e-4);
    assert_eq!(result.len(), series.len());
    // d=0 should leave the series unchanged
    for (a, b) in result.iter().zip(series.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_frac_diff_expanding_d_one() {
    let series = vec![100.0, 101.0, 103.0, 106.0, 110.0];
    let result = frac_diff_expanding(&series, 1.0, 1e-4);
    // d=1 should approximate first differences
    assert_eq!(result.len(), series.len());
}

#[test]
fn test_frac_diff_expanding_fractional() {
    let series: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sqrt()).collect();
    let result = frac_diff_expanding(&series, 0.5, 1e-4);
    assert_eq!(result.len(), series.len());
    for &v in &result {
        assert!(v.is_finite());
    }
}

// ── find_min_d ──

#[test]
fn test_find_min_d() {
    // Random walk-like series (needs d close to 1 for stationarity)
    let series: Vec<f64> = {
        let mut s = vec![100.0];
        for i in 1..100 {
            s.push(s[i - 1] + ((i * 7 + 3) % 11) as f64 - 5.0);
        }
        s
    };
    let min_d = find_min_d(&series, 1.0, 0.1, 1e-4);
    assert!(min_d >= 0.0);
    assert!(min_d <= 1.0);
}

#[test]
fn test_find_min_d_stationary_series() {
    // Already stationary series: differences around 0
    let series: Vec<f64> = (0..100)
        .map(|i| ((i * 17 + 5) % 23) as f64 - 11.0)
        .collect();
    let min_d = find_min_d(&series, 1.0, 0.1, 1e-4);
    // Should find a small d since series is already stationary-like
    assert!(min_d >= 0.0);
}
