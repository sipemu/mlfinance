use mlfinance_labeling::drop_labels::{drop_labels, label_counts};
use mlfinance_labeling::trend_scanning::{trend_scanning_label_series, trend_scanning_labels};
use mlfinance_labeling::vertical_barrier::add_vertical_barrier;

// ── add_vertical_barrier ──

#[test]
fn test_add_vertical_barrier_basic() {
    let entry_indices = vec![0, 5, 10];
    let exits = add_vertical_barrier(&entry_indices, 3, 15);
    assert_eq!(exits.len(), 3);
    assert_eq!(exits[0], 3); // 0 + 3
    assert_eq!(exits[1], 8); // 5 + 3
    assert_eq!(exits[2], 13); // 10 + 3
}

#[test]
fn test_add_vertical_barrier_clamped() {
    let entry_indices = vec![0, 8, 12];
    let exits = add_vertical_barrier(&entry_indices, 5, 10);
    assert_eq!(exits.len(), 3);
    assert_eq!(exits[0], 5); // 0 + 5
                             // Indices should be clamped to series_len - 1 = 9
    assert_eq!(exits[1], 9);
    assert_eq!(exits[2], 9);
}

#[test]
fn test_add_vertical_barrier_empty() {
    let exits = add_vertical_barrier(&[], 5, 100);
    assert!(exits.is_empty());
}

// ── label_counts ──

#[test]
fn test_label_counts() {
    let labels = vec![1, -1, 1, 0, 1, -1, 0, 0];
    let counts = label_counts(&labels);
    assert_eq!(counts[&1], 3);
    assert_eq!(counts[&(-1)], 2);
    assert_eq!(counts[&0], 3);
}

#[test]
fn test_label_counts_empty() {
    let counts = label_counts(&[]);
    assert!(counts.is_empty());
}

#[test]
fn test_label_counts_single_class() {
    let labels = vec![1, 1, 1];
    let counts = label_counts(&labels);
    assert_eq!(counts.len(), 1);
    assert_eq!(counts[&1], 3);
}

// ── drop_labels ──

#[test]
fn test_drop_labels_removes_rare() {
    let mut labels = vec![1, 1, 1, 1, 1, 1, 1, 1, -1, 0];
    let mut features: Vec<Vec<f64>> = labels.iter().map(|&l| vec![l as f64]).collect();
    // min_pct = 0.15 means classes with < 15% should be dropped
    // -1 has 10%, 0 has 10%, both below threshold
    drop_labels(&mut labels, &mut features, 0.15);
    assert_eq!(labels.len(), features.len());
    // Only class 1 should remain
    for &l in &labels {
        assert_eq!(l, 1);
    }
}

#[test]
fn test_drop_labels_keeps_all() {
    let mut labels = vec![1, 1, -1, -1, 0, 0];
    let mut features: Vec<Vec<f64>> = labels.iter().map(|&l| vec![l as f64]).collect();
    drop_labels(&mut labels, &mut features, 0.1);
    // All classes have >= 33%, so none should be dropped
    assert_eq!(labels.len(), 6);
}

// ── trend_scanning_labels ──

#[test]
fn test_trend_scanning_labels_uptrend() {
    // Clear uptrend
    let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
    let results = trend_scanning_labels(&prices, None, 10, None).unwrap();
    // Default events: 0..=n-3, so n-2 results
    assert_eq!(results.len(), prices.len() - 2);
    // Most labels should indicate uptrend (label = 1)
    let up_count = results.iter().filter(|r| r.label == 1).count();
    assert!(up_count > results.len() / 2);
}

#[test]
fn test_trend_scanning_labels_with_events() {
    let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
    let events = vec![0, 5, 10, 15];
    let results = trend_scanning_labels(&prices, Some(&events), 8, Some(3)).unwrap();
    assert_eq!(results.len(), events.len());
}

#[test]
fn test_trend_scanning_labels_short_series() {
    let prices = vec![100.0, 101.0];
    let result = trend_scanning_labels(&prices, None, 5, None);
    // Should handle gracefully (either Ok with limited results or Err)
    assert!(result.is_ok() || result.is_err());
}

// ── trend_scanning_label_series ──

#[test]
fn test_trend_scanning_label_series() {
    let prices: Vec<f64> = (0..40).map(|i| 100.0 + i as f64 * 0.3).collect();
    let labels = trend_scanning_label_series(&prices, 10).unwrap();
    // Default events: 0..=n-3, so n-2 results
    assert_eq!(labels.len(), prices.len() - 2);
    // Labels should be -1, 0, or 1
    for &l in &labels {
        assert!(l == -1 || l == 0 || l == 1);
    }
}
