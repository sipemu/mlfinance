use mlfinance_data::bars::tick_rule::classify_tick;
use mlfinance_data::multi_product::etf_trick::etf_trick;
use mlfinance_data::multi_product::pca_weights::pca_weights;
use mlfinance_data::multi_product::single_future_roll::{non_negative_rolled, roll_gaps};
use mlfinance_data::sampling::event_sampling::{linspace_sample, uniform_sample};
use ndarray::array;

// ── classify_tick ──

#[test]
fn test_classify_tick_uptick() {
    assert_eq!(classify_tick(101.0, 100.0, 0.0), 1.0);
}

#[test]
fn test_classify_tick_downtick() {
    assert_eq!(classify_tick(99.0, 100.0, 0.0), -1.0);
}

#[test]
fn test_classify_tick_no_change_uses_prev() {
    assert_eq!(classify_tick(100.0, 100.0, 1.0), 1.0);
    assert_eq!(classify_tick(100.0, 100.0, -1.0), -1.0);
}

// ── pca_weights ──

#[test]
fn test_pca_weights_basic() {
    // Diagonal covariance => equal risk parity should weight inversely by variance
    let cov = array![[0.04, 0.0], [0.0, 0.16]];
    let weights = pca_weights(&cov, None).unwrap();
    assert_eq!(weights.len(), 2);
    // Weights should sum to 1 (or close)
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.1 || (sum + 1.0).abs() < 0.1);
}

#[test]
fn test_pca_weights_with_risk_target() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let weights = pca_weights(&cov, Some(0.5)).unwrap();
    assert_eq!(weights.len(), 2);
    for &w in weights.iter() {
        assert!(w.is_finite());
    }
}

// ── roll_gaps ──

#[test]
fn test_roll_gaps_no_rolls() {
    let prices = vec![100.0, 101.0, 102.0, 103.0];
    let roll_dates: Vec<usize> = vec![];
    let gaps = roll_gaps(&prices, &roll_dates);
    assert_eq!(gaps.len(), prices.len());
    // No rolls => all gaps are 0
    for &g in &gaps {
        assert_eq!(g, 0.0);
    }
}

#[test]
fn test_roll_gaps_with_roll() {
    let prices = vec![100.0, 101.0, 105.0, 106.0];
    let roll_dates = vec![2]; // Roll at index 2
    let gaps = roll_gaps(&prices, &roll_dates);
    assert_eq!(gaps.len(), prices.len());
}

// ── non_negative_rolled ──

#[test]
fn test_non_negative_rolled() {
    let prices = vec![100.0, 101.0, 105.0, 106.0, 107.0];
    let roll_dates = vec![2];
    let rolled = non_negative_rolled(&prices, &roll_dates);
    assert_eq!(rolled.len(), prices.len());
    // All values should be non-negative
    for &v in &rolled {
        assert!(v >= 0.0, "non_negative_rolled should be >= 0, got {v}");
    }
}

#[test]
fn test_non_negative_rolled_no_rolls() {
    let prices = vec![50.0, 51.0, 52.0];
    let rolled = non_negative_rolled(&prices, &[]);
    assert_eq!(rolled.len(), 3);
    for &v in &rolled {
        assert!(v >= 0.0);
    }
}

// ── etf_trick ──

#[test]
fn test_etf_trick_single_product() {
    let prices = vec![vec![100.0, 101.0, 102.0]];
    let weights = vec![vec![1.0, 1.0, 1.0]];
    let etf = etf_trick(&prices, &weights);
    assert_eq!(etf.len(), 3);
    for &v in &etf {
        assert!(v.is_finite());
    }
}

#[test]
fn test_etf_trick_two_products() {
    let prices = vec![vec![100.0, 101.0, 102.0], vec![200.0, 202.0, 204.0]];
    let weights = vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]];
    let etf = etf_trick(&prices, &weights);
    assert_eq!(etf.len(), 3);
    for &v in &etf {
        assert!(v.is_finite());
    }
}

// ── linspace_sample ──

#[test]
fn test_linspace_sample() {
    let samples = linspace_sample(0, 100, 5);
    assert_eq!(samples.len(), 5);
    // Should be sorted
    for i in 1..samples.len() {
        assert!(samples[i] >= samples[i - 1]);
    }
    // samples[0] is usize, always >= 0
    assert!(*samples.last().unwrap() <= 100);
}

#[test]
fn test_linspace_sample_two_points() {
    let samples = linspace_sample(0, 10, 2);
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0], 0);
    assert_eq!(samples[1], 10);
}

// ── uniform_sample ──

#[test]
fn test_uniform_sample() {
    let samples = uniform_sample(5, 100, 42);
    assert!(samples.len() <= 5);
    // Should be sorted and deduplicated
    for i in 1..samples.len() {
        assert!(samples[i] > samples[i - 1]);
    }
    for &s in &samples {
        assert!(s < 100);
    }
}

#[test]
fn test_uniform_sample_deterministic() {
    let s1 = uniform_sample(10, 1000, 42);
    let s2 = uniform_sample(10, 1000, 42);
    assert_eq!(s1, s2);
}

#[test]
fn test_uniform_sample_different_seeds() {
    let s1 = uniform_sample(10, 1000, 42);
    let s2 = uniform_sample(10, 1000, 99);
    assert_ne!(s1, s2);
}
