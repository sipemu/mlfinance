use mlfinance_backtesting::bet_sizing::probability_to_size::{power_bet_size, sigmoid_bet_size};

#[test]
fn sigmoid_nan_prob() {
    let result = sigmoid_bet_size(f64::NAN, 2);
    assert!(result.is_nan() || result.abs() <= 1.0);
}

#[test]
fn sigmoid_zero_classes() {
    assert_eq!(sigmoid_bet_size(0.5, 0), 0.0);
}

#[test]
fn sigmoid_one_class() {
    assert_eq!(sigmoid_bet_size(0.5, 1), 0.0);
}

#[test]
fn sigmoid_prob_out_of_range_clamped() {
    let s1 = sigmoid_bet_size(-0.5, 2);
    assert!((-1.0..=1.0).contains(&s1), "Clamped for prob<0: {}", s1);

    let s2 = sigmoid_bet_size(1.5, 2);
    assert!((-1.0..=1.0).contains(&s2), "Clamped for prob>1: {}", s2);
}

#[test]
fn power_zero_exponent() {
    let result = power_bet_size(0.7, 2, 0.0);
    assert_eq!(result, 0.0, "Zero exponent should return 0");
}

#[test]
fn power_negative_exponent() {
    let result = power_bet_size(0.7, 2, -1.0);
    assert_eq!(result, 0.0, "Negative exponent should return 0");
}

#[test]
fn power_nan_prob() {
    let result = power_bet_size(f64::NAN, 2, 2.0);
    // Should not panic; NaN propagates
    let _ = result;
}

#[test]
fn sigmoid_extremes() {
    // Probability 0 and 1 should give -1 and +1 for binary
    let at_0 = sigmoid_bet_size(0.0, 2);
    let at_1 = sigmoid_bet_size(1.0, 2);
    assert!((at_0 - (-1.0)).abs() < 1e-10);
    assert!((at_1 - 1.0).abs() < 1e-10);
}

#[test]
fn power_large_exponent() {
    // Very large exponent should make moderate probabilities near zero
    let result = power_bet_size(0.6, 2, 100.0);
    assert!(
        result.abs() < 0.01,
        "Large exponent should make marginal probs near 0, got {}",
        result
    );
}
