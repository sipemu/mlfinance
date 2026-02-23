use mlfinance_sampling::fracdiff::ffd::frac_diff_ffd;
use mlfinance_sampling::fracdiff::weights::{get_weights, get_weights_ffd};
use proptest::prelude::*;

#[test]
fn ffd_empty_series() {
    let result = frac_diff_ffd(&[], 0.5, 1e-4);
    assert!(result.is_empty());
}

#[test]
fn ffd_d_zero_is_identity() {
    // d=0: only weight is w[0]=1, rest are 0
    let w = get_weights(0.0, 10);
    assert!((w[0] - 1.0).abs() < 1e-10);
    for &wi in &w[1..] {
        assert!(wi.abs() < 1e-10, "d=0 weight should be 0, got {}", wi);
    }
}

#[test]
fn ffd_d_one_is_first_difference() {
    let series = vec![10.0, 12.0, 15.0, 13.0, 20.0];
    let result = frac_diff_ffd(&series, 1.0, 0.01);
    // d=1: weights = [1, -1], so result[t] = series[t] - series[t-1]
    assert!(result[0].is_nan());
    assert!((result[1] - 2.0).abs() < 1e-10);
    assert!((result[2] - 3.0).abs() < 1e-10);
    assert!((result[3] - (-2.0)).abs() < 1e-10);
    assert!((result[4] - 7.0).abs() < 1e-10);
}

#[test]
fn ffd_d_two_is_second_difference() {
    let series = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    let result = frac_diff_ffd(&series, 2.0, 0.01);
    // d=2: weights = [1, -2, 1]
    // result[2] = 1*6 + (-2)*3 + 1*1 = 6 - 6 + 1 = 1
    // result[3] = 1*10 + (-2)*6 + 1*3 = 10 - 12 + 3 = 1
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
    assert!((result[2] - 1.0).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - 1.0).abs() < 1e-10);
}

#[test]
fn ffd_threshold_one_only_first_weight() {
    // Threshold=1.0 means only w[0]=1 survives (since w[1]'s abs < 1 for d<1)
    let w = get_weights_ffd(0.5, 1.0);
    assert_eq!(w.len(), 1);
    assert!((w[0] - 1.0).abs() < 1e-10);
}

#[test]
fn ffd_very_small_threshold_many_weights() {
    let w = get_weights_ffd(0.5, 1e-15);
    assert!(
        w.len() > 100,
        "Very small threshold should produce many weights, got {}",
        w.len()
    );
}

#[test]
fn ffd_nan_in_series() {
    let series = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let result = frac_diff_ffd(&series, 0.5, 0.1);
    // Should not panic; NaN should propagate
    assert_eq!(result.len(), series.len());
}

#[test]
fn ffd_inf_in_series() {
    let series = vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0];
    let result = frac_diff_ffd(&series, 0.5, 0.1);
    // Should not panic
    assert_eq!(result.len(), series.len());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn ffd_never_panics(
        series in proptest::collection::vec(-1e6f64..1e6, 0..500),
        d in 0.01f64..2.0,
        threshold in 1e-8f64..1.0
    ) {
        let _ = frac_diff_ffd(&series, d, threshold);
    }
}
