use mlfinance_backtesting::bet_sizing::probability_to_size::{power_bet_size, sigmoid_bet_size};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn sigmoid_bet_size_in_range(prob in 0.0f64..1.0, num_classes in 2usize..10) {
        let size = sigmoid_bet_size(prob, num_classes);
        prop_assert!(
            (-1.0 - 1e-10..=1.0 + 1e-10).contains(&size),
            "Sigmoid size {} out of [-1, 1]",
            size
        );
    }

    #[test]
    fn power_bet_size_in_range(prob in 0.0f64..1.0, num_classes in 2usize..10, exp in 0.1f64..5.0) {
        let size = power_bet_size(prob, num_classes, exp);
        prop_assert!(
            (-1.0 - 1e-10..=1.0 + 1e-10).contains(&size),
            "Power size {} out of [-1, 1]",
            size
        );
    }

    #[test]
    fn sigmoid_binary_at_half_is_zero(prob_offset in -0.001f64..0.001) {
        let size = sigmoid_bet_size(0.5 + prob_offset, 2);
        // At exactly 0.5, should be 0.0; near 0.5, should be near 0
        prop_assert!(
            size.abs() < 0.01,
            "Sigmoid at p={} should be near 0, got {}",
            0.5 + prob_offset,
            size
        );
    }

    #[test]
    fn power_at_exp1_equals_sigmoid(prob in 0.0f64..1.0, num_classes in 2usize..10) {
        let sig = sigmoid_bet_size(prob, num_classes);
        let pow = power_bet_size(prob, num_classes, 1.0);
        prop_assert!(
            (sig - pow).abs() < 1e-10,
            "Power(exp=1) {} != sigmoid {} at prob={}",
            pow,
            sig,
            prob
        );
    }
}
