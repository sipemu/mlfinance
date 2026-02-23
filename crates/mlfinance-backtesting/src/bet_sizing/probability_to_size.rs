//! Bet sizing from predicted probabilities (Snippet 10.1).
//!
//! Converts a predicted probability into a position size using various
//! mapping functions. Supports binary and multi-class classifiers.

/// Sigmoid bet sizing: converts probability to bet size.
///
/// For binary classification (`num_classes == 2`), the mapping is linear:
/// `size = 2 * prob - 1`, yielding values in `[-1, 1]`.
///
/// For multi-class classification, the signal is normalized relative to the
/// uniform probability `1/num_classes` and scaled so that a probability of 1.0
/// maps to a size of 1.0.
///
/// # Arguments
/// * `prob` - Predicted probability in `[0, 1]`.
/// * `num_classes` - Number of classes (must be >= 2).
///
/// # Returns
/// Bet size in `[-1, 1]`.
pub fn sigmoid_bet_size(prob: f64, num_classes: usize) -> f64 {
    if num_classes < 2 {
        return 0.0;
    }

    if num_classes == 2 {
        // Binary: linear mapping
        return (2.0 * prob - 1.0).clamp(-1.0, 1.0);
    }

    // Multi-class: normalize relative to uniform probability
    let uniform = 1.0 / num_classes as f64;
    let scaling = 1.0 - uniform; // max deviation from uniform

    if scaling == 0.0 {
        return 0.0;
    }

    let size = (prob - uniform) / scaling;
    size.clamp(-1.0, 1.0)
}

/// Power bet sizing.
///
/// Similar to sigmoid bet sizing but applies a power transformation to control
/// aggressiveness. Higher exponents make the sizing more conservative for
/// marginal signals and more aggressive for strong signals.
///
/// # Arguments
/// * `prob` - Predicted probability in `[0, 1]`.
/// * `num_classes` - Number of classes (must be >= 2).
/// * `exponent` - Power exponent (must be > 0). Values > 1 are conservative,
///   values < 1 are aggressive.
///
/// # Returns
/// Bet size in `[-1, 1]`.
pub fn power_bet_size(prob: f64, num_classes: usize, exponent: f64) -> f64 {
    if num_classes < 2 || exponent <= 0.0 {
        return 0.0;
    }

    let linear_size = sigmoid_bet_size(prob, num_classes);
    let sign = linear_size.signum();
    let magnitude = linear_size.abs().powf(exponent);

    sign * magnitude.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_binary_extremes() {
        let size_0 = sigmoid_bet_size(0.0, 2);
        let size_1 = sigmoid_bet_size(1.0, 2);
        assert!((size_0 - (-1.0)).abs() < 1e-10);
        assert!((size_1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_binary_midpoint() {
        let size = sigmoid_bet_size(0.5, 2);
        assert!(size.abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_multiclass() {
        // For 3 classes, uniform = 1/3
        let size = sigmoid_bet_size(1.0 / 3.0, 3);
        assert!(size.abs() < 1e-10, "uniform prob should give zero size");

        let size_max = sigmoid_bet_size(1.0, 3);
        assert!((size_max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_invalid_classes() {
        assert_eq!(sigmoid_bet_size(0.5, 0), 0.0);
        assert_eq!(sigmoid_bet_size(0.5, 1), 0.0);
    }

    #[test]
    fn test_power_bet_size_exponent_1() {
        // With exponent = 1, should equal sigmoid
        let sig = sigmoid_bet_size(0.7, 2);
        let pow = power_bet_size(0.7, 2, 1.0);
        assert!((sig - pow).abs() < 1e-10);
    }

    #[test]
    fn test_power_bet_size_conservative() {
        // Exponent > 1 should reduce magnitude for marginal signals
        let linear = sigmoid_bet_size(0.6, 2).abs();
        let power = power_bet_size(0.6, 2, 2.0).abs();
        assert!(power < linear, "conservative sizing should be smaller");
    }

    #[test]
    fn test_power_bet_size_aggressive() {
        // Exponent < 1 should increase magnitude for marginal signals
        let linear = sigmoid_bet_size(0.6, 2).abs();
        let power = power_bet_size(0.6, 2, 0.5).abs();
        assert!(power > linear, "aggressive sizing should be larger");
    }

    #[test]
    fn test_power_invalid_exponent() {
        assert_eq!(power_bet_size(0.7, 2, 0.0), 0.0);
        assert_eq!(power_bet_size(0.7, 2, -1.0), 0.0);
    }
}
