//! Signal discretization (Snippet 10.3).
//!
//! Converts continuous bet signals into discrete steps, which is useful for
//! order management systems that require integer lot sizes.

/// Discretize a continuous signal into integer steps.
///
/// Rounds the signal to the nearest multiple of `step_size`. For example,
/// with a `step_size` of 0.1, a signal of 0.34 becomes 0.3 and 0.35 becomes
/// 0.4 (banker's rounding is not used; standard rounding applies).
///
/// # Arguments
/// * `signal` - Continuous signal value.
/// * `step_size` - Size of each discrete step. Must be positive.
///
/// # Returns
/// The discretized signal value. Returns 0.0 if `step_size` is not positive.
pub fn discrete_signal(signal: f64, step_size: f64) -> f64 {
    if step_size <= 0.0 || !step_size.is_finite() {
        return 0.0;
    }
    if !signal.is_finite() {
        return 0.0;
    }

    (signal / step_size).round() * step_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_discretization() {
        let result = discrete_signal(0.34, 0.1);
        assert!((result - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_round_up() {
        let result = discrete_signal(0.36, 0.1);
        assert!((result - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_exact_step() {
        let result = discrete_signal(0.5, 0.1);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_negative_signal() {
        let result = discrete_signal(-0.34, 0.1);
        assert!((result - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_zero_signal() {
        let result = discrete_signal(0.0, 0.1);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_step_size() {
        assert_eq!(discrete_signal(0.5, 0.0), 0.0);
        assert_eq!(discrete_signal(0.5, -0.1), 0.0);
    }

    #[test]
    fn test_large_step() {
        let result = discrete_signal(0.4, 1.0);
        assert!((result - 0.0).abs() < 1e-10);

        let result = discrete_signal(0.6, 1.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_finite() {
        assert_eq!(discrete_signal(f64::NAN, 0.1), 0.0);
        assert_eq!(discrete_signal(f64::INFINITY, 0.1), 0.0);
        assert_eq!(discrete_signal(0.5, f64::NAN), 0.0);
    }
}
