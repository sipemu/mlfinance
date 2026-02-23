//! Dynamic bet sizing with limit prices (Snippet 10.4).
//!
//! Computes limit prices and bet sizes that account for the current position
//! and maximum allowed position, using a sigmoid-based concavity function.

/// Configuration for dynamic bet sizing.
#[derive(Debug, Clone)]
pub struct DynamicBetSize {
    /// Target position from the model signal.
    pub target_position: f64,
    /// Current position held.
    pub current_position: f64,
    /// Maximum allowed position (absolute value).
    pub max_position: f64,
    /// Volatility parameter (sigma) for the sigmoid.
    pub sigma: f64,
}

/// Inverse of the sigmoid sizing function.
///
/// Given a position `m` and max position `w`, returns the forecast value
/// that would produce that position.
fn inv_price(m: f64, w: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let clamped = (m / w).clamp(-1.0 + 1e-10, 1.0 - 1e-10);
    clamped.atanh()
}

/// Compute limit price and bet size for dynamic position management.
///
/// Uses a sigmoid mapping from forecast to position to determine the
/// appropriate limit price and trade size given current vs. target positions.
///
/// # Arguments
/// * `config` - Dynamic bet sizing configuration.
/// * `forecast` - Current model forecast / signal.
///
/// # Returns
/// A tuple `(limit_price_delta, bet_size)` where:
/// - `limit_price_delta`: The price delta from mid at which to place the limit
///   order (relative to sigma).
/// - `bet_size`: Number of units to trade (positive = buy, negative = sell).
pub fn dynamic_bet_size(config: &DynamicBetSize, forecast: f64) -> (f64, f64) {
    if config.max_position <= 0.0 || config.sigma <= 0.0 {
        return (0.0, 0.0);
    }

    let w = config.max_position;

    // Target position from the forecast via sigmoid
    let target = w * forecast.tanh();

    // Required trade
    let trade_size = target - config.current_position;

    if trade_size.abs() < 1e-12 {
        return (0.0, 0.0);
    }

    // Limit price: inverse price of the new position
    let new_position = config.current_position + trade_size;
    let limit_delta = inv_price(new_position, w) * config.sigma;

    (limit_delta, trade_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_forecast() {
        let config = DynamicBetSize {
            target_position: 0.0,
            current_position: 0.0,
            max_position: 100.0,
            sigma: 1.0,
        };
        let (limit, size) = dynamic_bet_size(&config, 0.0);
        assert!(limit.abs() < 1e-10);
        assert!(size.abs() < 1e-10);
    }

    #[test]
    fn test_positive_forecast() {
        let config = DynamicBetSize {
            target_position: 0.0,
            current_position: 0.0,
            max_position: 100.0,
            sigma: 1.0,
        };
        let (limit, size) = dynamic_bet_size(&config, 1.0);
        assert!(limit > 0.0, "positive forecast should yield positive limit");
        assert!(size > 0.0, "positive forecast should yield buy");
    }

    #[test]
    fn test_negative_forecast() {
        let config = DynamicBetSize {
            target_position: 0.0,
            current_position: 0.0,
            max_position: 100.0,
            sigma: 1.0,
        };
        let (_limit, size) = dynamic_bet_size(&config, -1.0);
        assert!(size < 0.0, "negative forecast should yield sell");
    }

    #[test]
    fn test_already_at_target() {
        let config = DynamicBetSize {
            target_position: 0.0,
            current_position: 76.16, // tanh(1.0)*100 ~ 76.16
            max_position: 100.0,
            sigma: 1.0,
        };
        let (_, size) = dynamic_bet_size(&config, 1.0);
        // Should need very little trade
        assert!(size.abs() < 1.0);
    }

    #[test]
    fn test_invalid_max_position() {
        let config = DynamicBetSize {
            target_position: 0.0,
            current_position: 0.0,
            max_position: 0.0,
            sigma: 1.0,
        };
        let (limit, size) = dynamic_bet_size(&config, 1.0);
        assert_eq!(limit, 0.0);
        assert_eq!(size, 0.0);
    }

    #[test]
    fn test_invalid_sigma() {
        let config = DynamicBetSize {
            target_position: 0.0,
            current_position: 0.0,
            max_position: 100.0,
            sigma: 0.0,
        };
        let (limit, size) = dynamic_bet_size(&config, 1.0);
        assert_eq!(limit, 0.0);
        assert_eq!(size, 0.0);
    }
}
