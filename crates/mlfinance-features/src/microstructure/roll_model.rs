/// Roll effective spread estimate.
///
/// The Roll model estimates the bid-ask spread from the first-order
/// autocovariance of price changes.
/// spread = 2 * sqrt(-cov) if cov < 0, else 0
///
/// # Arguments
///
/// * `prices` - Price time series (at least 3 prices required).
///
/// # Returns
///
/// Estimated effective spread. Returns 0.0 if the series is too short or
/// the autocovariance is non-negative (trending prices).
pub fn roll_spread(prices: &[f64]) -> f64 {
    if prices.len() < 3 {
        return 0.0;
    }

    // Compute price changes
    let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    // Compute first-order autocovariance
    let n = changes.len();
    if n < 2 {
        return 0.0;
    }

    let mean = changes.iter().sum::<f64>() / n as f64;

    let mut autocov = 0.0;
    for i in 1..n {
        autocov += (changes[i] - mean) * (changes[i - 1] - mean);
    }
    autocov /= (n - 1) as f64;

    if autocov < 0.0 {
        2.0 * (-autocov).sqrt()
    } else {
        0.0
    }
}

/// Roll model with rolling window.
///
/// Computes the Roll spread estimate over a rolling window.
///
/// # Arguments
///
/// * `prices` - Price time series.
/// * `window` - Rolling window size (must be >= 3).
///
/// # Returns
///
/// Vector of Roll spread estimates for each window position.
/// Returns an empty vector if the data is shorter than the window or window < 3.
pub fn roll_spread_rolling(prices: &[f64], window: usize) -> Vec<f64> {
    if prices.len() < window || window < 3 {
        return vec![];
    }

    prices.windows(window).map(roll_spread).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_spread_positive() {
        // Prices that bounce between bid and ask
        let prices = vec![100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 100.5];
        let spread = roll_spread(&prices);
        assert!(spread > 0.0);
    }

    #[test]
    fn test_roll_spread_trending() {
        // Pure trend: no spread signal
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let spread = roll_spread(&prices);
        // Trending prices have positive autocovariance, so spread = 0
        assert_eq!(spread, 0.0);
    }

    #[test]
    fn test_roll_spread_short() {
        let spread = roll_spread(&[100.0, 101.0]);
        assert_eq!(spread, 0.0);
    }

    #[test]
    fn test_roll_spread_rolling() {
        let prices = vec![
            100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 100.5,
        ];
        let spreads = roll_spread_rolling(&prices, 5);
        assert_eq!(spreads.len(), prices.len() - 5 + 1);
        for &s in &spreads {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_roll_spread_rolling_short() {
        let spreads = roll_spread_rolling(&[100.0, 101.0], 5);
        assert!(spreads.is_empty());
    }
}
