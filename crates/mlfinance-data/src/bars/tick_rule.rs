/// Classify a tick using the tick rule.
///
/// Returns +1.0 if `price > prev_price`, -1.0 if `price < prev_price`,
/// otherwise returns `prev_bt` (the previous tick classification).
///
/// This is the standard tick rule used in market microstructure to classify
/// trades as buyer- or seller-initiated.
///
/// # Arguments
///
/// * `price` - Current trade price.
/// * `prev_price` - Previous trade price.
/// * `prev_bt` - Previous tick classification (+1.0 or -1.0), used when
///   the price is unchanged.
///
/// # Returns
///
/// `+1.0` for a buy-initiated tick, `-1.0` for a sell-initiated tick.
pub fn classify_tick(price: f64, prev_price: f64, prev_bt: f64) -> f64 {
    if price > prev_price {
        1.0
    } else if price < prev_price {
        -1.0
    } else {
        prev_bt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uptick() {
        assert_eq!(classify_tick(101.0, 100.0, -1.0), 1.0);
    }

    #[test]
    fn test_downtick() {
        assert_eq!(classify_tick(99.0, 100.0, 1.0), -1.0);
    }

    #[test]
    fn test_no_change_keeps_previous() {
        assert_eq!(classify_tick(100.0, 100.0, 1.0), 1.0);
        assert_eq!(classify_tick(100.0, 100.0, -1.0), -1.0);
    }

    #[test]
    fn test_sequence() {
        let prices = [100.0, 100.5, 100.5, 99.0, 99.0, 100.0];
        let mut bt = 1.0;
        let mut prev = prices[0];
        let mut results = vec![];
        for &p in &prices[1..] {
            bt = classify_tick(p, prev, bt);
            results.push(bt);
            prev = p;
        }
        assert_eq!(results, vec![1.0, 1.0, -1.0, -1.0, 1.0]);
    }
}
