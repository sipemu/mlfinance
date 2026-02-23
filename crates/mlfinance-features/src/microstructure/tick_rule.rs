/// Classify trades using the tick rule.
///
/// Returns +1 for uptick (price increased), -1 for downtick (price decreased),
/// and carries forward the previous classification on zero-tick (no change).
/// The first trade always receives 0.0 (no prior reference).
///
/// # Arguments
///
/// * `prices` - Trade price time series.
///
/// # Returns
///
/// Vector of trade signs (+1.0, -1.0, or 0.0) with the same length as `prices`.
pub fn tick_rule_classify(prices: &[f64]) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }
    if prices.len() == 1 {
        return vec![0.0];
    }

    let mut result = Vec::with_capacity(prices.len());
    result.push(0.0); // First trade has no prior reference

    let mut last_sign = 0.0;

    for i in 1..prices.len() {
        let diff = prices[i] - prices[i - 1];
        if diff > 0.0 {
            last_sign = 1.0;
        } else if diff < 0.0 {
            last_sign = -1.0;
        }
        // If diff == 0, carry forward last_sign
        result.push(last_sign);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_rule_uptick() {
        let prices = vec![100.0, 101.0, 102.0];
        let signs = tick_rule_classify(&prices);
        assert_eq!(signs, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tick_rule_downtick() {
        let prices = vec![102.0, 101.0, 100.0];
        let signs = tick_rule_classify(&prices);
        assert_eq!(signs, vec![0.0, -1.0, -1.0]);
    }

    #[test]
    fn test_tick_rule_zero_tick() {
        let prices = vec![100.0, 101.0, 101.0, 100.0];
        let signs = tick_rule_classify(&prices);
        assert_eq!(signs, vec![0.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_tick_rule_mixed() {
        let prices = vec![100.0, 101.0, 99.0, 99.0, 100.0];
        let signs = tick_rule_classify(&prices);
        assert_eq!(signs, vec![0.0, 1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_tick_rule_empty() {
        let signs = tick_rule_classify(&[]);
        assert!(signs.is_empty());
    }

    #[test]
    fn test_tick_rule_single() {
        let signs = tick_rule_classify(&[100.0]);
        assert_eq!(signs, vec![0.0]);
    }
}
