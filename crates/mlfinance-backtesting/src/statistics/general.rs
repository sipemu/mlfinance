//! General strategy statistics (Snippets 14.1-14.2).
//!
//! Basic performance metrics including hit ratio, average holding period,
//! and portfolio turnover.

/// Hit ratio: fraction of positive returns.
///
/// # Arguments
/// * `returns` - Slice of return values.
///
/// # Returns
/// Fraction in `[0, 1]`. Returns 0.0 if the slice is empty.
pub fn hit_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let positives = returns.iter().filter(|&&r| r > 0.0).count();
    positives as f64 / returns.len() as f64
}

/// Average holding period.
///
/// Given a set of entry/exit bar index pairs, computes the mean holding period.
///
/// # Arguments
/// * `entry_exit_pairs` - Slice of `(entry_bar, exit_bar)` tuples. Each pair
///   represents one trade.
///
/// # Returns
/// Average number of bars held per trade. Returns 0.0 if no trades.
pub fn avg_holding_period(entry_exit_pairs: &[(usize, usize)]) -> f64 {
    if entry_exit_pairs.is_empty() {
        return 0.0;
    }

    let total: f64 = entry_exit_pairs
        .iter()
        .map(|&(entry, exit)| {
            if exit >= entry {
                (exit - entry) as f64
            } else {
                0.0
            }
        })
        .sum();

    total / entry_exit_pairs.len() as f64
}

/// Turnover: average absolute position change.
///
/// Measures how much the portfolio position changes from bar to bar on average.
///
/// # Arguments
/// * `positions` - Slice of position values over time.
///
/// # Returns
/// Average absolute change in position. Returns 0.0 if fewer than 2 positions.
pub fn turnover(positions: &[f64]) -> f64 {
    if positions.len() < 2 {
        return 0.0;
    }

    let total_change: f64 = positions.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

    total_change / (positions.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_ratio_basic() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.005];
        let hr = hit_ratio(&returns);
        assert!((hr - 0.6).abs() < 1e-10, "hr={} expected 0.6", hr);
    }

    #[test]
    fn test_hit_ratio_all_positive() {
        let returns = vec![0.01, 0.02, 0.03];
        assert!((hit_ratio(&returns) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_ratio_all_negative() {
        let returns = vec![-0.01, -0.02, -0.03];
        assert!((hit_ratio(&returns) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_ratio_zeros() {
        // Zero returns are not positive
        let returns = vec![0.0, 0.0, 0.0];
        assert!((hit_ratio(&returns) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_ratio_empty() {
        assert_eq!(hit_ratio(&[]), 0.0);
    }

    #[test]
    fn test_avg_holding_period_basic() {
        let trades = vec![(0, 5), (10, 15), (20, 30)];
        let ahp = avg_holding_period(&trades);
        // (5 + 5 + 10) / 3 = 6.666...
        assert!((ahp - 20.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_avg_holding_period_single() {
        let trades = vec![(0, 10)];
        assert!((avg_holding_period(&trades) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_avg_holding_period_empty() {
        assert_eq!(avg_holding_period(&[]), 0.0);
    }

    #[test]
    fn test_avg_holding_period_reversed() {
        // Reversed pairs should be treated as 0 duration
        let trades = vec![(10, 5)];
        assert!((avg_holding_period(&trades) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_turnover_basic() {
        let positions = vec![0.0, 1.0, 0.5, -0.5, 0.0];
        let t = turnover(&positions);
        // Changes: 1.0, 0.5, 1.0, 0.5 -> avg = 3.0/4 = 0.75
        assert!((t - 0.75).abs() < 1e-10, "turnover={} expected 0.75", t);
    }

    #[test]
    fn test_turnover_constant() {
        let positions = vec![1.0, 1.0, 1.0];
        assert!((turnover(&positions) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_turnover_empty() {
        assert_eq!(turnover(&[]), 0.0);
    }

    #[test]
    fn test_turnover_single() {
        assert_eq!(turnover(&[1.0]), 0.0);
    }
}
