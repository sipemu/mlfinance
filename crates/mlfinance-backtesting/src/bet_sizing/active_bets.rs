//! Average active signals at each time point (Snippet 10.2).
//!
//! Given a set of overlapping signals with start/end indices, computes the
//! average active signal value at each bar.

/// Average active signals at each time point.
///
/// For each bar in `[0, num_bars)`, accumulates all signals that are active
/// at that bar (i.e., `start <= bar <= end`) and returns the average signal
/// value at each bar. Bars with no active signals receive a value of 0.0.
///
/// # Arguments
/// * `signals` - Slice of `(start, end, signal_value)` tuples. `start` and
///   `end` are inclusive bar indices.
/// * `num_bars` - Total number of bars in the time series.
///
/// # Returns
/// A vector of length `num_bars` with the average signal at each bar.
pub fn avg_active_signals(signals: &[(usize, usize, f64)], num_bars: usize) -> Vec<f64> {
    if num_bars == 0 {
        return Vec::new();
    }

    let mut sum = vec![0.0_f64; num_bars];
    let mut count = vec![0_usize; num_bars];

    for &(start, end, value) in signals {
        let end_clamped = end.min(num_bars - 1);
        for bar in start..=end_clamped {
            sum[bar] += value;
            count[bar] += 1;
        }
    }

    sum.iter()
        .zip(count.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_signal() {
        let signals = vec![(1, 3, 1.0)];
        let result = avg_active_signals(&signals, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
        assert!((result[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlapping_signals() {
        let signals = vec![(0, 2, 1.0), (1, 3, -1.0)];
        let result = avg_active_signals(&signals, 4);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10); // avg(1, -1) = 0
        assert!((result[2] - 0.0).abs() < 1e-10); // avg(1, -1) = 0
        assert!((result[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_empty_signals() {
        let signals: Vec<(usize, usize, f64)> = vec![];
        let result = avg_active_signals(&signals, 5);
        assert_eq!(result, vec![0.0; 5]);
    }

    #[test]
    fn test_zero_bars() {
        let signals = vec![(0, 2, 1.0)];
        let result = avg_active_signals(&signals, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_signal_beyond_bars() {
        // Signal extends beyond num_bars -- should be clamped
        let signals = vec![(3, 10, 2.0)];
        let result = avg_active_signals(&signals, 5);
        assert_eq!(result.len(), 5);
        assert!((result[3] - 2.0).abs() < 1e-10);
        assert!((result[4] - 2.0).abs() < 1e-10);
    }
}
