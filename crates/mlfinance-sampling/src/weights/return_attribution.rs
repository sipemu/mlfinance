use crate::concurrency::num_co_events::num_co_events;

/// Compute sample weights based on absolute return attribution (Snippet 4.10).
///
/// Each sample's weight is proportional to its uniqueness times the absolute return.
/// Specifically, for event `i` with span `[start_i, end_i]` and return `r_i`:
///
/// ```text
/// weight[i] = |r_i| * sum(1/ct for t in start_i..=end_i) / (end_i - start_i + 1)
/// ```
///
/// where `ct` is the number of concurrent events at time `t`.
///
/// The resulting weights are normalized so that they sum to the number of events.
///
/// # Arguments
/// * `events` - list of `(start_idx, end_idx)` pairs representing label spans
/// * `returns` - return associated with each event (same length as `events`)
/// * `num_bars` - total number of bars in the dataset
///
/// # Returns
/// A vector of weights with the same length as `events`.
pub fn return_attribution_weights(
    events: &[(usize, usize)],
    returns: &[f64],
    num_bars: usize,
) -> Vec<f64> {
    assert_eq!(
        events.len(),
        returns.len(),
        "events and returns must have the same length"
    );

    if events.is_empty() {
        return Vec::new();
    }

    let counts = num_co_events(events, num_bars);

    let mut weights: Vec<f64> = events
        .iter()
        .zip(returns.iter())
        .map(|(&(start, end), &ret)| {
            let end_clamped = end.min(num_bars.saturating_sub(1));
            if start > end_clamped {
                return 0.0;
            }
            let span_len = end_clamped - start + 1;
            if span_len == 0 {
                return 0.0;
            }

            let uniqueness: f64 = (start..=end_clamped)
                .map(|t| {
                    if counts[t] > 0 {
                        1.0 / counts[t] as f64
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / span_len as f64;

            uniqueness * ret.abs()
        })
        .collect();

    // Normalize weights so they sum to the number of events
    let total: f64 = weights.iter().sum();
    if total > 1e-15 {
        let n = weights.len() as f64;
        for w in &mut weights {
            *w = *w * n / total;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let w = return_attribution_weights(&[], &[], 5);
        assert!(w.is_empty());
    }

    #[test]
    fn test_single_event() {
        let events = vec![(0, 4)];
        let returns = vec![0.05];
        let w = return_attribution_weights(&events, &returns, 5);
        assert_eq!(w.len(), 1);
        // Single event normalized to sum = 1
        assert!((w[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equal_returns_non_overlapping() {
        let events = vec![(0, 1), (2, 3)];
        let returns = vec![0.1, 0.1];
        let w = return_attribution_weights(&events, &returns, 4);
        assert_eq!(w.len(), 2);
        // Equal returns and equal uniqueness -> equal weights
        assert!((w[0] - w[1]).abs() < 1e-10);
        // Weights should sum to num_events = 2
        let total: f64 = w.iter().sum();
        assert!((total - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_different_returns() {
        let events = vec![(0, 1), (2, 3)];
        let returns = vec![0.1, 0.3];
        let w = return_attribution_weights(&events, &returns, 4);
        // Higher return -> higher weight
        assert!(w[1] > w[0]);
    }

    #[test]
    fn test_negative_returns_use_absolute_value() {
        let events = vec![(0, 1), (2, 3)];
        let returns_pos = vec![0.1, 0.3];
        let returns_neg = vec![-0.1, -0.3];
        let w_pos = return_attribution_weights(&events, &returns_pos, 4);
        let w_neg = return_attribution_weights(&events, &returns_neg, 4);
        assert!((w_pos[0] - w_neg[0]).abs() < 1e-10);
        assert!((w_pos[1] - w_neg[1]).abs() < 1e-10);
    }

    #[test]
    fn test_overlapping_events() {
        let events = vec![(0, 2), (1, 3)];
        let returns = vec![0.1, 0.1];
        let w = return_attribution_weights(&events, &returns, 4);
        // Both events have equal returns and symmetric overlap -> equal weights
        assert!((w[0] - w[1]).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "events and returns must have the same length")]
    fn test_length_mismatch() {
        return_attribution_weights(&[(0, 1)], &[0.1, 0.2], 2);
    }
}
