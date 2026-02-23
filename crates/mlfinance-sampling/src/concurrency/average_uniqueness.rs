use crate::concurrency::num_co_events::num_co_events;

/// Compute average uniqueness of each sample.
///
/// For each event `i` with span `[start_i, end_i]`, uniqueness is defined as:
///
/// ```text
/// uniqueness[i] = mean(1 / ct for t in start_i..=end_i)
/// ```
///
/// where `ct` is the number of concurrent events at time `t`.
/// This corresponds to `mpSampleTW` from the book.
///
/// # Arguments
/// * `events` - list of `(start_idx, end_idx)` pairs representing label spans
/// * `num_bars` - total number of bars in the dataset
///
/// # Returns
/// A vector of length `events.len()` with the average uniqueness of each sample.
pub fn average_uniqueness(events: &[(usize, usize)], num_bars: usize) -> Vec<f64> {
    let counts = num_co_events(events, num_bars);

    events
        .iter()
        .map(|&(start, end)| {
            let end_clamped = end.min(num_bars.saturating_sub(1));
            if start > end_clamped {
                return 0.0;
            }
            let span_len = end_clamped - start + 1;
            if span_len == 0 {
                return 0.0;
            }
            let sum_inv: f64 = (start..=end_clamped)
                .map(|t| {
                    if counts[t] > 0 {
                        1.0 / counts[t] as f64
                    } else {
                        0.0
                    }
                })
                .sum();
            sum_inv / span_len as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_event_full_uniqueness() {
        let events = vec![(0, 4)];
        let uniqueness = average_uniqueness(&events, 5);
        assert_eq!(uniqueness.len(), 1);
        assert!((uniqueness[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_non_overlapping_events() {
        let events = vec![(0, 1), (3, 4)];
        let uniqueness = average_uniqueness(&events, 5);
        assert_eq!(uniqueness.len(), 2);
        assert!((uniqueness[0] - 1.0).abs() < 1e-10);
        assert!((uniqueness[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_fully_overlapping_events() {
        let events = vec![(0, 2), (0, 2)];
        let uniqueness = average_uniqueness(&events, 3);
        assert_eq!(uniqueness.len(), 2);
        // Each time step has 2 concurrent events, so uniqueness = 1/2
        assert!((uniqueness[0] - 0.5).abs() < 1e-10);
        assert!((uniqueness[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_partially_overlapping_events() {
        // Event 0: [0, 2], Event 1: [1, 3]
        // Counts: [1, 2, 2, 1]
        // Uniqueness of event 0: (1/1 + 1/2 + 1/2) / 3 = 2/3
        // Uniqueness of event 1: (1/2 + 1/2 + 1/1) / 3 = 2/3
        let events = vec![(0, 2), (1, 3)];
        let uniqueness = average_uniqueness(&events, 4);
        assert_eq!(uniqueness.len(), 2);
        assert!((uniqueness[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((uniqueness[1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_events() {
        let uniqueness = average_uniqueness(&[], 5);
        assert!(uniqueness.is_empty());
    }

    #[test]
    fn test_three_events_varying_overlap() {
        // Event 0: [0, 1], Event 1: [1, 2], Event 2: [2, 3]
        // Counts: [1, 2, 2, 1]
        // Event 0 uniqueness: (1/1 + 1/2) / 2 = 0.75
        // Event 1 uniqueness: (1/2 + 1/2) / 2 = 0.5
        // Event 2 uniqueness: (1/2 + 1/1) / 2 = 0.75
        let events = vec![(0, 1), (1, 2), (2, 3)];
        let uniqueness = average_uniqueness(&events, 4);
        assert!((uniqueness[0] - 0.75).abs() < 1e-10);
        assert!((uniqueness[1] - 0.5).abs() < 1e-10);
        assert!((uniqueness[2] - 0.75).abs() < 1e-10);
    }
}
