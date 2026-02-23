//! Symmetric CUSUM filter (Snippet 2.4 from "Advances in Financial Machine Learning").
//!
//! The CUSUM filter detects structural breaks in a time series by tracking
//! positive and negative cumulative sums of changes. When either exceeds a
//! threshold, an event is signaled and the counters are reset.

/// Symmetric CUSUM filter.
///
/// Detects structural breaks by tracking positive and negative cumulative sums
/// of changes in the input values. An event is triggered when either the
/// positive or negative cumulative sum exceeds the threshold.
///
/// # Arguments
/// - `values`: the input time series (e.g., log prices or returns)
/// - `threshold`: the CUSUM threshold `h` for event detection
///
/// # Returns
/// A `Vec<usize>` of indices where events are detected.
///
/// # Example
/// ```
/// use mlfinance_data::sampling::cusum_filter::cusum_filter;
///
/// let values = vec![100.0, 101.0, 102.5, 100.0, 98.0, 97.0, 99.0];
/// let events = cusum_filter(&values, 2.0);
/// // Events are triggered where cumulative change exceeds 2.0
/// ```
pub fn cusum_filter(values: &[f64], threshold: f64) -> Vec<usize> {
    if values.len() < 2 {
        return vec![];
    }

    let mut events = Vec::new();
    let mut s_pos = 0.0_f64;
    let mut s_neg = 0.0_f64;

    for i in 1..values.len() {
        let diff = values[i] - values[i - 1];

        s_pos = (s_pos + diff).max(0.0);
        s_neg = (s_neg + diff).min(0.0);

        if s_pos >= threshold || s_neg <= -threshold {
            events.push(i);
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_no_events() {
        let values = vec![100.0, 100.1, 100.0, 100.1, 100.0];
        let events = cusum_filter(&values, 5.0);
        assert!(events.is_empty());
    }

    #[test]
    fn test_cusum_upward_break() {
        // Steady increase: each step +1, threshold at 2
        let values = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let events = cusum_filter(&values, 2.0);
        // After index 2: s_pos = 2.0 >= 2.0 -> event at index 2
        assert!(!events.is_empty());
        assert_eq!(events[0], 2);
    }

    #[test]
    fn test_cusum_downward_break() {
        let values = vec![100.0, 99.0, 98.0, 97.0, 96.0];
        let events = cusum_filter(&values, 2.0);
        // After index 2: s_neg = -2.0 <= -2.0 -> event at index 2
        assert!(!events.is_empty());
        assert_eq!(events[0], 2);
    }

    #[test]
    fn test_cusum_multiple_events() {
        let values = vec![100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0];
        let events = cusum_filter(&values, 2.0);
        // At least two events: one up, one down
        assert!(events.len() >= 2);
    }

    #[test]
    fn test_cusum_reset_after_event() {
        // After an event, counters reset
        let values = vec![100.0, 103.0, 103.0, 106.0];
        let events = cusum_filter(&values, 2.5);
        // index 1: s_pos=3 >= 2.5 -> event, reset
        // index 2: s_pos=0 (no change)
        // index 3: s_pos=3 >= 2.5 -> event
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], 1);
        assert_eq!(events[1], 3);
    }

    #[test]
    fn test_cusum_empty_input() {
        assert_eq!(cusum_filter(&[], 1.0), Vec::<usize>::new());
    }

    #[test]
    fn test_cusum_single_value() {
        assert_eq!(cusum_filter(&[100.0], 1.0), Vec::<usize>::new());
    }
}
