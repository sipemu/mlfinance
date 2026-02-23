/// Count the number of concurrent labels at each time index.
///
/// For each bar index `t` in `0..num_bars`, counts how many events
/// are active (i.e., `start_idx <= t <= end_idx`).
///
/// # Arguments
/// * `events` - list of `(start_idx, end_idx)` pairs representing label spans
/// * `num_bars` - total number of bars in the dataset
///
/// # Returns
/// A vector of length `num_bars` where entry `t` is the count of concurrent events at time `t`.
pub fn num_co_events(events: &[(usize, usize)], num_bars: usize) -> Vec<usize> {
    let mut counts = vec![0usize; num_bars];
    for &(start, end) in events {
        let end_clamped = end.min(num_bars.saturating_sub(1));
        for count in &mut counts[start..=end_clamped] {
            *count += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_events() {
        let counts = num_co_events(&[], 5);
        assert_eq!(counts, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_single_event() {
        let events = vec![(1, 3)];
        let counts = num_co_events(&events, 5);
        assert_eq!(counts, vec![0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_overlapping_events() {
        let events = vec![(0, 2), (1, 3), (2, 4)];
        let counts = num_co_events(&events, 5);
        assert_eq!(counts, vec![1, 2, 3, 2, 1]);
    }

    #[test]
    fn test_non_overlapping_events() {
        let events = vec![(0, 1), (3, 4)];
        let counts = num_co_events(&events, 6);
        assert_eq!(counts, vec![1, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_event_exceeds_num_bars() {
        let events = vec![(3, 10)];
        let counts = num_co_events(&events, 5);
        assert_eq!(counts, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_all_bars_covered() {
        let events = vec![(0, 4)];
        let counts = num_co_events(&events, 5);
        assert_eq!(counts, vec![1, 1, 1, 1, 1]);
    }
}
