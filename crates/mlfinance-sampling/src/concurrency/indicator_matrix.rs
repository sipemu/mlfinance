use ndarray::Array2;

/// Build binary indicator matrix: `I[t, i] = 1.0` if event `i` is active at time `t`.
///
/// This corresponds to `getIndMatrix` from the book. The resulting matrix has
/// shape `(num_bars, num_events)` where each column represents an event and
/// each row represents a time bar.
///
/// # Arguments
/// * `events` - list of `(start_idx, end_idx)` pairs representing label spans
/// * `num_bars` - total number of bars in the dataset
///
/// # Returns
/// A 2D array of shape `(num_bars, events.len())` with `1.0` where an event is active.
pub fn get_indicator_matrix(events: &[(usize, usize)], num_bars: usize) -> Array2<f64> {
    let num_events = events.len();
    let mut matrix = Array2::<f64>::zeros((num_bars, num_events));
    for (i, &(start, end)) in events.iter().enumerate() {
        let end_clamped = end.min(num_bars.saturating_sub(1));
        for t in start..=end_clamped {
            matrix[[t, i]] = 1.0;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_events() {
        let matrix = get_indicator_matrix(&[], 3);
        assert_eq!(matrix.shape(), &[3, 0]);
    }

    #[test]
    fn test_single_event() {
        let events = vec![(1, 2)];
        let matrix = get_indicator_matrix(&events, 4);
        assert_eq!(matrix.shape(), &[4, 1]);
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[1, 0]], 1.0);
        assert_eq!(matrix[[2, 0]], 1.0);
        assert_eq!(matrix[[3, 0]], 0.0);
    }

    #[test]
    fn test_overlapping_events() {
        let events = vec![(0, 2), (1, 3)];
        let matrix = get_indicator_matrix(&events, 4);
        assert_eq!(matrix.shape(), &[4, 2]);
        // Event 0: active at 0, 1, 2
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 0]], 1.0);
        assert_eq!(matrix[[2, 0]], 1.0);
        assert_eq!(matrix[[3, 0]], 0.0);
        // Event 1: active at 1, 2, 3
        assert_eq!(matrix[[0, 1]], 0.0);
        assert_eq!(matrix[[1, 1]], 1.0);
        assert_eq!(matrix[[2, 1]], 1.0);
        assert_eq!(matrix[[3, 1]], 1.0);
    }

    #[test]
    fn test_event_clamped_to_num_bars() {
        let events = vec![(2, 10)];
        let matrix = get_indicator_matrix(&events, 5);
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[1, 0]], 0.0);
        assert_eq!(matrix[[2, 0]], 1.0);
        assert_eq!(matrix[[3, 0]], 1.0);
        assert_eq!(matrix[[4, 0]], 1.0);
    }

    #[test]
    fn test_non_overlapping_events() {
        let events = vec![(0, 0), (2, 2)];
        let matrix = get_indicator_matrix(&events, 3);
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 0]], 0.0);
        assert_eq!(matrix[[2, 0]], 0.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        assert_eq!(matrix[[2, 1]], 1.0);
    }
}
