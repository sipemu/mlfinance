//! Vertical barrier computation.
//!
//! Adds a vertical barrier at a fixed number of bars from each event entry,
//! clamped to the end of the series.

/// Given entry indices and a holding period, return the exit index for each entry.
///
/// For each entry index, computes `entry + max_holding`, clamped to
/// `series_len - 1` to prevent out-of-bounds access.
///
/// # Arguments
/// * `entry_indices` - Indices of the entry bars.
/// * `max_holding` - Maximum holding period in number of bars.
/// * `series_len` - Total length of the price series.
///
/// # Returns
/// A vector of exit indices, one per entry. If an entry index is already
/// beyond or at `series_len`, the exit index is clamped to `series_len - 1`.
pub fn add_vertical_barrier(
    entry_indices: &[usize],
    max_holding: usize,
    series_len: usize,
) -> Vec<usize> {
    if series_len == 0 {
        return vec![];
    }
    let max_valid = series_len - 1;
    entry_indices
        .iter()
        .map(|&entry| (entry + max_holding).min(max_valid))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vertical_barrier_basic() {
        let entries = vec![0, 5, 10];
        let exits = add_vertical_barrier(&entries, 3, 20);
        assert_eq!(exits, vec![3, 8, 13]);
    }

    #[test]
    fn test_add_vertical_barrier_clamped() {
        let entries = vec![0, 5, 18];
        let exits = add_vertical_barrier(&entries, 5, 20);
        assert_eq!(exits, vec![5, 10, 19]); // 18 + 5 = 23, clamped to 19
    }

    #[test]
    fn test_add_vertical_barrier_at_end() {
        let entries = vec![19];
        let exits = add_vertical_barrier(&entries, 5, 20);
        assert_eq!(exits, vec![19]); // 19 + 5 = 24, clamped to 19
    }

    #[test]
    fn test_add_vertical_barrier_empty() {
        let entries: Vec<usize> = vec![];
        let exits = add_vertical_barrier(&entries, 5, 20);
        assert!(exits.is_empty());
    }

    #[test]
    fn test_add_vertical_barrier_zero_holding() {
        let entries = vec![0, 5, 10];
        let exits = add_vertical_barrier(&entries, 0, 20);
        assert_eq!(exits, vec![0, 5, 10]);
    }

    #[test]
    fn test_add_vertical_barrier_zero_series_len() {
        let entries = vec![0];
        let exits = add_vertical_barrier(&entries, 5, 0);
        assert!(exits.is_empty());
    }

    #[test]
    fn test_add_vertical_barrier_single_element_series() {
        let entries = vec![0];
        let exits = add_vertical_barrier(&entries, 5, 1);
        assert_eq!(exits, vec![0]);
    }
}
