use ndarray::Array2;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Sequential bootstrap: draw samples respecting temporal dependencies.
///
/// Implements Snippets 4.5-4.6 from the book. The algorithm:
///
/// 1. Start with all indices as candidates.
/// 2. For each draw, compute the average uniqueness of each candidate
///    given the already-selected samples.
/// 3. Sample with probability proportional to uniqueness.
/// 4. Repeat until `num_samples` are drawn.
///
/// # Arguments
/// * `ind_matrix` - binary indicator matrix from `get_indicator_matrix`, shape `(num_bars, num_events)`
/// * `num_samples` - number of samples to draw
/// * `seed` - random seed for reproducibility
///
/// # Returns
/// A vector of `num_samples` indices drawn via the sequential bootstrap procedure.
pub fn seq_bootstrap(ind_matrix: &Array2<f64>, num_samples: usize, seed: u64) -> Vec<usize> {
    let num_events = ind_matrix.ncols();
    if num_events == 0 || num_samples == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected: Vec<usize> = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let uniqueness = compute_candidate_uniqueness(ind_matrix, &selected);

        // If all uniqueness values are zero or near-zero, fall back to uniform sampling
        let total: f64 = uniqueness.iter().sum();
        if total < 1e-15 {
            let idx = rng.gen_range(0..num_events);
            selected.push(idx);
            continue;
        }

        let dist = WeightedIndex::new(&uniqueness).expect("weights must be non-negative");
        let idx = dist.sample(&mut rng);
        selected.push(idx);
    }

    selected
}

/// Compute the average uniqueness of each candidate event given already-selected samples.
///
/// For each candidate `j`, we compute the average of `1/ct` over the bars where `j` is active,
/// where `ct` is the number of selected events (plus candidate `j`) active at bar `t`.
fn compute_candidate_uniqueness(ind_matrix: &Array2<f64>, selected: &[usize]) -> Vec<f64> {
    let num_bars = ind_matrix.nrows();
    let num_events = ind_matrix.ncols();

    // Count concurrency from already-selected events at each bar
    let mut concurrency = vec![0.0f64; num_bars];
    for &s in selected {
        for t in 0..num_bars {
            concurrency[t] += ind_matrix[[t, s]];
        }
    }

    let mut uniqueness = vec![0.0f64; num_events];

    for j in 0..num_events {
        let mut sum_inv_ct = 0.0;
        let mut active_count = 0usize;

        for t in 0..num_bars {
            if ind_matrix[[t, j]] > 0.0 {
                // ct includes the selected events plus this candidate
                let ct = concurrency[t] + 1.0;
                sum_inv_ct += 1.0 / ct;
                active_count += 1;
            }
        }

        if active_count > 0 {
            uniqueness[j] = sum_inv_ct / active_count as f64;
        }
    }

    uniqueness
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_seq_bootstrap_empty() {
        let matrix = Array2::<f64>::zeros((5, 0));
        let result = seq_bootstrap(&matrix, 3, 42);
        assert!(result.is_empty());
    }

    #[test]
    fn test_seq_bootstrap_zero_samples() {
        let matrix = array![[1.0], [1.0], [0.0]];
        let result = seq_bootstrap(&matrix, 0, 42);
        assert!(result.is_empty());
    }

    #[test]
    fn test_seq_bootstrap_single_event() {
        let matrix = array![[1.0], [1.0], [1.0]];
        let result = seq_bootstrap(&matrix, 5, 42);
        assert_eq!(result.len(), 5);
        // Only one event, so all draws must be index 0
        for &idx in &result {
            assert_eq!(idx, 0);
        }
    }

    #[test]
    fn test_seq_bootstrap_non_overlapping() {
        // Two non-overlapping events: should have equal uniqueness
        let matrix = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let result = seq_bootstrap(&matrix, 100, 42);
        assert_eq!(result.len(), 100);
        // All indices should be 0 or 1
        for &idx in &result {
            assert!(idx < 2);
        }
    }

    #[test]
    fn test_seq_bootstrap_deterministic() {
        let matrix = array![
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let r1 = seq_bootstrap(&matrix, 10, 123);
        let r2 = seq_bootstrap(&matrix, 10, 123);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_seq_bootstrap_favors_unique_samples() {
        // Event 0: bars 0..4 (overlaps heavily with event 1)
        // Event 1: bars 0..4 (overlaps heavily with event 0)
        // Event 2: bars 5..9 (no overlap with 0 or 1)
        let mut matrix = Array2::<f64>::zeros((10, 3));
        for t in 0..5 {
            matrix[[t, 0]] = 1.0;
            matrix[[t, 1]] = 1.0;
        }
        for t in 5..10 {
            matrix[[t, 2]] = 1.0;
        }

        // After selecting event 0, event 2 should be more unique than event 1
        let uniqueness = compute_candidate_uniqueness(&matrix, &[0]);
        assert!(
            uniqueness[2] > uniqueness[1],
            "event 2 (non-overlapping) should be more unique than event 1 (overlapping)"
        );
    }
}
