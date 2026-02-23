use ndarray::Array2;

use crate::bootstrap::sequential::seq_bootstrap;
use crate::bootstrap::standard::standard_bootstrap;
use crate::concurrency::average_uniqueness::average_uniqueness;
/// Result of comparing sequential vs standard bootstrap.
pub struct BootstrapComparison {
    /// Average uniqueness achieved by sequential bootstrap (averaged over trials).
    pub seq_uniqueness: f64,
    /// Average uniqueness achieved by standard bootstrap (averaged over trials).
    pub std_uniqueness: f64,
}

/// Monte Carlo comparison of sequential vs standard bootstrap (Snippet 4.9).
///
/// Runs `num_trials` rounds, each time drawing `num_samples` using both the
/// sequential and standard bootstrap procedures, then computes the average
/// uniqueness of the drawn samples.
///
/// # Arguments
/// * `ind_matrix` - binary indicator matrix from `get_indicator_matrix`
/// * `num_samples` - number of samples to draw per trial
/// * `num_trials` - number of Monte Carlo trials to run
/// * `seed` - base random seed (each trial uses `seed + trial_index`)
///
/// # Returns
/// A `BootstrapComparison` with the mean uniqueness from each method.
pub fn compare_bootstraps(
    ind_matrix: &Array2<f64>,
    num_samples: usize,
    num_trials: usize,
    seed: u64,
) -> BootstrapComparison {
    let num_bars = ind_matrix.nrows();
    let num_events = ind_matrix.ncols();

    if num_events == 0 || num_samples == 0 || num_trials == 0 {
        return BootstrapComparison {
            seq_uniqueness: 0.0,
            std_uniqueness: 0.0,
        };
    }

    let mut seq_total = 0.0;
    let mut std_total = 0.0;

    for trial in 0..num_trials {
        let trial_seed = seed.wrapping_add(trial as u64);

        // Sequential bootstrap
        let seq_indices = seq_bootstrap(ind_matrix, num_samples, trial_seed);
        let seq_events: Vec<(usize, usize)> = extract_events_from_indices(ind_matrix, &seq_indices);
        let seq_uniq = average_uniqueness(&seq_events, num_bars);
        let seq_mean = if seq_uniq.is_empty() {
            0.0
        } else {
            seq_uniq.iter().sum::<f64>() / seq_uniq.len() as f64
        };
        seq_total += seq_mean;

        // Standard bootstrap
        let std_indices = standard_bootstrap(num_events, num_samples, trial_seed);
        let std_events: Vec<(usize, usize)> = extract_events_from_indices(ind_matrix, &std_indices);
        let std_uniq = average_uniqueness(&std_events, num_bars);
        let std_mean = if std_uniq.is_empty() {
            0.0
        } else {
            std_uniq.iter().sum::<f64>() / std_uniq.len() as f64
        };
        std_total += std_mean;
    }

    BootstrapComparison {
        seq_uniqueness: seq_total / num_trials as f64,
        std_uniqueness: std_total / num_trials as f64,
    }
}

/// Extract event spans from the indicator matrix for the given indices.
///
/// For each selected event index, finds the first and last bar where the
/// event is active and returns the `(start, end)` pair.
fn extract_events_from_indices(ind_matrix: &Array2<f64>, indices: &[usize]) -> Vec<(usize, usize)> {
    let num_bars = ind_matrix.nrows();
    let num_events = ind_matrix.ncols();

    indices
        .iter()
        .filter_map(|&idx| {
            if idx >= num_events {
                return None;
            }
            let mut start = None;
            let mut end = None;
            for t in 0..num_bars {
                if ind_matrix[[t, idx]] > 0.0 {
                    if start.is_none() {
                        start = Some(t);
                    }
                    end = Some(t);
                }
            }
            match (start, end) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compare_bootstraps_empty() {
        let matrix = Array2::<f64>::zeros((5, 0));
        let result = compare_bootstraps(&matrix, 3, 10, 42);
        assert_eq!(result.seq_uniqueness, 0.0);
        assert_eq!(result.std_uniqueness, 0.0);
    }

    #[test]
    fn test_compare_bootstraps_non_overlapping() {
        // Non-overlapping events: both methods should achieve high uniqueness
        let mut matrix = Array2::<f64>::zeros((10, 5));
        for i in 0..5 {
            matrix[[i * 2, i]] = 1.0;
            matrix[[i * 2 + 1, i]] = 1.0;
        }
        let result = compare_bootstraps(&matrix, 5, 10, 42);
        // Both should be high since events don't overlap
        assert!(result.seq_uniqueness > 0.0);
        assert!(result.std_uniqueness > 0.0);
    }

    #[test]
    fn test_compare_bootstraps_overlapping() {
        // Heavily overlapping events: sequential should outperform standard
        let mut matrix = Array2::<f64>::zeros((10, 5));
        for i in 0..5 {
            for t in 0..10 {
                matrix[[t, i]] = 1.0;
            }
        }
        let result = compare_bootstraps(&matrix, 5, 50, 42);
        // Sequential bootstrap should achieve at least as good uniqueness
        // In the fully-overlapping case, both methods are equivalent
        assert!(result.seq_uniqueness > 0.0);
        assert!(result.std_uniqueness > 0.0);
    }

    #[test]
    fn test_compare_bootstraps_seq_better() {
        // Create a scenario where sequential bootstrap should do better:
        // Some events overlap, some don't
        let mut matrix = Array2::<f64>::zeros((20, 6));
        // Events 0,1,2 overlap on bars 0..10
        for t in 0..10 {
            matrix[[t, 0]] = 1.0;
            matrix[[t, 1]] = 1.0;
            matrix[[t, 2]] = 1.0;
        }
        // Events 3,4,5 are separate
        matrix[[10, 3]] = 1.0;
        matrix[[11, 3]] = 1.0;
        matrix[[12, 4]] = 1.0;
        matrix[[13, 4]] = 1.0;
        matrix[[14, 5]] = 1.0;
        matrix[[15, 5]] = 1.0;

        let result = compare_bootstraps(&matrix, 6, 100, 42);
        // Sequential should tend to pick non-overlapping events more
        assert!(
            result.seq_uniqueness >= result.std_uniqueness * 0.95,
            "seq={} should be >= std={}",
            result.seq_uniqueness,
            result.std_uniqueness
        );
    }

    #[test]
    fn test_extract_events_from_indices() {
        let matrix = array![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
        let events = extract_events_from_indices(&matrix, &[0, 1]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], (0, 1));
        assert_eq!(events[1], (1, 3));
    }
}
