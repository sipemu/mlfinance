//! Purged K-Fold cross-validation with embargo (Snippets 7.1-7.3).
//!
//! When sample i is in the test set, purge from training any sample whose event
//! window overlaps with sample i's event window. Additionally, embargo samples
//! immediately after the test set to prevent information leakage.

/// Configuration for purged K-Fold cross-validation.
#[derive(Debug, Clone)]
pub struct PurgedKFold {
    /// Number of splits.
    pub n_splits: usize,
    /// Fraction of total samples to embargo after each test set.
    pub embargo_pct: f64,
}

/// Indices for a single train/test fold.
#[derive(Debug, Clone)]
pub struct FoldIndices {
    /// Indices of training samples.
    pub train: Vec<usize>,
    /// Indices of test samples.
    pub test: Vec<usize>,
}

impl PurgedKFold {
    /// Create a new PurgedKFold configuration.
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds (must be >= 2).
    /// * `embargo_pct` - Fraction of total samples to embargo (in [0, 1]).
    pub fn new(n_splits: usize, embargo_pct: f64) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        assert!(
            (0.0..=1.0).contains(&embargo_pct),
            "embargo_pct must be in [0, 1]"
        );
        PurgedKFold {
            n_splits,
            embargo_pct,
        }
    }

    /// Generate purged k-fold splits.
    ///
    /// # Arguments
    /// * `events` - Slice of (start_idx, end_idx) pairs for each sample. The start and end
    ///   indices denote the time span of each sample's event (e.g., the bar indices that
    ///   the label spans).
    /// * `n_samples` - Total number of samples.
    ///
    /// # Returns
    /// A vector of `FoldIndices`, one per fold.
    pub fn split(&self, events: &[(usize, usize)], n_samples: usize) -> Vec<FoldIndices> {
        assert_eq!(
            events.len(),
            n_samples,
            "events length must equal n_samples"
        );

        let embargo_size = (n_samples as f64 * self.embargo_pct).ceil() as usize;

        // Create fold assignments: divide samples evenly into n_splits folds
        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        // Assign each sample to a fold
        let mut fold_assignments = Vec::with_capacity(n_samples);
        let mut idx = 0;
        for fold_id in 0..self.n_splits {
            let this_fold_size = fold_size + if fold_id < remainder { 1 } else { 0 };
            for _ in 0..this_fold_size {
                fold_assignments.push(fold_id);
                idx += 1;
            }
        }
        let _ = idx; // suppress unused warning

        let mut folds = Vec::with_capacity(self.n_splits);

        for fold_id in 0..self.n_splits {
            // Test indices are the samples assigned to this fold
            let test_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| fold_assignments[i] == fold_id)
                .collect();

            // Find the time range of the test set
            let test_start = test_indices.iter().map(|&i| events[i].0).min().unwrap_or(0);
            let test_end = test_indices.iter().map(|&i| events[i].1).max().unwrap_or(0);

            // Compute embargo boundary: embargo_size samples after the last test index
            let last_test_idx = test_indices.iter().copied().max().unwrap_or(0);
            let embargo_end = (last_test_idx + embargo_size).min(n_samples.saturating_sub(1));

            // Build training set: exclude any sample that overlaps with the test event window
            // or falls within the embargo period.
            let mut train_indices = Vec::new();
            for i in 0..n_samples {
                if fold_assignments[i] == fold_id {
                    // This sample is in the test set
                    continue;
                }

                let (sample_start, sample_end) = events[i];

                // Purge: remove if this sample's event overlaps with the test event window
                // Two intervals [a1,a2] and [b1,b2] overlap if a1 <= b2 and b1 <= a2
                let overlaps = sample_start <= test_end && test_start <= sample_end;
                if overlaps {
                    continue;
                }

                // Embargo: remove samples immediately after the test set
                // A sample is embargoed if its index is within embargo_size of the test set end
                if i > last_test_idx && i <= embargo_end {
                    continue;
                }

                train_indices.push(i);
            }

            folds.push(FoldIndices {
                train: train_indices,
                test: test_indices,
            });
        }

        folds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split_no_overlap() {
        // 10 samples, each with non-overlapping events
        let events: Vec<(usize, usize)> = (0..10).map(|i| (i, i)).collect();
        let kfold = PurgedKFold::new(5, 0.0);
        let folds = kfold.split(&events, 10);

        assert_eq!(folds.len(), 5);

        // Each test set should have 2 samples
        for fold in &folds {
            assert_eq!(fold.test.len(), 2);
        }

        // All samples should appear in exactly one test set
        let mut all_test: Vec<usize> = folds.iter().flat_map(|f| f.test.iter().copied()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_purging_removes_overlapping_samples() {
        // Sample 0 has event spanning indices 0..5
        // Sample 1 has event spanning indices 3..7 (overlaps with sample 0)
        // Sample 2 has event spanning indices 8..10 (no overlap)
        let events = vec![(0, 5), (3, 7), (8, 10)];
        let kfold = PurgedKFold::new(3, 0.0);
        let folds = kfold.split(&events, 3);

        // In the fold where sample 0 is the test set, sample 1 should be purged
        // (because event(1) = [3,7] overlaps with event(0) = [0,5])
        let fold_0 = &folds[0];
        assert!(fold_0.test.contains(&0));
        assert!(
            !fold_0.train.contains(&1),
            "Sample 1 should be purged from training when sample 0 is in test"
        );
        // Sample 2 should be in training (no overlap)
        assert!(fold_0.train.contains(&2));
    }

    #[test]
    fn test_embargo_removes_samples_after_test() {
        // 10 samples with non-overlapping events
        let events: Vec<(usize, usize)> = (0..10).map(|i| (i, i)).collect();
        let kfold = PurgedKFold::new(5, 0.1); // 10% embargo = 1 sample
        let folds = kfold.split(&events, 10);

        // Check that embargo samples are removed from training
        // First fold: test = {0, 1}, embargo should remove sample 2
        let fold_0 = &folds[0];
        assert!(fold_0.test.contains(&0));
        assert!(fold_0.test.contains(&1));
        assert!(
            !fold_0.train.contains(&2),
            "Sample 2 should be embargoed after test set {{0, 1}}"
        );
    }

    #[test]
    fn test_split_with_uneven_samples() {
        let events: Vec<(usize, usize)> = (0..7).map(|i| (i, i)).collect();
        let kfold = PurgedKFold::new(3, 0.0);
        let folds = kfold.split(&events, 7);

        assert_eq!(folds.len(), 3);
        let total_test: usize = folds.iter().map(|f| f.test.len()).sum();
        assert_eq!(total_test, 7);
    }

    #[test]
    #[should_panic(expected = "n_splits must be at least 2")]
    fn test_invalid_n_splits() {
        PurgedKFold::new(1, 0.0);
    }

    #[test]
    #[should_panic(expected = "embargo_pct must be in [0, 1]")]
    fn test_invalid_embargo_pct() {
        PurgedKFold::new(5, 1.5);
    }

    #[test]
    fn test_no_train_test_overlap() {
        let events: Vec<(usize, usize)> = (0..20).map(|i| (i, i + 2)).collect();
        let kfold = PurgedKFold::new(4, 0.05);
        let folds = kfold.split(&events, 20);

        for fold in &folds {
            for &test_idx in &fold.test {
                assert!(
                    !fold.train.contains(&test_idx),
                    "Test sample {} found in training set",
                    test_idx
                );
            }
        }
    }
}
