//! Drop labels: remove rare classes (Snippet 3.8).
//!
//! When label classes are heavily imbalanced, rare classes can be dropped
//! from the training set to improve model performance. This module provides
//! utilities to count label occurrences and remove samples whose labels
//! are too rare.

use std::collections::HashMap;

/// Count occurrences of each label value.
///
/// # Arguments
/// * `labels` - Slice of integer labels.
///
/// # Returns
/// A `HashMap` mapping each unique label value to its count.
pub fn label_counts(labels: &[i32]) -> HashMap<i32, usize> {
    let mut counts = HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    counts
}

/// Drop labels that represent less than `min_pct` of total.
///
/// Removes samples (label + corresponding feature row) in-place for any
/// label class that occurs in fewer than `min_pct` fraction of all samples.
///
/// # Arguments
/// * `labels` - Mutable vector of labels. Modified in-place.
/// * `features` - Mutable vector of feature rows (one `Vec<f64>` per sample).
///   Must have the same length as `labels`. Modified in-place.
/// * `min_pct` - Minimum fraction (0.0 to 1.0) of total samples a class
///   must represent to be retained.
///
/// # Panics
/// This function does not panic, but if `labels` and `features` have
/// different lengths, it processes up to the shorter length.
pub fn drop_labels(labels: &mut Vec<i32>, features: &mut Vec<Vec<f64>>, min_pct: f64) {
    if labels.is_empty() {
        return;
    }

    let total = labels.len() as f64;
    let counts = label_counts(labels);

    // Determine which label classes to keep
    let keep_classes: std::collections::HashSet<i32> = counts
        .into_iter()
        .filter(|&(_, count)| (count as f64 / total) >= min_pct)
        .map(|(label, _)| label)
        .collect();

    // Build mask of which samples to keep
    let mask: Vec<bool> = labels.iter().map(|l| keep_classes.contains(l)).collect();

    // Filter both labels and features in-place
    let mut write_idx = 0;
    for read_idx in 0..mask.len().min(labels.len()).min(features.len()) {
        if mask[read_idx] {
            labels[write_idx] = labels[read_idx];
            features[write_idx] = features[read_idx].clone();
            write_idx += 1;
        }
    }
    labels.truncate(write_idx);
    features.truncate(write_idx);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_counts_basic() {
        let labels = vec![1, -1, 1, 0, 1, -1, 1];
        let counts = label_counts(&labels);
        assert_eq!(counts[&1], 4);
        assert_eq!(counts[&-1], 2);
        assert_eq!(counts[&0], 1);
    }

    #[test]
    fn test_label_counts_empty() {
        let labels: Vec<i32> = vec![];
        let counts = label_counts(&labels);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_label_counts_single_class() {
        let labels = vec![1, 1, 1];
        let counts = label_counts(&labels);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&1], 3);
    }

    #[test]
    fn test_drop_labels_removes_rare() {
        // 10 samples: 5 ones, 4 minus-ones, 1 zero
        let mut labels = vec![1, 1, 1, 1, 1, -1, -1, -1, -1, 0];
        let mut features: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();

        // min_pct = 0.15 => class must have >= 15% representation
        // 1: 50% (keep), -1: 40% (keep), 0: 10% (drop)
        drop_labels(&mut labels, &mut features, 0.15);

        assert_eq!(labels.len(), 9);
        assert_eq!(features.len(), 9);
        assert!(!labels.contains(&0));
    }

    #[test]
    fn test_drop_labels_none_dropped() {
        let mut labels = vec![1, -1, 1, -1];
        let mut features: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();

        // Both at 50%, threshold 40% => nothing dropped
        drop_labels(&mut labels, &mut features, 0.4);

        assert_eq!(labels.len(), 4);
        assert_eq!(features.len(), 4);
    }

    #[test]
    fn test_drop_labels_all_dropped() {
        let mut labels = vec![1, -1, 0];
        let mut features: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64]).collect();

        // Each class at 33%, threshold 50% => all dropped
        drop_labels(&mut labels, &mut features, 0.5);

        assert!(labels.is_empty());
        assert!(features.is_empty());
    }

    #[test]
    fn test_drop_labels_empty() {
        let mut labels: Vec<i32> = vec![];
        let mut features: Vec<Vec<f64>> = vec![];
        drop_labels(&mut labels, &mut features, 0.1);
        assert!(labels.is_empty());
        assert!(features.is_empty());
    }

    #[test]
    fn test_drop_labels_preserves_features() {
        let mut labels = vec![1, 0, 1, 1];
        let mut features = vec![
            vec![10.0, 20.0],
            vec![30.0, 40.0],
            vec![50.0, 60.0],
            vec![70.0, 80.0],
        ];

        // 1: 75% (keep), 0: 25% (drop with min_pct=0.3)
        drop_labels(&mut labels, &mut features, 0.3);

        assert_eq!(labels, vec![1, 1, 1]);
        assert_eq!(
            features,
            vec![vec![10.0, 20.0], vec![50.0, 60.0], vec![70.0, 80.0]]
        );
    }
}
