//! Mean Decrease Impurity feature importance (Snippet 8.2).
//!
//! Computes mean importance per feature across all trees in an ensemble.

use ndarray::Array1;

/// Compute MDI feature importance from tree-based model.
///
/// Each tree in the ensemble provides a vector of feature importances.
/// This function computes the mean importance for each feature across all trees.
///
/// # Arguments
/// * `importances_per_tree` - Each inner `Vec<f64>` is the feature importances from one tree.
///   All inner vectors must have the same length (the number of features).
///
/// # Returns
/// Mean importance per feature as an `Array1<f64>`.
///
/// # Panics
/// Panics if `importances_per_tree` is empty or if inner vectors have inconsistent lengths.
pub fn mean_decrease_impurity(importances_per_tree: &[Vec<f64>]) -> Array1<f64> {
    assert!(
        !importances_per_tree.is_empty(),
        "importances_per_tree must not be empty"
    );

    let n_features = importances_per_tree[0].len();
    for (i, imp) in importances_per_tree.iter().enumerate() {
        assert_eq!(
            imp.len(),
            n_features,
            "Tree {} has {} features, expected {}",
            i,
            imp.len(),
            n_features
        );
    }

    let n_trees = importances_per_tree.len();
    let mut mean_importances = Array1::zeros(n_features);

    for imp in importances_per_tree {
        for (j, &val) in imp.iter().enumerate() {
            mean_importances[j] += val;
        }
    }

    mean_importances /= n_trees as f64;
    mean_importances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_decrease_impurity_basic() {
        let importances = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.6, 0.2, 0.2],
        ];
        let result = mean_decrease_impurity(&importances);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_mean_decrease_impurity_single_tree() {
        let importances = vec![vec![0.7, 0.2, 0.1]];
        let result = mean_decrease_impurity(&importances);
        assert!((result[0] - 0.7).abs() < 1e-10);
        assert!((result[1] - 0.2).abs() < 1e-10);
        assert!((result[2] - 0.1).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "importances_per_tree must not be empty")]
    fn test_mean_decrease_impurity_empty() {
        let importances: Vec<Vec<f64>> = vec![];
        mean_decrease_impurity(&importances);
    }

    #[test]
    #[should_panic(expected = "Tree 1 has 2 features, expected 3")]
    fn test_mean_decrease_impurity_inconsistent_lengths() {
        let importances = vec![vec![0.5, 0.3, 0.2], vec![0.6, 0.4]];
        mean_decrease_impurity(&importances);
    }

    #[test]
    fn test_mean_decrease_impurity_sum_conservation() {
        // If each tree's importances sum to 1, the mean should also sum to 1
        let importances = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.4, 0.2]];
        let result = mean_decrease_impurity(&importances);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
