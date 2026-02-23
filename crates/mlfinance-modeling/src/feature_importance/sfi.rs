//! Single Feature Importance (Snippet 8.4).
//!
//! Train the classifier on each feature individually and compute the cross-validated
//! score for each single-feature model.

use mlfinance_core::traits::Classifier;
use ndarray::{Array1, Array2};

use crate::cross_validation::cv_score::cv_score;

/// Compute Single Feature Importance.
///
/// For each feature, trains the classifier using only that feature and computes
/// the mean cross-validated score. This measures each feature's individual
/// predictive power.
///
/// # Arguments
/// * `classifier` - A mutable classifier implementing the `Classifier` trait.
/// * `x` - Feature matrix of shape (n_samples, n_features).
/// * `y` - Target vector of length n_samples.
/// * `events` - Slice of (start_idx, end_idx) pairs for each sample's event.
/// * `n_splits` - Number of cross-validation folds.
/// * `scoring` - Scoring function: takes (y_true, y_pred) and returns a score.
///
/// # Returns
/// Vector of mean CV scores, one per feature.
pub fn single_feature_importance(
    classifier: &mut dyn Classifier,
    x: &Array2<f64>,
    y: &Array1<f64>,
    events: &[(usize, usize)],
    n_splits: usize,
    scoring: &dyn Fn(&Array1<f64>, &Array1<f64>) -> f64,
) -> Vec<f64> {
    let n_features = x.ncols();
    let mut feature_scores = Vec::with_capacity(n_features);

    for j in 0..n_features {
        // Extract single-feature matrix
        let x_single = x
            .column(j)
            .to_owned()
            .into_shape_with_order((x.nrows(), 1))
            .unwrap();

        let scores = cv_score(
            classifier, &x_single, y, events, n_splits, 0.0, // no embargo for SFI
            None, scoring,
        );

        // Mean score across folds
        let mean_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        feature_scores.push(mean_score);
    }

    feature_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// A very simple classifier that predicts based on whether the first
    /// feature is above the training mean.
    struct MeanClassifier {
        mean: f64,
    }

    impl MeanClassifier {
        fn new() -> Self {
            MeanClassifier { mean: 0.0 }
        }
    }

    impl Classifier for MeanClassifier {
        fn fit(&mut self, x: &Array2<f64>, _y: &Array1<f64>, _sample_weight: Option<&Array1<f64>>) {
            self.mean = x.column(0).mean().unwrap_or(0.0);
        }

        fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
            Array1::from_iter(
                (0..x.nrows()).map(|i| if x[[i, 0]] > self.mean { 1.0 } else { -1.0 }),
            )
        }

        fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
            Array2::zeros((x.nrows(), 2))
        }

        fn feature_importances(&self) -> Option<Array1<f64>> {
            None
        }
    }

    fn accuracy_scorer(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| (t - p).abs() < 1e-10)
            .count();
        correct as f64 / y_true.len().max(1) as f64
    }

    #[test]
    fn test_single_feature_importance_returns_correct_length() {
        let n = 20;
        let n_features = 3;
        let x = Array2::from_shape_fn((n, n_features), |(i, j)| (i * n_features + j) as f64);
        let y = Array1::from_iter((0..n).map(|i| if i < n / 2 { -1.0 } else { 1.0 }));
        let events: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();

        let mut clf = MeanClassifier::new();
        let scores = single_feature_importance(&mut clf, &x, &y, &events, 3, &accuracy_scorer);

        assert_eq!(scores.len(), n_features);
        for &s in &scores {
            assert!((0.0..=1.0).contains(&s), "Score out of range: {}", s);
        }
    }
}
