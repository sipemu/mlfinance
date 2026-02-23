//! Cross-validation scoring with purged K-Fold (Snippet 7.4).

use mlfinance_core::traits::Classifier;
use ndarray::{Array1, Array2};

use crate::cross_validation::purged_kfold::PurgedKFold;

/// Compute cross-validated scores using purged k-fold.
///
/// For each fold, train the classifier on the training set and score on the test set.
///
/// # Arguments
/// * `classifier` - A mutable reference to a classifier implementing the `Classifier` trait.
/// * `x` - Feature matrix of shape (n_samples, n_features).
/// * `y` - Target vector of length n_samples.
/// * `events` - Slice of (start_idx, end_idx) pairs for each sample's event.
/// * `n_splits` - Number of cross-validation folds.
/// * `embargo_pct` - Fraction of total samples to embargo.
/// * `sample_weight` - Optional sample weights for fitting.
/// * `scoring` - Scoring function: takes (y_true, y_pred) and returns a score.
///
/// # Returns
/// Vector of scores, one per fold.
#[allow(clippy::too_many_arguments)]
pub fn cv_score(
    classifier: &mut dyn Classifier,
    x: &Array2<f64>,
    y: &Array1<f64>,
    events: &[(usize, usize)],
    n_splits: usize,
    embargo_pct: f64,
    sample_weight: Option<&Array1<f64>>,
    scoring: &dyn Fn(&Array1<f64>, &Array1<f64>) -> f64,
) -> Vec<f64> {
    let n_samples = x.nrows();
    assert_eq!(y.len(), n_samples, "y length must match x rows");
    assert_eq!(events.len(), n_samples, "events length must match x rows");

    let kfold = PurgedKFold::new(n_splits, embargo_pct);
    let folds = kfold.split(events, n_samples);

    let mut scores = Vec::with_capacity(n_splits);

    for fold in &folds {
        if fold.train.is_empty() || fold.test.is_empty() {
            continue;
        }

        // Extract training data
        let n_features = x.ncols();
        let x_train = select_rows(x, &fold.train, n_features);
        let y_train = select_elements(y, &fold.train);

        let train_weights = sample_weight.map(|sw| select_elements(sw, &fold.train));

        // Fit classifier
        classifier.fit(&x_train, &y_train, train_weights.as_ref());

        // Extract test data
        let x_test = select_rows(x, &fold.test, n_features);
        let y_test = select_elements(y, &fold.test);

        // Predict and score
        let y_pred = classifier.predict(&x_test);
        let score = scoring(&y_test, &y_pred);
        scores.push(score);
    }

    scores
}

/// Select rows from a 2D array by index.
fn select_rows(x: &Array2<f64>, indices: &[usize], n_features: usize) -> Array2<f64> {
    let n = indices.len();
    let mut result = Array2::zeros((n, n_features));
    for (row_idx, &sample_idx) in indices.iter().enumerate() {
        result.row_mut(row_idx).assign(&x.row(sample_idx));
    }
    result
}

/// Select elements from a 1D array by index.
fn select_elements(arr: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_iter(indices.iter().map(|&i| arr[i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    /// A simple threshold classifier for testing.
    struct ThresholdClassifier {
        threshold: f64,
    }

    impl ThresholdClassifier {
        fn new() -> Self {
            ThresholdClassifier { threshold: 0.0 }
        }
    }

    impl Classifier for ThresholdClassifier {
        fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, _sample_weight: Option<&Array1<f64>>) {
            // Find a simple threshold: mean of feature 0 for positive/negative classes
            let mut pos_sum = 0.0;
            let mut neg_sum = 0.0;
            let mut pos_count = 0.0;
            let mut neg_count = 0.0;
            for i in 0..x.nrows() {
                if y[i] > 0.0 {
                    pos_sum += x[[i, 0]];
                    pos_count += 1.0;
                } else {
                    neg_sum += x[[i, 0]];
                    neg_count += 1.0;
                }
            }
            let pos_mean = if pos_count > 0.0 {
                pos_sum / pos_count
            } else {
                0.0
            };
            let neg_mean = if neg_count > 0.0 {
                neg_sum / neg_count
            } else {
                0.0
            };
            self.threshold = (pos_mean + neg_mean) / 2.0;
        }

        fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
            Array1::from_iter((0..x.nrows()).map(|i| {
                if x[[i, 0]] > self.threshold {
                    1.0
                } else {
                    -1.0
                }
            }))
        }

        fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
            let preds = self.predict(x);
            let mut proba = Array2::zeros((x.nrows(), 2));
            for i in 0..x.nrows() {
                if preds[i] > 0.0 {
                    proba[[i, 0]] = 0.0;
                    proba[[i, 1]] = 1.0;
                } else {
                    proba[[i, 0]] = 1.0;
                    proba[[i, 1]] = 0.0;
                }
            }
            proba
        }

        fn feature_importances(&self) -> Option<Array1<f64>> {
            Some(array![1.0])
        }
    }

    fn accuracy_scorer(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| (t - p).abs() < 1e-10)
            .count();
        correct as f64 / y_true.len() as f64
    }

    #[test]
    fn test_cv_score_basic() {
        // Create separable data
        let n = 20;
        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[[i, 0]] = i as f64;
            y[i] = if i < n / 2 { -1.0 } else { 1.0 };
        }

        let events: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        let mut clf = ThresholdClassifier::new();

        let scores = cv_score(&mut clf, &x, &y, &events, 5, 0.0, None, &accuracy_scorer);

        assert_eq!(scores.len(), 5);
        // Scores should be reasonable (the data is linearly separable)
        for &s in &scores {
            assert!(s >= 0.0 && s <= 1.0, "Score out of range: {}", s);
        }
    }

    #[test]
    fn test_select_rows() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let selected = select_rows(&x, &[0, 2], 2);
        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected[[0, 0]], 1.0);
        assert_eq!(selected[[1, 0]], 5.0);
    }

    #[test]
    fn test_select_elements() {
        let arr = array![10.0, 20.0, 30.0, 40.0];
        let selected = select_elements(&arr, &[1, 3]);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 20.0);
        assert_eq!(selected[1], 40.0);
    }
}
