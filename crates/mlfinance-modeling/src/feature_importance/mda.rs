//! Mean Decrease Accuracy feature importance (Snippet 8.3).
//!
//! Measures the importance of each feature by permuting it and observing
//! the drop in the model's score.

use mlfinance_core::traits::Classifier;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Compute Mean Decrease Accuracy feature importance.
///
/// For each feature, permute its values across samples and measure the change
/// in score. Features that cause a larger drop in score when permuted are
/// more important.
///
/// # Arguments
/// * `classifier` - A fitted classifier.
/// * `x` - Feature matrix of shape (n_samples, n_features).
/// * `y` - Target vector of length n_samples.
/// * `scoring` - Scoring function: takes (y_true, y_pred) and returns a score.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
/// Feature importance vector of length n_features: the drop in score for each feature.
pub fn mean_decrease_accuracy(
    classifier: &dyn Classifier,
    x: &Array2<f64>,
    y: &Array1<f64>,
    scoring: &dyn Fn(&Array1<f64>, &Array1<f64>) -> f64,
    seed: u64,
) -> Array1<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples == 0 || n_features == 0 {
        return Array1::zeros(n_features);
    }

    // Baseline score (no permutation)
    let y_pred = classifier.predict(x);
    let baseline_score = scoring(y, &y_pred);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importances = Array1::zeros(n_features);

    for j in 0..n_features {
        // Create a copy of x with column j permuted
        let mut x_permuted = x.clone();

        // Extract column j values and permute
        let mut col_values: Vec<f64> = (0..n_samples).map(|i| x[[i, j]]).collect();
        col_values.shuffle(&mut rng);

        // Replace column j with permuted values
        for (i, &val) in col_values.iter().enumerate() {
            x_permuted[[i, j]] = val;
        }

        // Score with permuted feature
        let y_pred_perm = classifier.predict(&x_permuted);
        let permuted_score = scoring(y, &y_pred_perm);

        // Importance = drop in score
        importances[j] = baseline_score - permuted_score;
    }

    importances
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple classifier that always returns the same predictions.
    struct ConstantClassifier {
        predictions: Array1<f64>,
    }

    impl ConstantClassifier {
        fn new(predictions: Array1<f64>) -> Self {
            Self { predictions }
        }
    }

    impl Classifier for ConstantClassifier {
        fn fit(
            &mut self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
            _sample_weight: Option<&Array1<f64>>,
        ) {
        }

        fn predict(&self, _x: &Array2<f64>) -> Array1<f64> {
            self.predictions.clone()
        }

        fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
            Array2::zeros((x.nrows(), 2))
        }

        fn feature_importances(&self) -> Option<Array1<f64>> {
            None
        }
    }

    #[test]
    fn test_mda_constant_classifier() {
        // A classifier that always predicts the same thing regardless of features
        // should have zero importance for all features.
        let y = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0]);
        let clf = ConstantClassifier::new(y.clone());
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();

        let accuracy = |y_true: &Array1<f64>, y_pred: &Array1<f64>| -> f64 {
            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&t, &p)| (t - p).abs() < 1e-10)
                .count();
            correct as f64 / y_true.len() as f64
        };

        let importances = mean_decrease_accuracy(&clf, &x, &y, &accuracy, 42);

        // All importances should be 0 since predictions don't depend on features
        assert_eq!(importances.len(), 3);
        for &imp in importances.iter() {
            assert!(imp.abs() < 1e-10, "Expected 0 importance, got {}", imp);
        }
    }

    #[test]
    fn test_mda_empty() {
        let clf = ConstantClassifier::new(Array1::zeros(0));
        let x = Array2::zeros((0, 3));
        let y = Array1::zeros(0);
        let scoring = |_y_true: &Array1<f64>, _y_pred: &Array1<f64>| -> f64 { 0.0 };

        let importances = mean_decrease_accuracy(&clf, &x, &y, &scoring, 42);
        assert_eq!(importances.len(), 3);
    }
}
