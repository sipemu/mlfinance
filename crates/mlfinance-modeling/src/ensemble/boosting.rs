//! AdaBoost formulation.
//!
//! Implements the weight update step of the AdaBoost algorithm.

/// Compute AdaBoost sample weights after one round.
///
/// # Arguments
/// * `predictions` - Predicted labels (e.g., -1.0 or 1.0).
/// * `labels` - True labels (e.g., -1.0 or 1.0).
/// * `weights` - Current sample weights (must sum to 1.0 or will be normalized).
///
/// # Returns
/// `(new_weights, alpha)` where `new_weights` are the updated and normalized sample weights
/// and `alpha` is the classifier weight for this round.
///
/// # Panics
/// Panics if the input slices have different lengths.
pub fn adaboost_update(predictions: &[f64], labels: &[f64], weights: &[f64]) -> (Vec<f64>, f64) {
    assert_eq!(
        predictions.len(),
        labels.len(),
        "predictions and labels must have same length"
    );
    assert_eq!(
        predictions.len(),
        weights.len(),
        "predictions and weights must have same length"
    );

    let n = predictions.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    // Normalize input weights
    let w_sum: f64 = weights.iter().sum();
    let norm_weights: Vec<f64> = if w_sum > 0.0 {
        weights.iter().map(|w| w / w_sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    // Compute weighted error rate
    let mut err = 0.0;
    for i in 0..n {
        if (predictions[i] - labels[i]).abs() > 1e-10 {
            err += norm_weights[i];
        }
    }

    // Clamp error to avoid division by zero or log of zero
    err = err.clamp(1e-10, 1.0 - 1e-10);

    // Compute classifier weight (alpha)
    let alpha = 0.5 * ((1.0 - err) / err).ln();

    // Update weights
    let mut new_weights = Vec::with_capacity(n);
    for i in 0..n {
        let correct = (predictions[i] - labels[i]).abs() < 1e-10;
        let factor = if correct { (-alpha).exp() } else { alpha.exp() };
        new_weights.push(norm_weights[i] * factor);
    }

    // Normalize new weights
    let new_sum: f64 = new_weights.iter().sum();
    if new_sum > 0.0 {
        for w in &mut new_weights {
            *w /= new_sum;
        }
    }

    (new_weights, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaboost_perfect_classifier() {
        let predictions = vec![1.0, -1.0, 1.0, -1.0];
        let labels = vec![1.0, -1.0, 1.0, -1.0];
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let (new_weights, alpha) = adaboost_update(&predictions, &labels, &weights);

        // Perfect classifier: error ~= 0, alpha should be large
        assert!(
            alpha > 5.0,
            "Alpha should be large for perfect classifier, got {}",
            alpha
        );
        // All weights should be equal (all correct)
        for w in &new_weights {
            assert!((w - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adaboost_half_correct() {
        let predictions = vec![1.0, 1.0, 1.0, 1.0];
        let labels = vec![1.0, 1.0, -1.0, -1.0];
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let (new_weights, alpha) = adaboost_update(&predictions, &labels, &weights);

        // 50% error rate: alpha should be ~0
        assert!(
            alpha.abs() < 0.1,
            "Alpha should be near 0 for 50% error, got {}",
            alpha
        );
        // With alpha near 0, weights stay approximately equal
        let sum: f64 = new_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights should still sum to 1");
        // All weights should be roughly equal
        for &w in &new_weights {
            assert!(
                (w - 0.25).abs() < 0.05,
                "Weight should be near 0.25, got {}",
                w
            );
        }
    }

    #[test]
    fn test_adaboost_weights_sum_to_one() {
        let predictions = vec![1.0, -1.0, 1.0, 1.0, -1.0];
        let labels = vec![1.0, 1.0, -1.0, 1.0, -1.0];
        let weights = vec![0.2, 0.2, 0.2, 0.2, 0.2];

        let (new_weights, _) = adaboost_update(&predictions, &labels, &weights);

        let sum: f64 = new_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_adaboost_empty() {
        let (new_weights, alpha) = adaboost_update(&[], &[], &[]);
        assert!(new_weights.is_empty());
        assert!((alpha - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaboost_misclassified_get_higher_weight() {
        let predictions = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let labels = vec![1.0, 1.0, 1.0, -1.0, 1.0];
        let weights = vec![0.2, 0.2, 0.2, 0.2, 0.2];

        let (new_weights, alpha) = adaboost_update(&predictions, &labels, &weights);

        // Sample 3 was misclassified, should have higher weight
        assert!(alpha > 0.0);
        assert!(
            new_weights[3] > new_weights[0],
            "Misclassified sample should have higher weight"
        );
    }
}
