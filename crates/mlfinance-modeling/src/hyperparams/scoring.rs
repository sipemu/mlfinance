//! Scoring functions for classification (Snippet 9 support).

/// Compute the F1 score for binary classification.
///
/// The F1 score is the harmonic mean of precision and recall.
/// Labels are expected to be -1.0 or 1.0 (positive class is 1.0).
///
/// # Arguments
/// * `y_true` - True labels.
/// * `y_pred` - Predicted labels.
///
/// # Returns
/// F1 score in [0, 1]. Returns 0.0 if precision + recall = 0 or if inputs are empty.
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() || y_pred.is_empty() {
        return 0.0;
    }
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have same length"
    );

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut fn_ = 0.0_f64;

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_pos = t > 0.0;
        let p_pos = p > 0.0;

        match (t_pos, p_pos) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (true, false) => fn_ += 1.0,
            (false, false) => {} // true negative
        }
    }

    let precision = if (tp + fp) > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if (tp + fn_) > 0.0 {
        tp / (tp + fn_)
    } else {
        0.0
    };

    if (precision + recall) > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Compute the negative log loss (binary cross-entropy).
///
/// # Arguments
/// * `y_true` - True labels (values: 0.0 or 1.0, where 1.0 is positive).
/// * `y_proba` - Predicted probabilities of the positive class (in [0, 1]).
///
/// # Returns
/// Negative log loss (negative so that higher is better, following sklearn convention).
/// Returns 0.0 if inputs are empty.
pub fn neg_log_loss(y_true: &[f64], y_proba: &[f64]) -> f64 {
    if y_true.is_empty() || y_proba.is_empty() {
        return 0.0;
    }
    assert_eq!(
        y_true.len(),
        y_proba.len(),
        "y_true and y_proba must have same length"
    );

    let eps = 1e-15;
    let n = y_true.len() as f64;

    let mut total_loss = 0.0;
    for (&t, &p) in y_true.iter().zip(y_proba.iter()) {
        let p_clamped = p.clamp(eps, 1.0 - eps);
        total_loss += t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln();
    }

    // Negative log loss: negate and average
    total_loss / n
}

/// Compute accuracy score.
///
/// # Arguments
/// * `y_true` - True labels.
/// * `y_pred` - Predicted labels.
///
/// # Returns
/// Fraction of correct predictions. Returns 0.0 if inputs are empty.
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() || y_pred.is_empty() {
        return 0.0;
    }
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have same length"
    );

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-10)
        .count();

    correct as f64 / y_true.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // F1 score tests

    #[test]
    fn test_f1_score_perfect() {
        let y_true = vec![1.0, -1.0, 1.0, -1.0];
        let y_pred = vec![1.0, -1.0, 1.0, -1.0];
        assert!((f1_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score_all_wrong() {
        let y_true = vec![1.0, 1.0, 1.0, 1.0];
        let y_pred = vec![-1.0, -1.0, -1.0, -1.0];
        assert!((f1_score(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score_mixed() {
        // 2 TP, 1 FP, 1 FN
        let y_true = vec![1.0, 1.0, -1.0, 1.0];
        let y_pred = vec![1.0, 1.0, 1.0, -1.0];
        // precision = 2/3, recall = 2/3, F1 = 2/3
        let f1 = f1_score(&y_true, &y_pred);
        assert!((f1 - 2.0 / 3.0).abs() < 1e-10, "Expected 2/3, got {}", f1);
    }

    #[test]
    fn test_f1_score_empty() {
        assert!((f1_score(&[], &[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score_no_positives() {
        let y_true = vec![-1.0, -1.0, -1.0];
        let y_pred = vec![-1.0, -1.0, -1.0];
        // No positive predictions or true positives: precision and recall are 0
        assert!((f1_score(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    // Negative log loss tests

    #[test]
    fn test_neg_log_loss_perfect() {
        let y_true = vec![1.0, 0.0, 1.0];
        let y_proba = vec![0.999, 0.001, 0.999];
        let loss = neg_log_loss(&y_true, &y_proba);
        // Should be close to 0 (high confidence correct predictions)
        assert!(
            loss > -0.01,
            "Expected near 0 loss for perfect predictions, got {}",
            loss
        );
    }

    #[test]
    fn test_neg_log_loss_bad() {
        let y_true = vec![1.0, 0.0, 1.0];
        let y_proba = vec![0.001, 0.999, 0.001];
        let loss = neg_log_loss(&y_true, &y_proba);
        // Should be very negative (high confidence wrong predictions)
        assert!(loss < -1.0, "Expected large negative loss, got {}", loss);
    }

    #[test]
    fn test_neg_log_loss_empty() {
        assert!((neg_log_loss(&[], &[]) - 0.0).abs() < 1e-10);
    }

    // Accuracy score tests

    #[test]
    fn test_accuracy_score_perfect() {
        let y_true = vec![1.0, -1.0, 1.0, -1.0];
        let y_pred = vec![1.0, -1.0, 1.0, -1.0];
        assert!((accuracy_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_score_all_wrong() {
        let y_true = vec![1.0, -1.0, 1.0, -1.0];
        let y_pred = vec![-1.0, 1.0, -1.0, 1.0];
        assert!((accuracy_score(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_score_half() {
        let y_true = vec![1.0, -1.0, 1.0, -1.0];
        let y_pred = vec![1.0, 1.0, -1.0, -1.0];
        assert!((accuracy_score(&y_true, &y_pred) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_score_empty() {
        assert!((accuracy_score(&[], &[]) - 0.0).abs() < 1e-10);
    }
}
