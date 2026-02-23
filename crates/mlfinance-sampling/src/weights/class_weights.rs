use std::collections::HashMap;

/// Compute balanced class weights inversely proportional to class frequency.
///
/// For each class `c` with count `n_c` among `N` total samples and `K` distinct classes:
///
/// ```text
/// weight[c] = N / (K * n_c)
/// ```
///
/// This is the standard "balanced" weighting used in scikit-learn and similar libraries
/// to give underrepresented classes higher weight.
///
/// # Arguments
/// * `labels` - integer class labels for each sample
///
/// # Returns
/// A `HashMap` mapping each class label to its weight.
pub fn balanced_class_weights(labels: &[i32]) -> HashMap<i32, f64> {
    let mut weights = HashMap::new();
    if labels.is_empty() {
        return weights;
    }

    // Count frequency of each class
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }

    let n = labels.len() as f64;
    let k = counts.len() as f64;

    for (&label, &count) in &counts {
        weights.insert(label, n / (k * count as f64));
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let w = balanced_class_weights(&[]);
        assert!(w.is_empty());
    }

    #[test]
    fn test_single_class() {
        let w = balanced_class_weights(&[1, 1, 1, 1]);
        assert_eq!(w.len(), 1);
        // N=4, K=1, n_c=4 -> weight = 4/(1*4) = 1.0
        assert!((w[&1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_balanced_classes() {
        let w = balanced_class_weights(&[0, 0, 1, 1]);
        assert_eq!(w.len(), 2);
        // N=4, K=2, n_0=2, n_1=2 -> weight = 4/(2*2) = 1.0
        assert!((w[&0] - 1.0).abs() < 1e-10);
        assert!((w[&1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_imbalanced_classes() {
        let w = balanced_class_weights(&[0, 0, 0, 1]);
        assert_eq!(w.len(), 2);
        // N=4, K=2
        // n_0=3 -> weight = 4/(2*3) = 2/3
        // n_1=1 -> weight = 4/(2*1) = 2.0
        assert!((w[&0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((w[&1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_three_classes() {
        let labels = vec![-1, -1, 0, 1, 1, 1];
        let w = balanced_class_weights(&labels);
        assert_eq!(w.len(), 3);
        // N=6, K=3
        // n_{-1}=2 -> weight = 6/(3*2) = 1.0
        // n_0=1 -> weight = 6/(3*1) = 2.0
        // n_1=3 -> weight = 6/(3*3) = 2/3
        assert!((w[&-1] - 1.0).abs() < 1e-10);
        assert!((w[&0] - 2.0).abs() < 1e-10);
        assert!((w[&1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_labels() {
        let w = balanced_class_weights(&[-1, -1, 1, 1]);
        assert_eq!(w.len(), 2);
        assert!((w[&-1] - 1.0).abs() < 1e-10);
        assert!((w[&1] - 1.0).abs() < 1e-10);
    }
}
