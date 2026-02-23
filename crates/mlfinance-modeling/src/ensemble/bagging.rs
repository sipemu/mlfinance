/// Bagging accuracy formula (Snippet 6.1).
///
/// Computes the expected accuracy of a bagging classifier given `n` independent
/// base classifiers, each with individual accuracy `p`.
///
/// The ensemble predicts by majority vote. The accuracy is:
///   sum_{k=ceil(n/2)}^{n} C(n,k) * p^k * (1-p)^(n-k)
///
/// # Arguments
/// * `n` - Number of base classifiers (must be > 0).
/// * `p` - Accuracy of each base classifier (probability of correct prediction, in [0, 1]).
///
/// # Returns
/// Expected accuracy of the bagged ensemble.
pub fn bagging_accuracy(n: usize, p: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    // Majority vote wins when k > n/2. For even n, we need k >= n/2 + 1.
    // For odd n, we need k >= (n+1)/2.
    // Using integer arithmetic: threshold = n/2 + 1 handles both.
    let threshold = n / 2 + 1;

    let mut accuracy = 0.0;
    // Compute C(n,k) iteratively using log-space to avoid overflow for large n.
    for k in threshold..=n {
        let log_comb = log_combination(n, k);
        let log_term = log_comb + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln();
        accuracy += log_term.exp();
    }

    // For even n, there is a tie case at k = n/2. We attribute half to correct.
    if n % 2 == 0 {
        let k = n / 2;
        let log_comb = log_combination(n, k);
        let log_term = log_comb + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln();
        accuracy += 0.5 * log_term.exp();
    }

    accuracy.clamp(0.0, 1.0)
}

/// Compute ln(C(n, k)) = ln(n!) - ln(k!) - ln((n-k)!)
fn log_combination(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    // Use the smaller of k and n-k to reduce computation
    let k = k.min(n - k);
    let mut result = 0.0;
    for i in 0..k {
        result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bagging_accuracy_perfect_classifier() {
        let acc = bagging_accuracy(10, 1.0);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bagging_accuracy_random_classifier() {
        let acc = bagging_accuracy(10, 0.5);
        assert!((acc - 0.5).abs() < 0.05, "Expected ~0.5, got {}", acc);
    }

    #[test]
    fn test_bagging_accuracy_good_classifier() {
        // With p=0.6 and n=10, bagging should improve accuracy
        let acc = bagging_accuracy(10, 0.6);
        assert!(acc > 0.6, "Bagging should improve accuracy, got {}", acc);
    }

    #[test]
    fn test_bagging_accuracy_bad_classifier() {
        // With p=0.4 (worse than random), bagging makes it even worse
        let acc = bagging_accuracy(10, 0.4);
        assert!(acc < 0.4, "Expected acc < 0.4, got {}", acc);
    }

    #[test]
    fn test_bagging_accuracy_single_classifier() {
        let acc = bagging_accuracy(1, 0.7);
        assert!((acc - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_bagging_accuracy_zero_classifiers() {
        let acc = bagging_accuracy(0, 0.7);
        assert!((acc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bagging_accuracy_many_classifiers() {
        // With many classifiers and p>0.5, accuracy should approach 1.0
        let acc = bagging_accuracy(101, 0.6);
        assert!(acc > 0.95, "Expected high accuracy, got {}", acc);
    }

    #[test]
    fn test_log_combination() {
        // C(5,2) = 10
        let lc = log_combination(5, 2);
        assert!((lc.exp() - 10.0).abs() < 1e-10);

        // C(10,0) = 1
        let lc = log_combination(10, 0);
        assert!((lc - 0.0).abs() < 1e-10);
    }
}
