use ndarray::Array2;

/// Shannon entropy of a discrete distribution.
///
/// H = -sum(p_i * log2(p_i)) for p_i > 0
///
/// # Arguments
///
/// * `probs` - Probability distribution (entries should sum to 1.0).
///
/// # Returns
///
/// Entropy in bits. Returns 0.0 if all probabilities are zero.
pub fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Redundancy: 1 - H/H_max.
///
/// H_max = log2(n) where n is the number of symbols.
/// Redundancy measures how far the distribution is from uniform.
///
/// # Arguments
///
/// * `probs` - Probability distribution (entries should sum to 1.0).
///
/// # Returns
///
/// Redundancy in [0, 1], where 0 means maximum entropy (uniform) and 1 means
/// deterministic.
pub fn redundancy(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n <= 1 {
        return 0.0;
    }
    let h_max = (n as f64).log2();
    if h_max < 1e-15 {
        return 0.0;
    }
    let h = shannon_entropy(probs);
    1.0 - h / h_max
}

/// Mutual information between two discrete variables.
///
/// I(X;Y) = sum_{x,y} p(x,y) * log2(p(x,y) / (p(x) * p(y)))
///
/// # Arguments
///
/// * `joint_probs` - Joint probability matrix where `joint_probs[[i,j]] = P(X=i, Y=j)`.
///   Entries should sum to 1.0.
///
/// # Returns
///
/// Mutual information in bits (non-negative).
pub fn mutual_information(joint_probs: &Array2<f64>) -> f64 {
    let n_x = joint_probs.nrows();
    let n_y = joint_probs.ncols();

    // Marginals
    let mut p_x = vec![0.0; n_x];
    let mut p_y = vec![0.0; n_y];

    for i in 0..n_x {
        for j in 0..n_y {
            p_x[i] += joint_probs[[i, j]];
            p_y[j] += joint_probs[[i, j]];
        }
    }

    let mut mi = 0.0;
    for i in 0..n_x {
        for j in 0..n_y {
            let p_xy = joint_probs[[i, j]];
            if p_xy > 0.0 && p_x[i] > 0.0 && p_y[j] > 0.0 {
                mi += p_xy * (p_xy / (p_x[i] * p_y[j])).log2();
            }
        }
    }

    mi
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution over 4 symbols: H = log2(4) = 2
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&probs);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        // Deterministic: H = 0
        let probs = vec![1.0, 0.0, 0.0];
        let h = shannon_entropy(&probs);
        assert!((h).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        // Fair coin: H = 1
        let probs = vec![0.5, 0.5];
        let h = shannon_entropy(&probs);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_redundancy_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let r = redundancy(&probs);
        assert!((r).abs() < 1e-10);
    }

    #[test]
    fn test_redundancy_deterministic() {
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let r = redundancy(&probs);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent: I = 0
        let joint = array![[0.25, 0.25], [0.25, 0.25]];
        let mi = mutual_information(&joint);
        assert!((mi).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_dependent() {
        // Perfectly dependent
        let joint = array![[0.5, 0.0], [0.0, 0.5]];
        let mi = mutual_information(&joint);
        assert!((mi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_nonnegative() {
        let joint = array![[0.3, 0.1], [0.2, 0.4]];
        let mi = mutual_information(&joint);
        assert!(mi >= -1e-10);
    }
}
