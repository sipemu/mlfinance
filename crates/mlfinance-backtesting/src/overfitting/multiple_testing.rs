//! Multiple testing corrections.
//!
//! When testing many strategies, the probability of finding a "significant"
//! result by chance increases. These correction methods adjust p-values
//! to control the family-wise error rate.

/// Bonferroni correction for multiple testing.
///
/// Multiplies each p-value by the number of tests. This is the most
/// conservative correction.
///
/// # Arguments
/// * `p_values` - Slice of p-values from individual tests.
///
/// # Returns
/// A vector of adjusted p-values, each clamped to `[0, 1]`.
pub fn bonferroni_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len() as f64;
    if n == 0.0 {
        return Vec::new();
    }
    p_values.iter().map(|&p| (p * n).min(1.0)).collect()
}

/// Holm-Bonferroni (step-down) correction for multiple testing.
///
/// Less conservative than Bonferroni. Sorts p-values in ascending order and
/// applies decreasing multipliers. Controls the family-wise error rate while
/// being more powerful than Bonferroni.
///
/// # Arguments
/// * `p_values` - Slice of p-values from individual tests.
///
/// # Returns
/// A vector of adjusted p-values in the original order, each clamped to `[0, 1]`.
pub fn holm_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    if n == 0 {
        return Vec::new();
    }

    // Create sorted indices
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute adjusted p-values in sorted order (step-down)
    let mut adjusted_sorted = vec![0.0_f64; n];
    let mut running_max = 0.0_f64;

    for (rank, &orig_idx) in indices.iter().enumerate() {
        let multiplier = (n - rank) as f64;
        let adj = (p_values[orig_idx] * multiplier).min(1.0);
        // Enforce monotonicity: adjusted values must not decrease
        running_max = running_max.max(adj);
        adjusted_sorted[rank] = running_max;
    }

    // Map back to original order
    let mut result = vec![0.0_f64; n];
    for (rank, &orig_idx) in indices.iter().enumerate() {
        result[orig_idx] = adjusted_sorted[rank];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonferroni_basic() {
        let p = vec![0.01, 0.04, 0.03, 0.005];
        let adj = bonferroni_correction(&p);
        assert_eq!(adj.len(), 4);
        assert!((adj[0] - 0.04).abs() < 1e-10);
        assert!((adj[1] - 0.16).abs() < 1e-10);
        assert!((adj[2] - 0.12).abs() < 1e-10);
        assert!((adj[3] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_bonferroni_clamp() {
        let p = vec![0.5, 0.3];
        let adj = bonferroni_correction(&p);
        assert!((adj[0] - 1.0).abs() < 1e-10); // 0.5 * 2 = 1.0
        assert!((adj[1] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_bonferroni_empty() {
        let adj = bonferroni_correction(&[]);
        assert!(adj.is_empty());
    }

    #[test]
    fn test_holm_basic() {
        let p = vec![0.01, 0.04, 0.03, 0.005];
        let adj = holm_correction(&p);
        assert_eq!(adj.len(), 4);
        // Sorted: 0.005(idx3), 0.01(idx0), 0.03(idx2), 0.04(idx1)
        // Holm multipliers: 4, 3, 2, 1
        // 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        // Monotonic enforcement: 0.02, 0.03, 0.06, 0.06
        assert!((adj[3] - 0.02).abs() < 1e-10);
        assert!((adj[0] - 0.03).abs() < 1e-10);
        assert!((adj[2] - 0.06).abs() < 1e-10);
        assert!((adj[1] - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_holm_empty() {
        let adj = holm_correction(&[]);
        assert!(adj.is_empty());
    }

    #[test]
    fn test_holm_single() {
        let adj = holm_correction(&[0.05]);
        assert_eq!(adj.len(), 1);
        assert!((adj[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_holm_clamp() {
        let p = vec![0.5, 0.6];
        let adj = holm_correction(&p);
        // Sorted: 0.5, 0.6. Multipliers: 2, 1
        // 0.5*2 = 1.0, 0.6*1 = 0.6 -> monotonic: 1.0, 1.0
        assert!((adj[0] - 1.0).abs() < 1e-10);
        assert!((adj[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_holm_less_conservative_than_bonferroni() {
        let p = vec![0.001, 0.01, 0.05];
        let bonf = bonferroni_correction(&p);
        let holm = holm_correction(&p);
        // Holm should be <= Bonferroni for each p-value (or equal)
        for i in 0..p.len() {
            assert!(
                holm[i] <= bonf[i] + 1e-10,
                "Holm should be <= Bonferroni: holm[{}]={} > bonf[{}]={}",
                i,
                holm[i],
                i,
                bonf[i]
            );
        }
    }
}
