//! Combinatorial Purged Cross-Validation (CPCV) -- Chapter 12.
//!
//! Generates all C(N, k) train/test splits for combinatorial cross-validation.
//! Unlike standard k-fold CV, CPCV produces many more paths, enabling
//! statistical analysis of backtest results.

use itertools::Itertools;

/// Configuration for CPCV.
#[derive(Debug, Clone)]
pub struct CpcvConfig {
    /// Total number of groups (N).
    pub n_groups: usize,
    /// Number of test groups per split (k).
    pub n_test_groups: usize,
}

/// A single CPCV split specifying which groups form the train vs. test set.
#[derive(Debug, Clone)]
pub struct CpcvSplit {
    /// Indices of groups used for training.
    pub train_groups: Vec<usize>,
    /// Indices of groups used for testing.
    pub test_groups: Vec<usize>,
}

impl CpcvConfig {
    /// Generate all C(N, k) CPCV splits.
    ///
    /// Each split assigns `k` groups to the test set and the remaining
    /// `N - k` groups to the training set.
    ///
    /// # Returns
    /// A vector of `CpcvSplit`, one for each combination.
    /// Returns an empty vector if `k == 0`, `k >= N`, or `N == 0`.
    pub fn generate_splits(&self) -> Vec<CpcvSplit> {
        if self.n_groups == 0 || self.n_test_groups == 0 || self.n_test_groups >= self.n_groups {
            return Vec::new();
        }

        let all_groups: Vec<usize> = (0..self.n_groups).collect();

        all_groups
            .iter()
            .copied()
            .combinations(self.n_test_groups)
            .map(|test_groups| {
                let train_groups: Vec<usize> = all_groups
                    .iter()
                    .copied()
                    .filter(|g| !test_groups.contains(g))
                    .collect();
                CpcvSplit {
                    train_groups,
                    test_groups,
                }
            })
            .collect()
    }

    /// Number of backtest paths: phi(N, k).
    ///
    /// The number of independent backtest paths is:
    ///   phi(N, k) = N / k * C(N-1, k-1)
    ///
    /// This counts how many times each group appears in the test set across
    /// all combinations, enabling the construction of complete backtest paths.
    ///
    /// # Returns
    /// The number of backtest paths, or 0 if the configuration is invalid.
    pub fn num_paths(&self) -> usize {
        if self.n_groups == 0 || self.n_test_groups == 0 || self.n_test_groups >= self.n_groups {
            return 0;
        }

        // phi(N, k) = C(N, k) * k / N
        // But more precisely: number of paths = C(N-1, k-1) (each group
        // appears in exactly C(N-1, k-1) test combinations)
        // Total paths = N * C(N-1, k-1) / k

        let n = self.n_groups;
        let k = self.n_test_groups;

        // C(N, k) * k / N = C(N-1, k-1)
        // Number of backtest paths = C(N-1, k-1)
        // Actually: each split contributes k test segments.
        // Total test segments = C(N, k) * k.
        // Each path needs N/k * k = N test segments? No.
        // A path covers all N groups. Each split covers k groups.
        // So a path needs N/k splits (if N divisible by k).
        // Total paths = C(N, k) * k / N

        let total_combinations = binomial(n, k);
        total_combinations * k / n
    }
}

/// Compute binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // optimization: C(n,k) = C(n, n-k)
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_splits_basic() {
        let config = CpcvConfig {
            n_groups: 4,
            n_test_groups: 2,
        };
        let splits = config.generate_splits();
        // C(4, 2) = 6
        assert_eq!(splits.len(), 6);

        for split in &splits {
            assert_eq!(split.test_groups.len(), 2);
            assert_eq!(split.train_groups.len(), 2);
            // No overlap
            for tg in &split.test_groups {
                assert!(!split.train_groups.contains(tg));
            }
        }
    }

    #[test]
    fn test_generate_splits_k1() {
        let config = CpcvConfig {
            n_groups: 5,
            n_test_groups: 1,
        };
        let splits = config.generate_splits();
        // C(5, 1) = 5 (equivalent to 5-fold CV)
        assert_eq!(splits.len(), 5);
    }

    #[test]
    fn test_generate_splits_invalid() {
        // k >= N
        let config = CpcvConfig {
            n_groups: 3,
            n_test_groups: 3,
        };
        assert!(config.generate_splits().is_empty());

        // k = 0
        let config = CpcvConfig {
            n_groups: 3,
            n_test_groups: 0,
        };
        assert!(config.generate_splits().is_empty());

        // N = 0
        let config = CpcvConfig {
            n_groups: 0,
            n_test_groups: 1,
        };
        assert!(config.generate_splits().is_empty());
    }

    #[test]
    fn test_num_paths() {
        let config = CpcvConfig {
            n_groups: 6,
            n_test_groups: 2,
        };
        // C(6,2) = 15, paths = 15 * 2 / 6 = 5
        assert_eq!(config.num_paths(), 5);
    }

    #[test]
    fn test_num_paths_k1() {
        let config = CpcvConfig {
            n_groups: 5,
            n_test_groups: 1,
        };
        // C(5,1) = 5, paths = 5 * 1 / 5 = 1 (standard k-fold)
        assert_eq!(config.num_paths(), 1);
    }

    #[test]
    fn test_num_paths_invalid() {
        let config = CpcvConfig {
            n_groups: 3,
            n_test_groups: 3,
        };
        assert_eq!(config.num_paths(), 0);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(6, 2), 15);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(3, 5), 0);
    }

    #[test]
    fn test_all_groups_covered() {
        let config = CpcvConfig {
            n_groups: 5,
            n_test_groups: 2,
        };
        let splits = config.generate_splits();
        // Every group should appear as a test group in some split
        for g in 0..5 {
            let appears = splits.iter().any(|s| s.test_groups.contains(&g));
            assert!(appears, "group {} should appear in test set", g);
        }
    }
}
