//! Combinatorially Symmetric Cross-Validation (CSCV).
//!
//! Implements the CSCV framework for detecting backtest overfitting.
//! Partitions the performance matrix into groups and evaluates strategy
//! selection stability across all symmetric train/test splits.

use itertools::Itertools;
use ndarray::Array2;

/// Result of CSCV analysis.
#[derive(Debug, Clone)]
pub struct CscvResult {
    /// Probability of Backtest Overfitting.
    pub pbo: f64,
    /// Distribution of rank logits for the best IS strategy evaluated OOS.
    pub rank_logits: Vec<f64>,
}

/// Compute strategy performance for given row indices.
fn strategy_perf(returns: &Array2<f64>, rows: &[usize], n_strategies: usize) -> Vec<f64> {
    (0..n_strategies)
        .map(|s| rows.iter().map(|&r| returns[[r, s]]).sum::<f64>())
        .collect()
}

/// Evaluate a single CSCV split, returning (rank_logit, is_overfit) or None if skipped.
fn evaluate_cscv_split(
    returns_matrix: &Array2<f64>,
    group_indices: &[usize],
    is_combo: &[usize],
    groups: &[Vec<usize>],
    n_strategies: usize,
) -> Option<(f64, bool)> {
    let oos_combo: Vec<usize> = group_indices
        .iter()
        .copied()
        .filter(|i| !is_combo.contains(i))
        .collect();

    let is_rows: Vec<usize> = is_combo
        .iter()
        .flat_map(|&g| groups[g].iter().copied())
        .collect();
    let oos_rows: Vec<usize> = oos_combo
        .iter()
        .flat_map(|&g| groups[g].iter().copied())
        .collect();

    if is_rows.is_empty() || oos_rows.is_empty() {
        return None;
    }

    let is_perf = strategy_perf(returns_matrix, &is_rows, n_strategies);
    let best_is = is_perf
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let oos_perf = strategy_perf(returns_matrix, &oos_rows, n_strategies);
    let best_oos_val = oos_perf[best_is];
    let rank = oos_perf.iter().filter(|&&p| p > best_oos_val).count() + 1;

    let n = n_strategies as f64;
    let r = rank as f64;
    let denom = n + 1.0 - r;
    let logit = if denom > 0.0 {
        (r / denom).ln()
    } else {
        f64::INFINITY
    };

    Some((logit, rank > n_strategies / 2))
}

/// Perform Combinatorially Symmetric Cross-Validation analysis.
///
/// Splits `num_groups` into two equal halves across all C(S, S/2) combinations.
/// For each split, identifies the best in-sample strategy and computes its
/// out-of-sample rank logit (log of rank / (N - rank)).
///
/// # Arguments
/// * `returns_matrix` - An `(T x N)` matrix where rows are time periods and
///   columns are strategies.
/// * `num_groups` - Number of groups to partition into (must be even and >= 2).
///
/// # Returns
/// A `CscvResult` containing the PBO and the distribution of rank logits.
pub fn cscv(returns_matrix: &Array2<f64>, num_groups: usize) -> CscvResult {
    let n_rows = returns_matrix.nrows();
    let n_strategies = returns_matrix.ncols();

    if n_strategies == 0
        || n_rows == 0
        || num_groups < 2
        || num_groups % 2 != 0
        || num_groups > n_rows
    {
        return CscvResult {
            pbo: 0.0,
            rank_logits: Vec::new(),
        };
    }

    let group_size = n_rows / num_groups;
    let groups: Vec<Vec<usize>> = (0..num_groups)
        .map(|g| {
            let start = g * group_size;
            let end = if g == num_groups - 1 {
                n_rows
            } else {
                (g + 1) * group_size
            };
            (start..end).collect()
        })
        .collect();

    let half = num_groups / 2;
    let group_indices: Vec<usize> = (0..num_groups).collect();
    let combinations: Vec<Vec<usize>> = group_indices.iter().copied().combinations(half).collect();

    let mut rank_logits = Vec::with_capacity(combinations.len());
    let mut overfit_count = 0usize;

    for is_combo in &combinations {
        if let Some((logit, is_overfit)) = evaluate_cscv_split(
            returns_matrix,
            &group_indices,
            is_combo,
            &groups,
            n_strategies,
        ) {
            rank_logits.push(logit);
            if is_overfit {
                overfit_count += 1;
            }
        }
    }

    let total = rank_logits.len().max(1);
    let pbo = overfit_count as f64 / total as f64;

    CscvResult { pbo, rank_logits }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cscv_basic() {
        let returns = array![
            [0.01, 0.02, -0.01],
            [0.02, -0.01, 0.01],
            [-0.01, 0.01, 0.02],
            [0.01, 0.02, -0.01],
            [0.02, -0.01, 0.01],
            [-0.01, 0.01, 0.02],
        ];
        let result = cscv(&returns, 2);
        assert!(result.pbo >= 0.0 && result.pbo <= 1.0);
        assert!(!result.rank_logits.is_empty());
    }

    #[test]
    fn test_cscv_empty() {
        let returns = Array2::<f64>::zeros((0, 0));
        let result = cscv(&returns, 2);
        assert_eq!(result.pbo, 0.0);
        assert!(result.rank_logits.is_empty());
    }

    #[test]
    fn test_cscv_odd_groups() {
        let returns = Array2::<f64>::zeros((6, 3));
        let result = cscv(&returns, 3);
        assert_eq!(result.pbo, 0.0);
    }

    #[test]
    fn test_cscv_rank_logits_finite() {
        let returns = array![
            [0.05, -0.02, 0.03, 0.01],
            [-0.01, 0.04, -0.02, 0.03],
            [0.03, 0.01, 0.04, -0.01],
            [-0.02, 0.03, 0.01, 0.04],
        ];
        let result = cscv(&returns, 2);
        for logit in &result.rank_logits {
            assert!(logit.is_finite() || *logit == f64::INFINITY);
        }
    }
}
