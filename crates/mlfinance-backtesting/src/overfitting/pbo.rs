//! Probability of Backtest Overfitting (PBO).
//!
//! Implements the PBO framework from Bailey et al. (2015). Given a matrix of
//! strategy returns across time periods, estimates the probability that the
//! best in-sample strategy underperforms out-of-sample.

use itertools::Itertools;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Compute strategy performance as sum of returns for given row indices.
fn strategy_performance(returns: &Array2<f64>, rows: &[usize], n_strategies: usize) -> Vec<f64> {
    (0..n_strategies)
        .map(|s| rows.iter().map(|&r| returns[[r, s]]).sum::<f64>())
        .collect()
}

/// Find the index of the maximum value in a slice.
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Evaluate a single IS/OOS split and return whether it's overfit (Some(true/false)) or skip (None).
fn evaluate_split(
    returns_matrix: &Array2<f64>,
    partition_indices: &[usize],
    is_combo: &[usize],
    groups: &[Vec<usize>],
    n_strategies: usize,
) -> Option<bool> {
    let oos_combo: Vec<usize> = partition_indices
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

    let is_perf = strategy_performance(returns_matrix, &is_rows, n_strategies);
    let best_is = argmax(&is_perf);

    let oos_perf = strategy_performance(returns_matrix, &oos_rows, n_strategies);
    let best_oos = oos_perf[best_is];
    let rank = oos_perf.iter().filter(|&&p| p <= best_oos).count();

    Some(rank <= n_strategies / 2)
}

/// Compute Probability of Backtest Overfitting from a matrix of strategy returns.
///
/// The algorithm:
/// 1. Partition time periods into `num_partitions` groups.
/// 2. For each combinatorial split into in-sample (IS) and out-of-sample (OOS)
///    halves, identify the best IS strategy.
/// 3. Compute the rank of that strategy OOS.
/// 4. PBO = fraction of splits where the best IS strategy ranks below median OOS.
///
/// # Arguments
/// * `returns_matrix` - An `(T x N)` matrix where rows are time periods and
///   columns are strategies.
/// * `num_partitions` - Number of groups to partition the time periods into
///   (must be even and >= 2).
/// * `seed` - Random seed for reproducibility of partition ordering.
///
/// # Returns
/// PBO in `[0, 1]` -- the probability that the selected strategy is overfit.
pub fn probability_of_backtest_overfitting(
    returns_matrix: &Array2<f64>,
    num_partitions: usize,
    seed: u64,
) -> f64 {
    let n_rows = returns_matrix.nrows();
    let n_strategies = returns_matrix.ncols();

    if n_strategies == 0 || n_rows == 0 || num_partitions < 2 || num_partitions % 2 != 0 {
        return 0.0;
    }

    // Cannot partition into more groups than rows
    if num_partitions > n_rows {
        return 0.0;
    }

    // Create partition indices
    let mut rng = StdRng::seed_from_u64(seed);
    let mut row_indices: Vec<usize> = (0..n_rows).collect();
    row_indices.shuffle(&mut rng);

    // Split rows into groups
    let group_size = n_rows / num_partitions;
    let groups: Vec<Vec<usize>> = (0..num_partitions)
        .map(|g| {
            let start = g * group_size;
            let end = if g == num_partitions - 1 {
                n_rows
            } else {
                (g + 1) * group_size
            };
            row_indices[start..end].to_vec()
        })
        .collect();

    let half = num_partitions / 2;
    let partition_indices: Vec<usize> = (0..num_partitions).collect();

    // Generate all C(S, S/2) combinations for IS
    let combinations: Vec<Vec<usize>> = partition_indices
        .iter()
        .copied()
        .combinations(half)
        .collect();

    let total_combinations = combinations.len();
    if total_combinations == 0 {
        return 0.0;
    }

    let overfit_count: usize = combinations
        .iter()
        .filter_map(|is_combo| {
            evaluate_split(
                returns_matrix,
                &partition_indices,
                is_combo,
                &groups,
                n_strategies,
            )
        })
        .filter(|&is_overfit| is_overfit)
        .count();

    overfit_count as f64 / total_combinations as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pbo_basic() {
        // 10 time periods, 3 strategies
        let returns = array![
            [0.01, 0.02, -0.01],
            [0.02, -0.01, 0.01],
            [-0.01, 0.01, 0.02],
            [0.01, 0.02, -0.01],
            [0.02, -0.01, 0.01],
            [-0.01, 0.01, 0.02],
            [0.01, 0.02, -0.01],
            [0.02, -0.01, 0.01],
            [-0.01, 0.01, 0.02],
            [0.01, 0.02, -0.01],
        ];
        let pbo = probability_of_backtest_overfitting(&returns, 2, 42);
        assert!(pbo >= 0.0 && pbo <= 1.0);
    }

    #[test]
    fn test_pbo_empty_matrix() {
        let returns = Array2::<f64>::zeros((0, 0));
        let pbo = probability_of_backtest_overfitting(&returns, 2, 42);
        assert_eq!(pbo, 0.0);
    }

    #[test]
    fn test_pbo_odd_partitions() {
        let returns = Array2::<f64>::zeros((10, 3));
        // Odd partitions should return 0
        let pbo = probability_of_backtest_overfitting(&returns, 3, 42);
        assert_eq!(pbo, 0.0);
    }

    #[test]
    fn test_pbo_too_many_partitions() {
        let returns = Array2::<f64>::zeros((4, 3));
        let pbo = probability_of_backtest_overfitting(&returns, 6, 42);
        assert_eq!(pbo, 0.0);
    }

    #[test]
    fn test_pbo_single_strategy() {
        let returns = Array2::<f64>::ones((10, 1));
        let pbo = probability_of_backtest_overfitting(&returns, 2, 42);
        // With one strategy, it always ranks #1, so not overfit
        assert!(pbo >= 0.0 && pbo <= 1.0);
    }
}
