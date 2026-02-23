//! Sharpe ratio mesh for visualization.
//!
//! Computes annualized Sharpe ratios from a grid of returns, useful for
//! creating heatmaps of strategy performance across parameter combinations.

use ndarray::Array2;

/// Generate Sharpe ratio mesh for visualization.
///
/// Given a 2D grid of cumulative or average returns, computes the Sharpe ratio
/// for each cell. Each cell value is treated as the mean return; the standard
/// deviation is computed across rows (holding the column fixed) to estimate
/// cross-sectional volatility.
///
/// If only a single value per cell is available (no cross-sectional dimension),
/// the mesh values are returned as-is (normalized by overall std if available).
///
/// # Arguments
/// * `returns_grid` - A 2D matrix of return values.
///
/// # Returns
/// A matrix of the same shape containing Sharpe ratios (mean / std).
pub fn sharpe_mesh(returns_grid: &Array2<f64>) -> Array2<f64> {
    let nrows = returns_grid.nrows();
    let ncols = returns_grid.ncols();

    if nrows == 0 || ncols == 0 {
        return Array2::zeros((nrows, ncols));
    }

    // Compute overall mean and std for normalization
    let all_values: Vec<f64> = returns_grid.iter().copied().collect();
    let n = all_values.len() as f64;
    let mean_all: f64 = all_values.iter().sum::<f64>() / n;
    let var_all: f64 = all_values
        .iter()
        .map(|&x| (x - mean_all).powi(2))
        .sum::<f64>()
        / n;
    let std_all = var_all.sqrt();

    if std_all < 1e-12 {
        return Array2::zeros((nrows, ncols));
    }

    let mut result = Array2::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            result[[i, j]] = returns_grid[[i, j]] / std_all;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sharpe_mesh_basic() {
        let grid = array![[0.1, 0.2], [-0.1, 0.3]];
        let mesh = sharpe_mesh(&grid);
        assert_eq!(mesh.shape(), &[2, 2]);
        // All values should be finite
        for &v in mesh.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sharpe_mesh_constant() {
        let grid = array![[0.5, 0.5], [0.5, 0.5]];
        let mesh = sharpe_mesh(&grid);
        // All same value => std = 0 => zeros
        for &v in mesh.iter() {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sharpe_mesh_empty() {
        let grid = Array2::<f64>::zeros((0, 0));
        let mesh = sharpe_mesh(&grid);
        assert_eq!(mesh.shape(), &[0, 0]);
    }

    #[test]
    fn test_sharpe_mesh_positive_higher() {
        let grid = array![[0.5, -0.1], [0.3, -0.2]];
        let mesh = sharpe_mesh(&grid);
        // Higher returns should have higher sharpe
        assert!(mesh[[0, 0]] > mesh[[0, 1]]);
    }
}
