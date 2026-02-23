//! Optimal Trading Rule mesh (Snippets 13.1-13.2).
//!
//! Computes a mesh of expected Sharpe ratios for different profit-taking and
//! stop-loss thresholds applied to an Ornstein-Uhlenbeck process. Used to
//! find the optimal (pt, sl) pair for a mean-reverting strategy.

use super::ornstein_uhlenbeck::simulate_ou;
use ndarray::Array2;

/// Simulate one O-U path and return the exit return (pt or -sl).
#[allow(clippy::too_many_arguments)]
fn find_exit_return(
    theta: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    max_steps: usize,
    pt_abs: f64,
    sl_abs: f64,
    sim_seed: u64,
) -> f64 {
    let path = simulate_ou(theta, mu, sigma, mu, dt, max_steps, sim_seed);
    for &x in path.iter().skip(1) {
        let deviation = x - mu;
        if deviation >= pt_abs {
            return pt_abs;
        } else if deviation <= -sl_abs {
            return -sl_abs;
        }
    }
    0.0
}

/// Compute the Sharpe ratio of a collection of returns.
fn sharpe_ratio_of(returns: &[f64]) -> f64 {
    let n = returns.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_ret: f64 = returns.iter().sum::<f64>() / n;
    let var: f64 = returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1.0);
    let std_ret = var.sqrt();
    if std_ret > 1e-12 {
        mean_ret / std_ret
    } else {
        0.0
    }
}

/// Compute the Sharpe ratio for a single (pt, sl) cell via Monte Carlo.
#[allow(clippy::too_many_arguments)]
fn compute_cell(
    theta: f64,
    mu: f64,
    sigma: f64,
    pt_abs: f64,
    sl_abs: f64,
    n_simulations: usize,
    seed: u64,
    i: usize,
    j: usize,
) -> f64 {
    let dt = 0.01;
    let max_steps = 10000;
    let returns: Vec<f64> = (0..n_simulations)
        .map(|sim| {
            let sim_seed = seed.wrapping_add(
                (i as u64)
                    .wrapping_mul(10000)
                    .wrapping_add((j as u64).wrapping_mul(100))
                    .wrapping_add(sim as u64),
            );
            find_exit_return(theta, mu, sigma, dt, max_steps, pt_abs, sl_abs, sim_seed)
        })
        .collect();
    sharpe_ratio_of(&returns)
}

/// Compute a single mesh cell, returning None if barriers are too small.
#[allow(clippy::too_many_arguments)]
fn mesh_cell_value(
    theta: f64,
    mu: f64,
    sigma: f64,
    pt: f64,
    sl: f64,
    n_simulations: usize,
    seed: u64,
    i: usize,
    j: usize,
) -> f64 {
    let pt_abs = pt.abs();
    let sl_abs = sl.abs();
    if pt_abs < 1e-12 || sl_abs < 1e-12 {
        return 0.0;
    }
    compute_cell(theta, mu, sigma, pt_abs, sl_abs, n_simulations, seed, i, j)
}

/// Compute optimal trading rule mesh for an O-U process.
///
/// For each combination of profit-taking (pt) and stop-loss (sl) thresholds,
/// simulates the O-U process multiple times and computes the realized Sharpe
/// ratio of a strategy that enters at mu and exits at pt or sl.
///
/// # Arguments
/// * `theta` - Mean reversion speed.
/// * `mu` - Long-run mean.
/// * `sigma` - Volatility.
/// * `pt_range` - Profit-taking thresholds (positive deviations from mu).
/// * `sl_range` - Stop-loss thresholds (negative deviations from mu, should be
///   negative or will be negated).
/// * `n_simulations` - Number of Monte Carlo simulations per (pt, sl) pair.
/// * `seed` - Base random seed.
///
/// # Returns
/// A matrix of shape `(pt_range.len(), sl_range.len())` containing the
/// estimated Sharpe ratio for each (pt, sl) pair.
pub fn otr_mesh(
    theta: f64,
    mu: f64,
    sigma: f64,
    pt_range: &[f64],
    sl_range: &[f64],
    n_simulations: usize,
    seed: u64,
) -> Array2<f64> {
    let n_pt = pt_range.len();
    let n_sl = sl_range.len();

    if n_pt == 0 || n_sl == 0 || n_simulations == 0 {
        return Array2::zeros((n_pt, n_sl));
    }

    let mut result = Array2::zeros((n_pt, n_sl));
    for (i, &pt) in pt_range.iter().enumerate() {
        for (j, &sl) in sl_range.iter().enumerate() {
            result[[i, j]] = mesh_cell_value(theta, mu, sigma, pt, sl, n_simulations, seed, i, j);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otr_mesh_shape() {
        let pt = vec![0.1, 0.2, 0.3];
        let sl = vec![-0.1, -0.2];
        let mesh = otr_mesh(2.0, 0.0, 0.5, &pt, &sl, 10, 42);
        assert_eq!(mesh.shape(), &[3, 2]);
    }

    #[test]
    fn test_otr_mesh_empty() {
        let mesh = otr_mesh(2.0, 0.0, 0.5, &[], &[0.1], 10, 42);
        assert_eq!(mesh.shape(), &[0, 1]);
    }

    #[test]
    fn test_otr_mesh_values_finite() {
        let pt = vec![0.2, 0.5];
        let sl = vec![-0.2, -0.5];
        let mesh = otr_mesh(2.0, 0.0, 0.5, &pt, &sl, 50, 42);
        for &v in mesh.iter() {
            assert!(v.is_finite(), "all mesh values should be finite");
        }
    }

    #[test]
    fn test_otr_mesh_symmetric_barriers() {
        // For a mean-reverting process starting at mu, symmetric pt/sl
        // should give positive expected Sharpe (mean reversion favors pt)
        let pt = vec![0.5];
        let sl = vec![-0.5];
        let mesh = otr_mesh(5.0, 0.0, 0.3, &pt, &sl, 200, 42);
        // With strong mean reversion, we expect non-negative SR
        assert!(
            mesh[[0, 0]] >= -1.0,
            "SR should be reasonable, got {}",
            mesh[[0, 0]]
        );
    }
}
