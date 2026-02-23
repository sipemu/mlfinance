use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use super::cla::cla_min_variance;
use super::hrp::hrp::hrp_weights;
use super::ivp::inverse_variance_weights;

/// Results of Monte Carlo comparison of HRP, CLA, and IVP allocation methods.
pub struct AllocationComparison {
    /// Average out-of-sample Sharpe ratio for HRP.
    pub hrp_sharpe: f64,
    /// Average out-of-sample Sharpe ratio for CLA.
    pub cla_sharpe: f64,
    /// Average out-of-sample Sharpe ratio for IVP.
    pub ivp_sharpe: f64,
    /// Average out-of-sample variance for HRP.
    pub hrp_variance: f64,
    /// Average out-of-sample variance for CLA.
    pub cla_variance: f64,
    /// Average out-of-sample variance for IVP.
    pub ivp_variance: f64,
}

/// Run a single Monte Carlo simulation: add noise, fit weights, evaluate OOS.
/// Returns (hrp_sharpe, cla_sharpe, ivp_sharpe, hrp_var, cla_var, ivp_var) or None if any method fails.
fn run_single_simulation(
    returns: &Array2<f64>,
    split: usize,
    normal: &Normal<f64>,
    rng: &mut rand::rngs::StdRng,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let mut noisy_returns = returns.clone();
    for row in noisy_returns.rows_mut() {
        for val in row {
            *val += normal.sample(rng);
        }
    }

    let in_sample = noisy_returns.slice(ndarray::s![..split, ..]).to_owned();
    let out_of_sample = noisy_returns.slice(ndarray::s![split.., ..]).to_owned();

    let cov_in = mlfinance_core::stats::covariance_matrix(&in_sample).ok()?;
    let hrp_w = hrp_weights(&in_sample).ok()?;
    let cla_w = cla_min_variance(&cov_in).ok()?;
    let ivp_w = inverse_variance_weights(&cov_in);

    let (hrp_s, hrp_v) = portfolio_metrics(&out_of_sample, &hrp_w);
    let (cla_s, cla_v) = portfolio_metrics(&out_of_sample, &cla_w);
    let (ivp_s, ivp_v) = portfolio_metrics(&out_of_sample, &ivp_w);

    Some((hrp_s, cla_s, ivp_s, hrp_v, cla_v, ivp_v))
}

/// Validate allocation comparison inputs.
fn validate_allocation_inputs(n_rows: usize, n_cols: usize) -> mlfinance_core::error::Result<()> {
    if n_rows < 4 {
        return Err(mlfinance_core::MlFinanceError::InsufficientData {
            expected: 4,
            actual: n_rows,
        });
    }
    if n_cols < 2 {
        return Err(mlfinance_core::MlFinanceError::InsufficientData {
            expected: 2,
            actual: n_cols,
        });
    }
    Ok(())
}

/// Average a vector of f64.
fn vec_mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Run Monte Carlo comparison of allocation methods.
///
/// For each simulation:
/// 1. Add Gaussian noise to returns
/// 2. Split returns into in-sample and out-of-sample halves
/// 3. Fit weights using in-sample data (HRP, CLA, IVP)
/// 4. Evaluate portfolio performance on out-of-sample data
///
/// # Arguments
///
/// * `returns` - Asset returns matrix (n_observations x n_assets), requires >= 4 rows and >= 2 columns.
/// * `n_simulations` - Number of Monte Carlo repetitions.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
///
/// An [`AllocationComparison`] with average Sharpe ratios and variances for each method.
///
/// # Errors
///
/// Returns an error if there is insufficient data (< 4 rows or < 2 columns)
/// or if all simulations fail.
pub fn compare_allocations(
    returns: &Array2<f64>,
    n_simulations: usize,
    seed: u64,
) -> mlfinance_core::error::Result<AllocationComparison> {
    validate_allocation_inputs(returns.nrows(), returns.ncols())?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal =
        Normal::new(0.0, 0.01).map_err(|e| mlfinance_core::MlFinanceError::ComputationError {
            msg: format!("failed to create normal distribution: {}", e),
        })?;

    let split = returns.nrows() / 2;
    let results: Vec<_> = (0..n_simulations)
        .filter_map(|_| run_single_simulation(returns, split, &normal, &mut rng))
        .collect();

    if results.is_empty() {
        return Err(mlfinance_core::MlFinanceError::ComputationError {
            msg: "all simulations failed".into(),
        });
    }

    let hrp_s: Vec<f64> = results.iter().map(|r| r.0).collect();
    let cla_s: Vec<f64> = results.iter().map(|r| r.1).collect();
    let ivp_s: Vec<f64> = results.iter().map(|r| r.2).collect();
    let hrp_v: Vec<f64> = results.iter().map(|r| r.3).collect();
    let cla_v: Vec<f64> = results.iter().map(|r| r.4).collect();
    let ivp_v: Vec<f64> = results.iter().map(|r| r.5).collect();

    Ok(AllocationComparison {
        hrp_sharpe: vec_mean(&hrp_s),
        cla_sharpe: vec_mean(&cla_s),
        ivp_sharpe: vec_mean(&ivp_s),
        hrp_variance: vec_mean(&hrp_v),
        cla_variance: vec_mean(&cla_v),
        ivp_variance: vec_mean(&ivp_v),
    })
}

/// Compute Sharpe ratio and variance for a portfolio.
fn portfolio_metrics(returns: &Array2<f64>, weights: &ndarray::Array1<f64>) -> (f64, f64) {
    let n = returns.nrows();
    if n == 0 {
        return (0.0, 0.0);
    }

    // Compute portfolio returns: r_p = returns * weights
    let port_returns: Vec<f64> = (0..n)
        .map(|i| {
            returns
                .row(i)
                .iter()
                .zip(weights.iter())
                .map(|(&r, &w)| r * w)
                .sum::<f64>()
        })
        .collect();

    let mean_ret = port_returns.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        port_returns
            .iter()
            .map(|&r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    let std = var.sqrt();

    let sharpe = if std > 1e-15 { mean_ret / std } else { 0.0 };

    (sharpe, var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compare_allocations() {
        let returns = array![
            [0.01, 0.02, -0.01],
            [0.02, 0.01, 0.005],
            [-0.005, 0.015, 0.01],
            [0.01, -0.01, 0.02],
            [0.005, 0.008, -0.005],
            [0.003, 0.012, 0.008],
            [-0.002, 0.005, 0.003],
            [0.008, -0.003, 0.015],
        ];
        let result = compare_allocations(&returns, 5, 42).unwrap();
        // Just verify all fields are finite
        assert!(result.hrp_sharpe.is_finite());
        assert!(result.cla_sharpe.is_finite());
        assert!(result.ivp_sharpe.is_finite());
        assert!(result.hrp_variance.is_finite());
        assert!(result.cla_variance.is_finite());
        assert!(result.ivp_variance.is_finite());
    }

    #[test]
    fn test_compare_allocations_insufficient() {
        let returns = array![[0.01, 0.02], [0.02, 0.01]];
        let result = compare_allocations(&returns, 5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_portfolio_metrics() {
        let returns = array![[0.01, 0.02], [0.02, 0.01], [-0.01, 0.015], [0.005, -0.005]];
        let weights = ndarray::array![0.5, 0.5];
        let (sharpe, var) = portfolio_metrics(&returns, &weights);
        assert!(sharpe.is_finite());
        assert!(var >= 0.0);
    }
}
