//! Ornstein-Uhlenbeck process simulation and estimation (Chapter 13).
//!
//! The O-U process is a mean-reverting stochastic process used to model
//! prices that fluctuate around a long-run equilibrium. It is defined by:
//!   dx = theta * (mu - x) * dt + sigma * dW
//!
//! where theta is the mean reversion speed, mu is the long-run mean, and
//! sigma is the volatility.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Simulate an Ornstein-Uhlenbeck process.
///
/// Uses the Euler-Maruyama discretization:
///   x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * Z
///
/// where Z ~ N(0, 1).
///
/// # Arguments
/// * `theta` - Mean reversion speed (must be > 0).
/// * `mu` - Long-run mean.
/// * `sigma` - Volatility (must be > 0).
/// * `x0` - Initial value.
/// * `dt` - Time step (must be > 0).
/// * `n_steps` - Number of simulation steps.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
/// A vector of length `n_steps + 1` containing the simulated path, starting
/// with `x0`.
pub fn simulate_ou(
    theta: f64,
    mu: f64,
    sigma: f64,
    x0: f64,
    dt: f64,
    n_steps: usize,
    seed: u64,
) -> Vec<f64> {
    if theta <= 0.0 || sigma <= 0.0 || dt <= 0.0 || n_steps == 0 {
        return vec![x0];
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");
    let sqrt_dt = dt.sqrt();

    let mut path = Vec::with_capacity(n_steps + 1);
    path.push(x0);

    let mut x = x0;
    for _ in 0..n_steps {
        let dw: f64 = normal.sample(&mut rng);
        x += theta * (mu - x) * dt + sigma * sqrt_dt * dw;
        path.push(x);
    }

    path
}

/// Estimate Ornstein-Uhlenbeck parameters from a time series.
///
/// Uses ordinary least squares on the discretized O-U equation:
///   x_{t+1} - x_t = theta * (mu - x_t) * dt + noise
///
/// Rearranging: dx = a + b * x_t, where
///   b = -theta * dt, a = theta * mu * dt
///
/// From these we recover theta, mu, and sigma.
///
/// # Arguments
/// * `series` - Time series observations (must have length >= 3).
/// * `dt` - Time step between observations.
///
/// # Returns
/// A tuple `(theta, mu, sigma)`. Returns `(0.0, 0.0, 0.0)` if the series is
/// too short or estimation fails.
pub fn estimate_ou_params(series: &[f64], dt: f64) -> (f64, f64, f64) {
    let n = series.len();
    if n < 3 || dt <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    // Compute differences dx_t = x_{t+1} - x_t
    let dx: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let x: Vec<f64> = series[..n - 1].to_vec();
    let m = dx.len();

    // OLS: dx = a + b * x
    let sum_x: f64 = x.iter().sum();
    let sum_dx: f64 = dx.iter().sum();
    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_xdx: f64 = x.iter().zip(dx.iter()).map(|(&xi, &dxi)| xi * dxi).sum();
    let mf = m as f64;

    let denom = mf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return (0.0, 0.0, 0.0);
    }

    let b = (mf * sum_xdx - sum_x * sum_dx) / denom;
    let a = (sum_dx - b * sum_x) / mf;

    // b = -theta * dt
    let theta = -b / dt;
    if theta <= 0.0 {
        // Not mean-reverting
        return (theta.max(0.0), 0.0, 0.0);
    }

    // a = theta * mu * dt
    let mu = a / (theta * dt);

    // Estimate sigma from residuals
    let residuals: Vec<f64> = dx
        .iter()
        .zip(x.iter())
        .map(|(&dxi, &xi)| dxi - a - b * xi)
        .collect();
    let var_residuals: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / mf;
    let sigma = (var_residuals / dt).sqrt();

    (theta, mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_ou_length() {
        let path = simulate_ou(1.0, 0.0, 0.5, 0.0, 0.01, 100, 42);
        assert_eq!(path.len(), 101);
        assert!((path[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_ou_mean_reversion() {
        // With strong mean reversion, path should stay near mu
        let path = simulate_ou(10.0, 5.0, 0.1, 5.0, 0.01, 1000, 42);
        let mean: f64 = path.iter().sum::<f64>() / path.len() as f64;
        assert!(
            (mean - 5.0).abs() < 1.0,
            "mean {} should be close to mu=5.0",
            mean
        );
    }

    #[test]
    fn test_simulate_ou_zero_steps() {
        let path = simulate_ou(1.0, 0.0, 0.5, 3.0, 0.01, 0, 42);
        assert_eq!(path, vec![3.0]);
    }

    #[test]
    fn test_simulate_ou_invalid_params() {
        // theta <= 0
        let path = simulate_ou(0.0, 0.0, 0.5, 1.0, 0.01, 10, 42);
        assert_eq!(path, vec![1.0]);
        // sigma <= 0
        let path = simulate_ou(1.0, 0.0, 0.0, 1.0, 0.01, 10, 42);
        assert_eq!(path, vec![1.0]);
    }

    #[test]
    fn test_estimate_ou_params() {
        // Generate a path with known parameters and estimate them back
        let theta_true = 2.0;
        let mu_true = 1.0;
        let sigma_true = 0.3;
        let dt = 0.01;

        let path = simulate_ou(theta_true, mu_true, sigma_true, mu_true, dt, 10000, 123);
        let (theta_est, mu_est, sigma_est) = estimate_ou_params(&path, dt);

        // Allow generous tolerance for stochastic estimation
        assert!(
            (theta_est - theta_true).abs() < 1.0,
            "theta_est={} vs true={}",
            theta_est,
            theta_true
        );
        assert!(
            (mu_est - mu_true).abs() < 0.5,
            "mu_est={} vs true={}",
            mu_est,
            mu_true
        );
        assert!(
            (sigma_est - sigma_true).abs() < 0.5,
            "sigma_est={} vs true={}",
            sigma_est,
            sigma_true
        );
    }

    #[test]
    fn test_estimate_ou_short_series() {
        let (theta, mu, sigma) = estimate_ou_params(&[1.0, 2.0], 0.01);
        assert_eq!(theta, 0.0);
        assert_eq!(mu, 0.0);
        assert_eq!(sigma, 0.0);
    }

    #[test]
    fn test_estimate_ou_invalid_dt() {
        let (theta, mu, sigma) = estimate_ou_params(&[1.0, 2.0, 3.0], 0.0);
        assert_eq!(theta, 0.0);
        assert_eq!(mu, 0.0);
        assert_eq!(sigma, 0.0);
    }
}
