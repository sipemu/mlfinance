use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Hasbrouck's lambda via Gibbs sampler (simplified).
///
/// Estimates trade impact from returns and trade signs using a
/// simplified Bayesian estimation via Gibbs sampling.
///
/// Model: r_t = lambda * q_t + epsilon_t
/// where q_t are trade signs (+1/-1) and epsilon_t ~ N(0, sigma^2)
///
/// The Gibbs sampler alternates between sampling lambda and sigma^2
/// from their conditional posterior distributions.
///
/// # Arguments
///
/// * `returns` - Asset returns for each period.
/// * `trade_signs` - Trade sign indicators (+1.0 for buy, -1.0 for sell).
/// * `n_iterations` - Number of Gibbs sampling iterations (first 25% are burn-in).
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
///
/// Posterior mean of lambda (trade impact coefficient). Returns 0.0 if
/// fewer than 2 data points or `n_iterations == 0`.
pub fn hasbrouck_lambda(
    returns: &[f64],
    trade_signs: &[f64],
    n_iterations: usize,
    seed: u64,
) -> f64 {
    let n = returns.len().min(trade_signs.len());
    if n < 2 || n_iterations == 0 {
        return 0.0;
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Initialize
    let mut lambda = 0.0;
    let mut sigma2 = returns.iter().map(|&r| r * r).sum::<f64>() / n as f64;
    if sigma2 < 1e-15 {
        sigma2 = 1e-10;
    }

    let burn_in = n_iterations / 4;
    let mut lambda_samples = Vec::with_capacity(n_iterations - burn_in);

    // Precompute: sum(q_t^2) and sum(q_t * r_t)
    let sum_q2: f64 = trade_signs[..n].iter().map(|&q| q * q).sum();
    let sum_qr: f64 = trade_signs[..n]
        .iter()
        .zip(returns[..n].iter())
        .map(|(&q, &r)| q * r)
        .sum();

    for iter in 0..n_iterations {
        // Sample lambda | sigma^2, data
        // Posterior: lambda ~ N(mu_lambda, var_lambda)
        // where mu_lambda = sum(q*r) / sum(q^2), var_lambda = sigma^2 / sum(q^2)
        if sum_q2 > 1e-15 {
            let var_lambda = sigma2 / sum_q2;
            let mu_lambda = sum_qr / sum_q2;

            let normal = Normal::new(mu_lambda, var_lambda.sqrt().max(1e-15)).unwrap();
            lambda = normal.sample(&mut rng);
        }

        // Sample sigma^2 | lambda, data
        // Compute sum of squared residuals
        let ssr: f64 = returns[..n]
            .iter()
            .zip(trade_signs[..n].iter())
            .map(|(&r, &q)| {
                let resid = r - lambda * q;
                resid * resid
            })
            .sum();

        // sigma^2 ~ InvGamma, approximate with point estimate for simplicity
        sigma2 = (ssr / n as f64).max(1e-15);

        // Add small noise to sigma2 for mixing
        let noise = Normal::new(0.0, sigma2 * 0.01).unwrap();
        sigma2 = (sigma2 + noise.sample(&mut rng)).max(1e-15);

        if iter >= burn_in {
            lambda_samples.push(lambda);
        }
    }

    if lambda_samples.is_empty() {
        return 0.0;
    }

    // Return posterior mean of lambda
    lambda_samples.iter().sum::<f64>() / lambda_samples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hasbrouck_positive_impact() {
        // Returns positively correlated with trade signs
        let trade_signs = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let returns: Vec<f64> = trade_signs.iter().map(|&q| 0.001 * q + 0.0001).collect();
        let lambda = hasbrouck_lambda(&returns, &trade_signs, 1000, 42);
        // Lambda should be close to 0.001
        assert!(lambda > 0.0);
    }

    #[test]
    fn test_hasbrouck_zero_impact() {
        let trade_signs = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let returns = vec![0.0; 5];
        let lambda = hasbrouck_lambda(&returns, &trade_signs, 500, 42);
        // Should be close to 0
        assert!(lambda.abs() < 0.01);
    }

    #[test]
    fn test_hasbrouck_short() {
        let lambda = hasbrouck_lambda(&[0.01], &[1.0], 100, 42);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_hasbrouck_zero_iterations() {
        let lambda = hasbrouck_lambda(&[0.01, -0.01], &[1.0, -1.0], 0, 42);
        assert_eq!(lambda, 0.0);
    }
}
