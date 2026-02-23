/// Brown-Durbin-Evans CUSUM test for structural change.
///
/// Computes the cumulative sum of recursive residuals (standardized),
/// scaled by 1/sqrt(n) for proper normalization.
///
/// # Arguments
///
/// * `residuals` - Regression residuals to test for structural change.
///
/// # Returns
///
/// A tuple `(cusum_values, critical_boundary)` where `cusum_values` is the
/// normalized cumulative sum and `critical_boundary` is the 5% critical value
/// (1.358). Returns `(vec![], 0.0)` if fewer than 2 residuals.
pub fn brown_durbin_evans(residuals: &[f64]) -> (Vec<f64>, f64) {
    let n = residuals.len();
    if n < 2 {
        return (vec![], 0.0);
    }

    // Compute mean and standard deviation of residuals
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let var = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = var.sqrt();

    if std < 1e-15 {
        return (vec![0.0; n], 0.0);
    }

    // Standardized residuals
    let standardized: Vec<f64> = residuals.iter().map(|&r| (r - mean) / std).collect();

    // Cumulative sum of standardized residuals
    let mut cusum = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    for &s in &standardized {
        cumulative += s;
        cusum.push(cumulative);
    }

    // Scale by 1/sqrt(n) for proper normalization
    let scale = 1.0 / (n as f64).sqrt();
    for c in cusum.iter_mut() {
        *c *= scale;
    }

    // 5% critical value boundary: approximately 1.358 for BDE CUSUM
    // The boundary is a function of sample fraction, but we return max boundary
    let critical = 1.358;

    (cusum, critical)
}

/// Chu-Stinchcombe-White CUSUM test.
///
/// Sequential test for structural change in a series of log prices.
/// Computes S_n = sum_{i=1}^{n} (y_i - y_{i-1}) / sigma.
/// Values exceeding 1.0 indicate rejection of the no-change null hypothesis.
///
/// # Arguments
///
/// * `log_prices` - Log price series (requires at least 3 values).
/// * `critical_value` - Critical value for the boundary function (e.g. 1.96 for 5%).
///
/// # Returns
///
/// Vector of CUSUM statistics normalized by the boundary function.
/// Returns an empty vector if the series is too short.
pub fn chu_stinchcombe_white(log_prices: &[f64], critical_value: f64) -> Vec<f64> {
    let n = log_prices.len();
    if n < 3 {
        return vec![];
    }

    // Compute log returns
    let returns: Vec<f64> = log_prices.windows(2).map(|w| w[1] - w[0]).collect();

    // Estimate volatility from the first half
    let half = returns.len() / 2;
    if half < 2 {
        return vec![];
    }

    let mean_ret = returns[..half].iter().sum::<f64>() / half as f64;
    let var = returns[..half]
        .iter()
        .map(|&r| (r - mean_ret).powi(2))
        .sum::<f64>()
        / (half - 1) as f64;
    let sigma = var.sqrt();

    if sigma < 1e-15 {
        return vec![0.0; returns.len()];
    }

    // Compute CUSUM statistics
    let mut cusum_stats = Vec::with_capacity(returns.len());
    let mut s_n = 0.0;

    for (i, &ret) in returns.iter().enumerate() {
        s_n += (ret - mean_ret) / sigma;
        // Boundary function: c * sqrt(n) * (1 + 2*(t/n))
        // where t is the current observation index
        let t_frac = (i + 1) as f64 / returns.len() as f64;
        let boundary = critical_value * ((i + 1) as f64).sqrt() * (1.0 + 2.0 * t_frac);
        // Return the standardized CUSUM relative to boundary
        let stat = s_n.abs() / boundary;
        cusum_stats.push(stat);
    }

    cusum_stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bde_constant_residuals() {
        let residuals = vec![0.0; 20];
        let (cusum, _critical) = brown_durbin_evans(&residuals);
        assert_eq!(cusum.len(), 20);
        for &c in &cusum {
            assert!((c).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bde_with_change() {
        // Residuals with a structural break
        let mut residuals = vec![0.1; 20];
        residuals.extend(vec![-0.5; 20]);
        let (cusum, critical) = brown_durbin_evans(&residuals);
        assert_eq!(cusum.len(), 40);
        assert!(critical > 0.0);
    }

    #[test]
    fn test_bde_short_series() {
        let (cusum, _) = brown_durbin_evans(&[1.0]);
        assert!(cusum.is_empty());
    }

    #[test]
    fn test_csw_basic() {
        let log_prices: Vec<f64> = (0..50)
            .map(|i| 4.6 + (i as f64) * 0.001 + (i as f64 * 0.1).sin() * 0.01)
            .collect();
        let stats = chu_stinchcombe_white(&log_prices, 1.96);
        assert!(!stats.is_empty());
        for &s in &stats {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_csw_short_series() {
        let stats = chu_stinchcombe_white(&[1.0, 2.0], 1.96);
        assert!(stats.is_empty());
    }
}
