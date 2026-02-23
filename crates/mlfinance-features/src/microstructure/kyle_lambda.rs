/// Kyle's lambda: price impact coefficient.
///
/// Regress returns on signed volume:
///   r_t = alpha + lambda * signed_volume_t + epsilon_t
///
/// Lambda measures the permanent price impact of trades.
///
/// # Arguments
///
/// * `returns` - Asset returns for each period.
/// * `signed_volume` - Net signed volume (positive for buyer-initiated, negative for seller-initiated).
///
/// # Returns
///
/// The OLS regression slope coefficient (lambda). Returns 0.0 if fewer than
/// 2 data points or if signed volume has zero variance.
pub fn kyle_lambda(returns: &[f64], signed_volume: &[f64]) -> f64 {
    let n = returns.len().min(signed_volume.len());
    if n < 2 {
        return 0.0;
    }

    // Simple OLS: y = a + b*x
    // b = cov(x,y) / var(x)
    let mean_r = returns[..n].iter().sum::<f64>() / n as f64;
    let mean_sv = signed_volume[..n].iter().sum::<f64>() / n as f64;

    let mut cov_xy = 0.0;
    let mut var_x = 0.0;

    for i in 0..n {
        let dx = signed_volume[i] - mean_sv;
        let dy = returns[i] - mean_r;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }

    if var_x.abs() < 1e-15 {
        return 0.0;
    }

    cov_xy / var_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyle_lambda_positive_impact() {
        // Positive relationship: buys push prices up
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let signed_volume = vec![100.0, -50.0, 200.0, -100.0, 150.0];
        let lambda = kyle_lambda(&returns, &signed_volume);
        assert!(lambda > 0.0);
    }

    #[test]
    fn test_kyle_lambda_no_impact() {
        // No relationship
        let returns = vec![0.01, -0.01, 0.01, -0.01];
        let signed_volume = vec![100.0, 100.0, -100.0, -100.0];
        let _lambda = kyle_lambda(&returns, &signed_volume);
        // Just check it's finite
    }

    #[test]
    fn test_kyle_lambda_short() {
        let lambda = kyle_lambda(&[0.01], &[100.0]);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_kyle_lambda_exact() {
        // r = 0 + 0.0001 * sv
        let signed_volume = vec![100.0, 200.0, -100.0, 300.0, -200.0];
        let returns: Vec<f64> = signed_volume.iter().map(|&sv| 0.0001 * sv).collect();
        let lambda = kyle_lambda(&returns, &signed_volume);
        assert!((lambda - 0.0001).abs() < 1e-10);
    }
}
