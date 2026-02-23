//! Sharpe Ratio computation.
//!
//! Computes the annualized Sharpe Ratio from a return series.

/// Annualized Sharpe Ratio.
///
/// Computes SR = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year),
/// where mean and std are computed from the return series and the risk-free rate
/// is expressed per period (same frequency as the returns).
///
/// # Arguments
/// * `returns` - Periodic return series.
/// * `risk_free_rate` - Per-period risk-free rate.
/// * `periods_per_year` - Number of periods per year (e.g., 252 for daily, 12 for monthly).
///
/// # Returns
/// Annualized Sharpe Ratio. Returns 0.0 if the series is too short or has zero volatility.
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    let n = returns.len();
    if n < 2 || periods_per_year <= 0.0 {
        return 0.0;
    }

    let nf = n as f64;
    let mean: f64 = returns.iter().sum::<f64>() / nf;
    let excess_mean = mean - risk_free_rate;
    let var: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let std = var.sqrt();

    if std < 1e-15 {
        return 0.0;
    }

    (excess_mean / std) * periods_per_year.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio_basic() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sr = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sr.is_finite());
        assert!(sr > 0.0, "positive mean returns should give positive SR");
    }

    #[test]
    fn test_sharpe_ratio_zero_risk_free() {
        let returns = vec![0.01, 0.02, 0.03, 0.01, 0.02];
        let sr = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sr > 0.0);
    }

    #[test]
    fn test_sharpe_ratio_with_risk_free() {
        let returns = vec![0.01, 0.02, 0.03, 0.01, 0.02];
        let sr_no_rf = sharpe_ratio(&returns, 0.0, 252.0);
        let sr_with_rf = sharpe_ratio(&returns, 0.01, 252.0);
        assert!(sr_with_rf < sr_no_rf, "risk-free rate should reduce SR");
    }

    #[test]
    fn test_sharpe_ratio_constant_returns() {
        let returns = vec![0.01, 0.01, 0.01, 0.01];
        let sr = sharpe_ratio(&returns, 0.0, 252.0);
        assert_eq!(sr, 0.0, "zero volatility should give zero SR");
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        assert_eq!(sharpe_ratio(&[], 0.0, 252.0), 0.0);
    }

    #[test]
    fn test_sharpe_ratio_single() {
        assert_eq!(sharpe_ratio(&[0.01], 0.0, 252.0), 0.0);
    }

    #[test]
    fn test_sharpe_ratio_negative() {
        let returns = vec![-0.01, -0.02, -0.01, -0.015, -0.005];
        let sr = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sr < 0.0, "negative returns should give negative SR");
    }

    #[test]
    fn test_sharpe_ratio_annualization() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sr_daily = sharpe_ratio(&returns, 0.0, 252.0);
        let sr_monthly = sharpe_ratio(&returns, 0.0, 12.0);
        // Daily annualization uses sqrt(252) vs sqrt(12), so should be larger
        assert!(sr_daily.abs() > sr_monthly.abs());
    }
}
