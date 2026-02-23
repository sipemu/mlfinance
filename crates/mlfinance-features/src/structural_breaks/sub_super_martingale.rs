/// Polynomial sub-martingale test for trend detection.
///
/// Tests whether the series follows a polynomial sub-martingale process.
/// Computes a t-statistic from polynomial-transformed increments:
/// sign(delta) * |delta|^degree.
///
/// # Arguments
///
/// * `series` - Time series to test (requires at least 3 values).
/// * `degree` - Polynomial degree for the transformation (must be > 0).
///
/// # Returns
///
/// The t-statistic. Positive values indicate upward drift (sub-martingale),
/// negative values indicate downward drift (super-martingale).
/// Returns 0.0 if the series is too short or degree is 0.
pub fn sm_poly(series: &[f64], degree: usize) -> f64 {
    let n = series.len();
    if n < 3 || degree == 0 {
        return 0.0;
    }

    // Compute increments
    let increments: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    // Compute polynomial-transformed increments: sign(delta) * |delta|^degree
    let transformed: Vec<f64> = increments
        .iter()
        .map(|&d| d.signum() * d.abs().powf(degree as f64))
        .collect();

    // Mean of transformed increments
    let mean_t = transformed.iter().sum::<f64>() / transformed.len() as f64;

    // Standard error
    let var_t = transformed
        .iter()
        .map(|&t| (t - mean_t).powi(2))
        .sum::<f64>()
        / (transformed.len() - 1).max(1) as f64;
    let se = (var_t / transformed.len() as f64).sqrt();

    if se < 1e-15 {
        return 0.0;
    }

    // t-statistic: mean / se
    mean_t / se
}

/// Exponential sub-martingale test for trend detection.
///
/// Uses exponential transformation of increments: exp(delta) - 1.
/// This transformation is asymmetric and sensitive to positive drift.
///
/// # Arguments
///
/// * `series` - Time series to test (requires at least 3 values).
///
/// # Returns
///
/// The t-statistic. Positive values indicate upward drift.
/// Returns 0.0 if the series is too short.
pub fn sm_exp(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 3 {
        return 0.0;
    }

    // Compute increments
    let increments: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    // Exponential transformation: exp(delta) - 1
    // This is sensitive to positive drift (sub-martingale)
    let transformed: Vec<f64> = increments.iter().map(|&d| d.exp() - 1.0).collect();

    // Mean of transformed increments
    let mean_t = transformed.iter().sum::<f64>() / transformed.len() as f64;

    // Standard error
    let var_t = transformed
        .iter()
        .map(|&t| (t - mean_t).powi(2))
        .sum::<f64>()
        / (transformed.len() - 1).max(1) as f64;
    let se = (var_t / transformed.len() as f64).sqrt();

    if se < 1e-15 {
        return 0.0;
    }

    mean_t / se
}

/// Power sub-martingale test for trend detection.
///
/// Uses power transformation of increments: sign(delta) * |delta|^power.
///
/// # Arguments
///
/// * `series` - Time series to test (requires at least 3 values).
/// * `power` - Exponent for the power transformation (must be positive).
///
/// # Returns
///
/// The t-statistic. Positive values indicate upward drift.
/// Returns 0.0 if the series is too short or power <= 0.
pub fn sm_power(series: &[f64], power: f64) -> f64 {
    let n = series.len();
    if n < 3 || power <= 0.0 {
        return 0.0;
    }

    // Compute increments
    let increments: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    // Power transformation: sign(delta) * |delta|^power
    let transformed: Vec<f64> = increments
        .iter()
        .map(|&d| d.signum() * d.abs().powf(power))
        .collect();

    // Mean of transformed increments
    let mean_t = transformed.iter().sum::<f64>() / transformed.len() as f64;

    // Standard error
    let var_t = transformed
        .iter()
        .map(|&t| (t - mean_t).powi(2))
        .sum::<f64>()
        / (transformed.len() - 1).max(1) as f64;
    let se = (var_t / transformed.len() as f64).sqrt();

    if se < 1e-15 {
        return 0.0;
    }

    mean_t / se
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm_poly_trending() {
        // Upward trending series with noise so variance is nonzero
        let series: Vec<f64> = (0..100)
            .map(|i| i as f64 * 0.1 + (i as f64 * 0.3).sin() * 0.02)
            .collect();
        let stat = sm_poly(&series, 1);
        assert!(stat > 0.0);
    }

    #[test]
    fn test_sm_poly_stationary() {
        // Oscillating series should be near zero
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).sin()).collect();
        let stat = sm_poly(&series, 1);
        // Should be close to zero
        assert!(stat.abs() < 3.0);
    }

    #[test]
    fn test_sm_exp_trending() {
        // Trending with noise so increments have nonzero variance
        let series: Vec<f64> = (0..100)
            .map(|i| i as f64 * 0.01 + (i as f64 * 0.7).sin() * 0.001)
            .collect();
        let stat = sm_exp(&series);
        assert!(stat > 0.0);
    }

    #[test]
    fn test_sm_power_trending() {
        // Trending with noise
        let series: Vec<f64> = (0..100)
            .map(|i| i as f64 * 0.1 + (i as f64 * 0.3).sin() * 0.02)
            .collect();
        let stat = sm_power(&series, 0.5);
        assert!(stat > 0.0);
    }

    #[test]
    fn test_sm_poly_short() {
        let stat = sm_poly(&[1.0, 2.0], 1);
        assert_eq!(stat, 0.0);
    }

    #[test]
    fn test_sm_poly_degree_zero() {
        let stat = sm_poly(&[1.0, 2.0, 3.0], 0);
        assert_eq!(stat, 0.0);
    }

    #[test]
    fn test_sm_power_negative_power() {
        let stat = sm_power(&[1.0, 2.0, 3.0], -1.0);
        assert_eq!(stat, 0.0);
    }
}
