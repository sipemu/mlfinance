use ndarray::{Array1, Array2};

/// ADF test: compute test statistic for unit root.
///
/// The ADF regression is:
///   delta_y\[t\] = alpha + beta*y\[t-1\] + gamma_1*delta_y\[t-1\] + ... + gamma_p*delta_y\[t-p\] + epsilon
///
/// The ADF statistic is the t-statistic for beta (coefficient on y\[t-1\]).
/// Negative values indicate evidence against the unit root null hypothesis.
///
/// # Arguments
///
/// * `series` - Time series to test for stationarity.
/// * `max_lags` - Maximum number of lagged differences to include.
///
/// # Returns
///
/// A tuple `(adf_stat, betas)` where `adf_stat` is the t-statistic and
/// `betas` is the vector of OLS coefficients. Returns `(0.0, vec![])` if
/// the series is too short.
pub fn adf_test(series: &[f64], max_lags: usize) -> (f64, Vec<f64>) {
    let n = series.len();
    if n < max_lags + 3 {
        return (0.0, vec![]);
    }

    // Compute first differences
    let diffs: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    // Build OLS regression:
    // y = delta_y[lag+1..], x = [1, y[lag..n-2], delta_y[lag-1..n-lag-2], ...]
    let effective_start = max_lags;
    let effective_len = diffs.len() - effective_start;

    if effective_len < max_lags + 2 {
        return (0.0, vec![]);
    }

    // Dependent variable: delta_y[effective_start..]
    let y: Vec<f64> = diffs[effective_start..].to_vec();

    // Number of regressors: intercept + lagged level + max_lags lagged differences
    let n_regressors = 2 + max_lags;

    // Build design matrix
    let mut x = Array2::zeros((effective_len, n_regressors));
    for t in 0..effective_len {
        // Intercept
        x[[t, 0]] = 1.0;
        // Lagged level: y[t + effective_start]  (series value at t + effective_start)
        x[[t, 1]] = series[t + effective_start];
        // Lagged differences
        for lag in 1..=max_lags {
            x[[t, 1 + lag]] = diffs[t + effective_start - lag];
        }
    }

    let betas = get_betas(&y, &x);
    let betas_vec: Vec<f64> = betas.to_vec();

    // Compute residuals
    let y_arr = Array1::from_vec(y.clone());
    let fitted = x.dot(&betas);
    let residuals = &y_arr - &fitted;

    // Compute standard error of beta[1] (coefficient on lagged level)
    let residual_var = residuals.iter().map(|&r| r * r).sum::<f64>()
        / (effective_len - n_regressors).max(1) as f64;

    // (X'X)^-1
    let xtx = x.t().dot(&x);
    let xtx_inv = match mlfinance_core::matrix::matrix_inverse(&xtx) {
        Ok(inv) => inv,
        Err(_) => return (0.0, betas_vec),
    };

    let se_beta1 = (residual_var * xtx_inv[[1, 1]]).max(0.0).sqrt();

    let adf_stat = if se_beta1 > 1e-15 {
        betas_vec[1] / se_beta1
    } else {
        0.0
    };

    (adf_stat, betas_vec)
}

/// Get OLS regression coefficients.
///
/// Solves y = X * beta via the normal equations: beta = (X'X)^{-1} X'y.
///
/// # Arguments
///
/// * `y` - Dependent variable vector.
/// * `x` - Design matrix with shape (n_observations, n_regressors).
///
/// # Returns
///
/// Coefficient vector of length n_regressors. Returns zeros if X'X is singular.
pub fn get_betas(y: &[f64], x: &Array2<f64>) -> Array1<f64> {
    let y_arr = Array1::from_vec(y.to_vec());
    let xtx = x.t().dot(x);
    let xty = x.t().dot(&y_arr);

    match mlfinance_core::matrix::matrix_inverse(&xtx) {
        Ok(xtx_inv) => xtx_inv.dot(&xty),
        Err(_) => {
            // Return zeros if matrix is singular
            Array1::zeros(x.ncols())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_random_walk() {
        // Random walk should have ADF stat close to 0 (non-rejection)
        let mut series = vec![0.0_f64; 100];
        let mut val = 0.0;
        for i in 0..100 {
            val += (i as f64 * 0.1).sin() * 0.01;
            series[i] = val;
        }
        let (stat, betas) = adf_test(&series, 1);
        assert!(stat.is_finite());
        assert!(!betas.is_empty());
    }

    #[test]
    fn test_adf_stationary() {
        // Generate a stationary AR(1) process: y_t = 0.5 * y_{t-1} + noise
        // This is clearly mean-reverting (coefficient < 1)
        let n = 300;
        let mut series = vec![0.0_f64; n];
        for i in 1..n {
            // Deterministic "noise" using sin
            let noise = (i as f64 * 1.7).sin() * 0.5;
            series[i] = 0.5 * series[i - 1] + noise;
        }
        let (stat, betas) = adf_test(&series, 1);
        assert!(stat.is_finite());
        assert!(!betas.is_empty());
        // Should be negative for a stationary series
        assert!(
            stat < 0.0,
            "ADF stat should be negative for stationary series, got {}",
            stat
        );
    }

    #[test]
    fn test_adf_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let (stat, betas) = adf_test(&series, 1);
        assert_eq!(stat, 0.0);
        assert!(betas.is_empty());
    }

    #[test]
    fn test_get_betas_simple() {
        // y = 2*x + 1
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let betas = get_betas(&y, &x);
        assert!((betas[0] - 1.0).abs() < 1e-10);
        assert!((betas[1] - 2.0).abs() < 1e-10);
    }
}
