//! Scalar math helpers: EWMA, cumulative sums, returns, and sign.
//!
//! These are standalone functions that operate on plain `&[f64]` slices. For
//! time-indexed variants see [`crate::series::TimeSeries`].

use crate::error::{MlFinanceError, Result};

/// Exponentially Weighted Moving Average.
///
/// Computes EWMA values for the input series using the given span.
/// The smoothing factor is `alpha = 2 / (span + 1)`.
///
/// # Arguments
///
/// * `values` - Input data slice (must be non-empty).
/// * `span` - EWMA span (must be > 0).
///
/// # Returns
///
/// A vector of the same length as `values` containing the EWMA series.
///
/// # Errors
///
/// Returns [`MlFinanceError::EmptySeries`] if `values` is empty, or
/// [`MlFinanceError::InvalidParameter`] if `span` is 0.
pub fn ewma(values: &[f64], span: usize) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(MlFinanceError::EmptySeries);
    }
    if span == 0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "span must be > 0".into(),
        });
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut result = Vec::with_capacity(values.len());
    result.push(values[0]);
    for i in 1..values.len() {
        let prev = result[i - 1];
        result.push(alpha * values[i] + (1.0 - alpha) * prev);
    }
    Ok(result)
}

/// EWMA of the standard deviation (volatility estimator).
///
/// Computes a rolling EWMA standard deviation over `span` using successive
/// differences. The first element is always `0.0`.
///
/// # Arguments
///
/// * `values` - Input data slice (must have at least 2 elements).
/// * `span` - EWMA span controlling the decay rate.
///
/// # Returns
///
/// A vector of the same length as `values` with EWMA volatility estimates.
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `values` has fewer than 2 elements.
pub fn ewma_std(values: &[f64], span: usize) -> Result<Vec<f64>> {
    if values.len() < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: values.len(),
        });
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut variance = 0.0_f64;
    let mut result = Vec::with_capacity(values.len());
    result.push(0.0);

    for i in 1..values.len() {
        let diff = values[i] - values[i - 1];
        variance = (1.0 - alpha) * variance + alpha * diff * diff;
        result.push(variance.sqrt());
    }
    Ok(result)
}

/// Cumulative sum of values.
///
/// # Arguments
///
/// * `values` - Input data slice.
///
/// # Returns
///
/// A vector of the same length where entry `i` equals `sum(values[0..=i])`.
pub fn cumsum(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    for &v in values {
        sum += v;
        result.push(sum);
    }
    result
}

/// Compute log returns: `ln(p[i] / p[i-1])`.
///
/// Non-positive prices produce a return of `0.0`.
///
/// # Arguments
///
/// * `prices` - Price series (at least 2 elements for non-empty output).
///
/// # Returns
///
/// A vector of length `prices.len() - 1`, or empty if fewer than 2 prices.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    prices
        .windows(2)
        .map(|w| {
            if w[0] <= 0.0 || w[1] <= 0.0 {
                0.0
            } else {
                (w[1] / w[0]).ln()
            }
        })
        .collect()
}

/// Compute simple (arithmetic) returns: `(p[i] - p[i-1]) / p[i-1]`.
///
/// Division by zero (when `p[i-1] == 0.0`) produces `0.0`.
///
/// # Arguments
///
/// * `prices` - Price series (at least 2 elements for non-empty output).
///
/// # Returns
///
/// A vector of length `prices.len() - 1`, or empty if fewer than 2 prices.
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    prices
        .windows(2)
        .map(|w| {
            if w[0] == 0.0 {
                0.0
            } else {
                (w[1] - w[0]) / w[0]
            }
        })
        .collect()
}

/// Sign function: returns `1.0` if `x > 0`, `-1.0` if `x < 0`, or `0.0` if `x == 0`.
///
/// # Arguments
///
/// * `x` - The value whose sign is queried.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewma(&vals, 3).unwrap();
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10);
        // alpha = 0.5: ewma[1] = 0.5*2 + 0.5*1 = 1.5
        assert!((result[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let lr = log_returns(&prices);
        assert_eq!(lr.len(), 2);
        assert!((lr[0] - (1.1_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_simple_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let sr = simple_returns(&prices);
        assert!((sr[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_cumsum() {
        assert_eq!(cumsum(&[1.0, 2.0, 3.0]), vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_sign() {
        assert_eq!(sign(5.0), 1.0);
        assert_eq!(sign(-3.0), -1.0);
        assert_eq!(sign(0.0), 0.0);
    }
}
