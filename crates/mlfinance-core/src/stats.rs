//! Descriptive statistics, correlation, and covariance matrices.
//!
//! All functions in this module operate on plain `&[f64]` slices or `ndarray`
//! arrays and return [`Result`] when they can fail due to
//! insufficient data.

use crate::error::{MlFinanceError, Result};
use ndarray::{Array1, Array2};

/// Compute the arithmetic mean of a slice.
///
/// # Arguments
///
/// * `values` - Non-empty data slice.
///
/// # Errors
///
/// Returns [`MlFinanceError::EmptySeries`] if `values` is empty.
pub fn mean(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(MlFinanceError::EmptySeries);
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

/// Compute the sample variance with the specified delta degrees of freedom.
///
/// # Arguments
///
/// * `values` - Data slice with more than `ddof` elements.
/// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance).
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `values.len() <= ddof`.
pub fn variance(values: &[f64], ddof: usize) -> Result<f64> {
    if values.len() <= ddof {
        return Err(MlFinanceError::InsufficientData {
            expected: ddof + 1,
            actual: values.len(),
        });
    }
    let m = mean(values)?;
    let sum_sq: f64 = values.iter().map(|&x| (x - m).powi(2)).sum();
    Ok(sum_sq / (values.len() - ddof) as f64)
}

/// Compute the sample standard deviation with the specified delta degrees of freedom.
///
/// # Arguments
///
/// * `values` - Data slice with more than `ddof` elements.
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample).
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `values.len() <= ddof`.
pub fn std_dev(values: &[f64], ddof: usize) -> Result<f64> {
    Ok(variance(values, ddof)?.sqrt())
}

/// Compute the adjusted Fisher-Pearson skewness coefficient.
///
/// # Arguments
///
/// * `values` - Data slice with at least 3 elements.
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `values` has fewer than 3 elements.
pub fn skewness(values: &[f64]) -> Result<f64> {
    let n = values.len();
    if n < 3 {
        return Err(MlFinanceError::InsufficientData {
            expected: 3,
            actual: n,
        });
    }
    let m = mean(values)?;
    let s = std_dev(values, 1)?;
    if s == 0.0 {
        return Ok(0.0);
    }
    let nf = n as f64;
    let sum: f64 = values.iter().map(|&x| ((x - m) / s).powi(3)).sum();
    Ok(sum * nf / ((nf - 1.0) * (nf - 2.0)))
}

/// Compute the excess kurtosis using the bias-corrected formula.
///
/// # Arguments
///
/// * `values` - Data slice with at least 4 elements.
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `values` has fewer than 4 elements.
pub fn kurtosis(values: &[f64]) -> Result<f64> {
    let n = values.len();
    if n < 4 {
        return Err(MlFinanceError::InsufficientData {
            expected: 4,
            actual: n,
        });
    }
    let m = mean(values)?;
    let s = std_dev(values, 1)?;
    if s == 0.0 {
        return Ok(0.0);
    }
    let nf = n as f64;
    let sum: f64 = values.iter().map(|&x| ((x - m) / s).powi(4)).sum();
    let excess = (nf * (nf + 1.0) * sum) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0))
        - (3.0 * (nf - 1.0).powi(2)) / ((nf - 2.0) * (nf - 3.0));
    Ok(excess)
}

/// Compute column means and standard deviations for a data matrix.
fn column_stats(data: &Array2<f64>, ncols: usize, nrows: usize) -> (Vec<f64>, Vec<f64>) {
    let means: Vec<f64> = (0..ncols)
        .map(|j| data.column(j).mean().unwrap_or(0.0))
        .collect();
    let stds: Vec<f64> = (0..ncols)
        .map(|j| {
            let m = means[j];
            let var: f64 =
                data.column(j).iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (nrows - 1) as f64;
            var.sqrt()
        })
        .collect();
    (means, stds)
}

/// Compute Pearson correlation between two columns.
fn pearson_pair(
    data: &Array2<f64>,
    means: &[f64],
    stds: &[f64],
    i: usize,
    j: usize,
    nrows: usize,
) -> f64 {
    if stds[i] == 0.0 || stds[j] == 0.0 {
        return 0.0;
    }
    let cov: f64 = (0..nrows)
        .map(|k| (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]))
        .sum::<f64>()
        / (nrows - 1) as f64;
    cov / (stds[i] * stds[j])
}

/// Compute the Pearson correlation matrix from an observation matrix.
///
/// # Arguments
///
/// * `data` - An `(n_samples, n_features)` matrix of observations.
///
/// # Returns
///
/// A symmetric `(n_features, n_features)` correlation matrix with ones on the diagonal.
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `data` has fewer than 2 rows.
pub fn correlation_matrix(data: &Array2<f64>) -> Result<Array2<f64>> {
    let ncols = data.ncols();
    let nrows = data.nrows();
    if nrows < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: nrows,
        });
    }

    let (means, stds) = column_stats(data, ncols, nrows);
    let mut corr = Array2::eye(ncols);

    for i in 0..ncols {
        for j in (i + 1)..ncols {
            let r = pearson_pair(data, &means, &stds, i, j, nrows);
            corr[[i, j]] = r;
            corr[[j, i]] = r;
        }
    }
    Ok(corr)
}

/// Compute the sample covariance matrix from an observation matrix.
///
/// Uses Bessel's correction (divides by `n - 1`).
///
/// # Arguments
///
/// * `data` - An `(n_samples, n_features)` matrix of observations.
///
/// # Returns
///
/// A symmetric `(n_features, n_features)` covariance matrix.
///
/// # Errors
///
/// Returns [`MlFinanceError::InsufficientData`] if `data` has fewer than 2 rows.
pub fn covariance_matrix(data: &Array2<f64>) -> Result<Array2<f64>> {
    let ncols = data.ncols();
    let nrows = data.nrows();
    if nrows < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: nrows,
        });
    }
    let means: Vec<f64> = (0..ncols)
        .map(|j| data.column(j).mean().unwrap_or(0.0))
        .collect();
    let mut cov = Array2::zeros((ncols, ncols));
    for i in 0..ncols {
        for j in i..ncols {
            let c: f64 = (0..nrows)
                .map(|k| (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]))
                .sum::<f64>()
                / (nrows - 1) as f64;
            cov[[i, j]] = c;
            cov[[j, i]] = c;
        }
    }
    Ok(cov)
}

/// Compute the weighted arithmetic mean.
///
/// # Arguments
///
/// * `values` - Data values.
/// * `weights` - Non-negative weights; must be the same length as `values` and sum to a non-zero value.
///
/// # Errors
///
/// Returns [`MlFinanceError::DimensionMismatch`] if `values` and `weights` differ
/// in length, or [`MlFinanceError::InvalidParameter`] if the weights sum to zero.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> Result<f64> {
    if values.len() != weights.len() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "values length {} != weights length {}",
                values.len(),
                weights.len()
            ),
        });
    }
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "weights sum to zero".into(),
        });
    }
    let sum: f64 = values.iter().zip(weights).map(|(&v, &w)| v * w).sum();
    Ok(sum / total_weight)
}

/// Compute the arithmetic mean of a 1-D ndarray.
///
/// Returns `0.0` for an empty array.
///
/// # Arguments
///
/// * `arr` - A 1-D array of values.
pub fn array_mean(arr: &Array1<f64>) -> f64 {
    arr.mean().unwrap_or(0.0)
}

/// Compute the standard deviation of a 1-D ndarray with the given delta degrees of freedom.
///
/// Returns `0.0` if the array has `ddof` or fewer elements.
///
/// # Arguments
///
/// * `arr` - A 1-D array of values.
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample).
pub fn array_std(arr: &Array1<f64>, ddof: usize) -> f64 {
    let n = arr.len();
    if n <= ddof {
        return 0.0;
    }
    let m = array_mean(arr);
    let var: f64 = arr.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n - ddof) as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let v = variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 1).unwrap();
        assert!((v - 4.571428571428571).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let s = std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 1).unwrap();
        assert!((s - 2.138089935299395).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix() {
        let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let corr = correlation_matrix(&data).unwrap();
        assert!((corr[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((corr[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean() {
        let wm = weighted_mean(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]).unwrap();
        assert!((wm - 2.0).abs() < 1e-10);
    }
}
