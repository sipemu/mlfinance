use mlfinance_core::error::{MlFinanceError, Result};
use mlfinance_core::stats::correlation_matrix;
use ndarray::Array2;

use crate::codependence::correlation::{
    absolute_angular_distance_from_corr, angular_distance_from_corr,
    distance_correlation as compute_distance_correlation, squared_angular_distance_from_corr,
};
use crate::codependence::gnpr_distance::spearmans_rho;
use crate::codependence::information::{mutual_information, variation_of_information};

/// Available dependence methods for pairwise computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependenceMethod {
    /// Pearson linear correlation
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Distance correlation (Szekely et al.)
    DistanceCorrelation,
    /// Mutual information (optionally normalized)
    MutualInformation,
    /// Variation of Information
    VariationOfInformation,
}

/// Compute the diagonal (self-dependence) value for a given method.
fn diagonal_value(col: &[f64], method: DependenceMethod) -> Result<f64> {
    Ok(match method {
        DependenceMethod::Spearman | DependenceMethod::DistanceCorrelation => 1.0,
        DependenceMethod::MutualInformation => mutual_information(col, col, None, true)?,
        DependenceMethod::VariationOfInformation => 0.0,
        DependenceMethod::Pearson => unreachable!(),
    })
}

/// Compute the pairwise dependence between two columns for a given method.
fn pairwise_value(col_i: &[f64], col_j: &[f64], method: DependenceMethod) -> Result<f64> {
    match method {
        DependenceMethod::Spearman => spearmans_rho(col_i, col_j),
        DependenceMethod::DistanceCorrelation => compute_distance_correlation(col_i, col_j),
        DependenceMethod::MutualInformation => mutual_information(col_i, col_j, None, true),
        DependenceMethod::VariationOfInformation => {
            variation_of_information(col_i, col_j, None, true)
        }
        DependenceMethod::Pearson => unreachable!(),
    }
}

/// Compute pairwise dependence matrix for columns of a data matrix.
///
/// `data` has shape (n_samples, n_features).
/// Returns an (n_features, n_features) matrix of pairwise dependence values.
/// Fill one row (and its symmetric counterpart) of the dependence matrix.
fn fill_dependence_row(
    result: &mut Array2<f64>,
    columns: &[Vec<f64>],
    i: usize,
    n_features: usize,
    method: DependenceMethod,
) -> Result<()> {
    result[[i, i]] = diagonal_value(&columns[i], method)?;
    for j in (i + 1)..n_features {
        let val = pairwise_value(&columns[i], &columns[j], method)?;
        result[[i, j]] = val;
        result[[j, i]] = val;
    }
    Ok(())
}

/// Validate dependence matrix inputs.
fn validate_dependence_inputs(n_samples: usize) -> Result<()> {
    if n_samples < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: n_samples,
        });
    }
    Ok(())
}

/// Compute a pairwise dependence matrix for columns of a data matrix.
///
/// Each entry `(i, j)` contains the dependence between columns `i` and `j`
/// according to the specified method. The matrix is always symmetric.
///
/// # Arguments
///
/// * `data` - Data matrix with shape (n_samples, n_features), requires >= 2 samples.
/// * `method` - The dependence measure to use (Pearson, Spearman, etc.).
///
/// # Returns
///
/// An (n_features x n_features) symmetric dependence matrix.
///
/// # Errors
///
/// Returns an error if there are fewer than 2 samples.
pub fn dependence_matrix(data: &Array2<f64>, method: DependenceMethod) -> Result<Array2<f64>> {
    let n_features = data.ncols();
    validate_dependence_inputs(data.nrows())?;

    if n_features == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    if method == DependenceMethod::Pearson {
        return correlation_matrix(data);
    }

    let mut result = Array2::zeros((n_features, n_features));
    let columns: Vec<Vec<f64>> = (0..n_features).map(|j| data.column(j).to_vec()).collect();

    for i in 0..n_features {
        fill_dependence_row(&mut result, &columns, i, n_features, method)?;
    }

    Ok(result)
}

/// Distance metric for converting correlation/dependence matrices to distance matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Angular distance: arccos(rho) / pi
    Angular,
    /// Absolute angular distance: arccos(|rho|) / pi
    AbsoluteAngular,
    /// Squared angular distance: arccos(rho^2) / pi
    SquaredAngular,
}

/// Convert a dependence matrix to a distance matrix.
///
/// Applies the specified angular distance metric element-wise
/// to a correlation or dependence matrix.
///
/// # Arguments
///
/// * `corr` - Square correlation or dependence matrix (n x n).
/// * `metric` - The angular distance metric to apply.
///
/// # Returns
///
/// An (n x n) symmetric distance matrix with non-negative entries.
///
/// # Errors
///
/// Returns an error if the matrix is not square.
pub fn distance_matrix(corr: &Array2<f64>, metric: DistanceMetric) -> Result<Array2<f64>> {
    let n = corr.nrows();
    if n != corr.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "matrix must be square: got {}x{}",
                corr.nrows(),
                corr.ncols()
            ),
        });
    }

    let mut dist = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let rho = corr[[i, j]];
            dist[[i, j]] = match metric {
                DistanceMetric::Angular => angular_distance_from_corr(rho),
                DistanceMetric::AbsoluteAngular => absolute_angular_distance_from_corr(rho),
                DistanceMetric::SquaredAngular => squared_angular_distance_from_corr(rho),
            };
        }
    }

    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---- dependence_matrix tests ----

    #[test]
    fn test_dependence_matrix_pearson() {
        let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let dm = dependence_matrix(&data, DependenceMethod::Pearson).unwrap();
        assert_abs_diff_eq!(dm[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[1, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dependence_matrix_spearman() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0]
        ];
        let dm = dependence_matrix(&data, DependenceMethod::Spearman).unwrap();
        assert_abs_diff_eq!(dm[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dependence_matrix_distance_correlation() {
        let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]];
        let dm = dependence_matrix(&data, DependenceMethod::DistanceCorrelation).unwrap();
        assert_abs_diff_eq!(dm[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dependence_matrix_mutual_information() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0],
            [9.0, 90.0],
            [10.0, 100.0]
        ];
        let dm = dependence_matrix(&data, DependenceMethod::MutualInformation).unwrap();
        // NMI should be high for perfectly correlated data
        assert!(dm[[0, 1]] > 0.5);
    }

    #[test]
    fn test_dependence_matrix_variation_of_information() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0],
            [9.0, 90.0],
            [10.0, 100.0]
        ];
        let dm = dependence_matrix(&data, DependenceMethod::VariationOfInformation).unwrap();
        // Diagonal should be 0 (no variation with self)
        assert_abs_diff_eq!(dm[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dm[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dependence_matrix_symmetric() {
        let data = array![
            [1.0, 5.0, 3.0],
            [2.0, 3.0, 7.0],
            [3.0, 7.0, 2.0],
            [4.0, 2.0, 8.0],
            [5.0, 8.0, 1.0]
        ];
        let dm = dependence_matrix(&data, DependenceMethod::Spearman).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dm[[i, j]], dm[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dependence_matrix_insufficient_data() {
        let data = array![[1.0, 2.0]];
        assert!(dependence_matrix(&data, DependenceMethod::Pearson).is_err());
    }

    #[test]
    fn test_dependence_matrix_empty_features() {
        let data = Array2::<f64>::zeros((5, 0));
        let dm = dependence_matrix(&data, DependenceMethod::Pearson).unwrap();
        assert_eq!(dm.nrows(), 0);
        assert_eq!(dm.ncols(), 0);
    }

    // ---- distance_matrix tests ----

    #[test]
    fn test_distance_matrix_angular_identity() {
        let corr = Array2::eye(3);
        let dist = distance_matrix(&corr, DistanceMetric::Angular).unwrap();
        // Diagonal: arccos(1)/pi = 0
        for i in 0..3 {
            assert_abs_diff_eq!(dist[[i, i]], 0.0, epsilon = 1e-10);
        }
        // Off-diagonal: arccos(0)/pi = 0.5
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert_abs_diff_eq!(dist[[i, j]], 0.5, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_distance_matrix_angular_perfect() {
        let corr = array![[1.0, 1.0], [1.0, 1.0]];
        let dist = distance_matrix(&corr, DistanceMetric::Angular).unwrap();
        assert_abs_diff_eq!(dist[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_matrix_absolute_angular() {
        let corr = array![[1.0, -0.8], [-0.8, 1.0]];
        let dist = distance_matrix(&corr, DistanceMetric::AbsoluteAngular).unwrap();
        // |rho| = 0.8, arccos(0.8)/pi
        let expected = 0.8_f64.acos() / std::f64::consts::PI;
        assert_abs_diff_eq!(dist[[0, 1]], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_matrix_squared_angular() {
        let corr = array![[1.0, 0.6], [0.6, 1.0]];
        let dist = distance_matrix(&corr, DistanceMetric::SquaredAngular).unwrap();
        // rho^2 = 0.36, arccos(0.36)/pi
        let expected = 0.36_f64.acos() / std::f64::consts::PI;
        assert_abs_diff_eq!(dist[[0, 1]], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_matrix_nonnegative() {
        let corr = array![[1.0, 0.3, -0.5], [0.3, 1.0, 0.7], [-0.5, 0.7, 1.0]];
        let dist = distance_matrix(&corr, DistanceMetric::Angular).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(dist[[i, j]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_distance_matrix_not_square() {
        let m = Array2::<f64>::zeros((2, 3));
        assert!(distance_matrix(&m, DistanceMetric::Angular).is_err());
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let corr = array![[1.0, 0.3, -0.5], [0.3, 1.0, 0.7], [-0.5, 0.7, 1.0]];
        let dist = distance_matrix(&corr, DistanceMetric::Angular).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dist[[i, j]], dist[[j, i]], epsilon = 1e-10);
            }
        }
    }
}
