//! Linear algebra utilities that avoid an external BLAS/LAPACK dependency.
//!
//! Provides eigendecomposition via power iteration ([`power_iteration_eig`]) and
//! matrix inversion via Gauss-Jordan elimination ([`matrix_inverse`]). These are
//! sufficient for the PCA and HRP algorithms used in this workspace without
//! requiring `ndarray-linalg` and its `openblas-static` linkage.

use crate::error::{MlFinanceError, Result};
use ndarray::{Array1, Array2};

/// Compute the sample covariance matrix from a data matrix.
///
/// This is a thin wrapper around [`crate::stats::covariance_matrix`].
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
pub fn cov_matrix(data: &Array2<f64>) -> Result<Array2<f64>> {
    crate::stats::covariance_matrix(data)
}

/// Find a single eigenvector via power iteration on the given matrix.
fn power_iterate_single(
    matrix: &Array2<f64>,
    n: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> (f64, Array1<f64>) {
    let mut v: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        v[i] = ((i + k + 1) as f64).sin();
    }
    let norm: f64 = v.dot(&v).sqrt();
    v /= norm;

    for _ in 0..max_iter {
        let new_v: Array1<f64> = matrix.dot(&v);
        let norm: f64 = new_v.dot(&new_v).sqrt();
        if norm < 1e-15 {
            break;
        }
        let new_v: Array1<f64> = new_v / norm;
        let diff: f64 = v
            .iter()
            .zip(new_v.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        v = new_v;
        if diff.sqrt() < tol {
            break;
        }
    }

    let eigenvalue: f64 = matrix.dot(&v).dot(&v);
    (eigenvalue, v)
}

/// Deflate a matrix by removing the component along the given eigenvector.
fn deflate_matrix(matrix: &mut Array2<f64>, eigenvalue: f64, v: &Array1<f64>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            matrix[[i, j]] -= eigenvalue * v[i] * v[j];
        }
    }
}

/// Eigendecomposition via deflated power iteration.
///
/// Extracts the leading `num_components` eigenvalues and eigenvectors of a
/// symmetric matrix, sorted by decreasing eigenvalue magnitude. This avoids
/// an external BLAS/LAPACK dependency at the cost of slower convergence on
/// ill-conditioned matrices.
///
/// # Arguments
///
/// * `matrix` - A symmetric square matrix of shape `(n, n)`.
/// * `num_components` - Number of eigen-pairs to extract (clamped to `n`).
/// * `max_iter` - Maximum power-iteration steps per component.
/// * `tol` - Convergence tolerance on the eigenvector change (L2 norm).
///
/// # Returns
///
/// A tuple `(eigenvalues, eigenvectors)` where `eigenvalues` has shape
/// `(num_components,)` and `eigenvectors` has shape `(n, num_components)` with
/// each column being a unit eigenvector.
///
/// # Errors
///
/// Returns [`MlFinanceError::DimensionMismatch`] if the matrix is not square.
pub fn power_iteration_eig(
    matrix: &Array2<f64>,
    num_components: usize,
    max_iter: usize,
    tol: f64,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "matrix must be square".into(),
        });
    }
    let num_components = num_components.min(n);
    let mut eigenvalues = Array1::zeros(num_components);
    let mut eigenvectors = Array2::zeros((n, num_components));
    let mut deflated = matrix.clone();

    for k in 0..num_components {
        let (eigenvalue, v) = power_iterate_single(&deflated, n, k, max_iter, tol);
        eigenvalues[k] = eigenvalue;
        eigenvectors.column_mut(k).assign(&v);
        deflate_matrix(&mut deflated, eigenvalue, &v, n);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Find the row with the largest absolute value in column `col`, starting from row `col`.
fn find_pivot_row(aug: &Array2<f64>, col: usize, n: usize) -> (usize, f64) {
    let mut max_row = col;
    let mut max_val = aug[[col, col]].abs();
    for row in (col + 1)..n {
        let val = aug[[row, col]].abs();
        if val > max_val {
            max_val = val;
            max_row = row;
        }
    }
    (max_row, max_val)
}

/// Swap two rows in the augmented matrix.
fn swap_rows(aug: &mut Array2<f64>, r1: usize, r2: usize, width: usize) {
    for j in 0..width {
        let tmp = aug[[r1, j]];
        aug[[r1, j]] = aug[[r2, j]];
        aug[[r2, j]] = tmp;
    }
}

/// Scale the pivot row and eliminate `col` from all other rows.
fn eliminate_column(aug: &mut Array2<f64>, col: usize, n: usize, width: usize) {
    let pivot = aug[[col, col]];
    for j in 0..width {
        aug[[col, j]] /= pivot;
    }
    for row in 0..n {
        if row == col {
            continue;
        }
        let factor = aug[[row, col]];
        for j in 0..width {
            aug[[row, j]] -= factor * aug[[col, j]];
        }
    }
}

/// Build the augmented matrix [A | I] for Gauss-Jordan elimination.
fn build_augmented(matrix: &Array2<f64>, n: usize) -> Array2<f64> {
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }
    aug
}

/// Extract the right half of the augmented matrix as the inverse.
fn extract_inverse(aug: &Array2<f64>, n: usize) -> Array2<f64> {
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = aug[[i, n + j]];
        }
    }
    result
}

/// Invert a 1x1 matrix.
fn invert_scalar(val: f64) -> Result<Array2<f64>> {
    if val.abs() < 1e-15 {
        return Err(MlFinanceError::ComputationError {
            msg: "singular matrix".into(),
        });
    }
    Ok(Array2::from_elem((1, 1), 1.0 / val))
}

/// Compute the inverse of a symmetric positive-definite matrix using Cholesky-like approach.
/// Falls back to pseudo-inverse if not positive definite.
/// Perform one step of Gauss-Jordan elimination: pivot, swap, and eliminate.
fn gauss_jordan_step(aug: &mut Array2<f64>, col: usize, n: usize, width: usize) -> Result<()> {
    let (max_row, max_val) = find_pivot_row(aug, col, n);
    if max_val < 1e-15 {
        return Err(MlFinanceError::ComputationError {
            msg: "singular matrix".into(),
        });
    }
    if max_row != col {
        swap_rows(aug, col, max_row, width);
    }
    eliminate_column(aug, col, n, width);
    Ok(())
}

/// Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
///
/// # Arguments
///
/// * `matrix` - A square matrix of shape `(n, n)`.
///
/// # Returns
///
/// The inverse matrix of the same shape.
///
/// # Errors
///
/// Returns [`MlFinanceError::DimensionMismatch`] if the matrix is not square, or
/// [`MlFinanceError::ComputationError`] if the matrix is singular.
pub fn matrix_inverse(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "matrix must be square".into(),
        });
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    if n == 1 {
        return invert_scalar(matrix[[0, 0]]);
    }

    let width = 2 * n;
    let mut aug = build_augmented(matrix, n);
    for col in 0..n {
        gauss_jordan_step(&mut aug, col, n, width)?;
    }
    Ok(extract_inverse(&aug, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matrix_inverse_2x2() {
        let m = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = matrix_inverse(&m).unwrap();
        // Check A * A^-1 ≈ I
        let prod = m.dot(&inv);
        assert!((prod[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((prod[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((prod[[0, 1]]).abs() < 1e-10);
        assert!((prod[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_power_iteration() {
        // Symmetric matrix with known eigenvalues
        let m = array![[2.0, 1.0], [1.0, 2.0]];
        let (vals, _vecs) = power_iteration_eig(&m, 2, 1000, 1e-12).unwrap();
        // Eigenvalues should be 3 and 1
        assert!((vals[0] - 3.0).abs() < 1e-4);
        assert!((vals[1] - 1.0).abs() < 1e-4);
    }
}
