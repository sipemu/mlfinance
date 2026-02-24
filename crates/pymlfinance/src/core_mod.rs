use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;

/// Exponentially weighted moving average.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input time series.
/// span : int
///     Decay span (alpha = 2 / (span + 1)).
///
/// Returns
/// -------
/// numpy.ndarray
///     EWMA-smoothed series of the same length.
///
/// Raises
/// ------
/// ValueError
///     If span < 1 or values is empty.
#[pyfunction]
fn ewma(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    span: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = py_to_vec(values);
    let result = to_pyresult(mlfinance::core::math::ewma(&v, span))?;
    Ok(vec_to_py_array(py, result))
}

/// Exponentially weighted moving standard deviation.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input time series.
/// span : int
///     Decay span (alpha = 2 / (span + 1)).
///
/// Returns
/// -------
/// numpy.ndarray
///     EWMA standard deviation series.
///
/// Raises
/// ------
/// ValueError
///     If span < 1 or values is empty.
#[pyfunction]
fn ewma_std(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    span: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let v = py_to_vec(values);
    let result = to_pyresult(mlfinance::core::math::ewma_std(&v, span))?;
    Ok(vec_to_py_array(py, result))
}

/// Cumulative sum of an array.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input array.
///
/// Returns
/// -------
/// numpy.ndarray
///     Cumulative sum with the same length as input.
#[pyfunction]
fn cumsum(py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let v = py_to_vec(values);
    vec_to_py_array(py, mlfinance::core::math::cumsum(&v))
}

/// Logarithmic returns from a price series.
///
/// Computes ``log(p[i] / p[i-1])`` for each consecutive pair.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series of length n.
///
/// Returns
/// -------
/// numpy.ndarray
///     Log returns of length n-1.
#[pyfunction]
fn log_returns(py: Python<'_>, prices: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let v = py_to_vec(prices);
    vec_to_py_array(py, mlfinance::core::math::log_returns(&v))
}

/// Simple (arithmetic) returns from a price series.
///
/// Computes ``(p[i] - p[i-1]) / p[i-1]`` for each consecutive pair.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series of length n.
///
/// Returns
/// -------
/// numpy.ndarray
///     Simple returns of length n-1.
#[pyfunction]
fn simple_returns(py: Python<'_>, prices: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let v = py_to_vec(prices);
    vec_to_py_array(py, mlfinance::core::math::simple_returns(&v))
}

/// Arithmetic mean of an array.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input array.
///
/// Returns
/// -------
/// float
///     Mean value.
///
/// Raises
/// ------
/// ValueError
///     If the array is empty.
#[pyfunction]
fn mean(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::mean(&v))
}

/// Sample variance with configurable degrees of freedom.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input array.
/// ddof : int, default 1
///     Delta degrees of freedom. Use 0 for population variance,
///     1 for sample variance (Bessel's correction).
///
/// Returns
/// -------
/// float
///     Variance of the input.
///
/// Raises
/// ------
/// ValueError
///     If the array has fewer elements than ddof + 1.
#[pyfunction]
#[pyo3(signature = (values, ddof=1))]
fn variance(values: PyReadonlyArray1<'_, f64>, ddof: usize) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::variance(&v, ddof))
}

/// Sample standard deviation with configurable degrees of freedom.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input array.
/// ddof : int, default 1
///     Delta degrees of freedom.
///
/// Returns
/// -------
/// float
///     Standard deviation.
#[pyfunction]
#[pyo3(signature = (values, ddof=1))]
fn std_dev(values: PyReadonlyArray1<'_, f64>, ddof: usize) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::std_dev(&v, ddof))
}

/// Sample skewness (third standardized moment).
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input array (length >= 3).
///
/// Returns
/// -------
/// float
///     Skewness coefficient.
#[pyfunction]
fn skewness(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::skewness(&v))
}

/// Excess kurtosis (fourth standardized moment minus 3).
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input array (length >= 4).
///
/// Returns
/// -------
/// float
///     Excess kurtosis (0 for a normal distribution).
#[pyfunction]
fn kurtosis(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::kurtosis(&v))
}

/// Weighted arithmetic mean.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input values.
/// weights : numpy.ndarray
///     Non-negative weights (same length as values). Need not sum to 1.
///
/// Returns
/// -------
/// float
///     Weighted mean.
#[pyfunction]
fn weighted_mean(
    values: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let v = py_to_vec(values);
    let w = py_to_vec(weights);
    to_pyresult(mlfinance::core::stats::weighted_mean(&v, &w))
}

/// Pearson correlation matrix.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     2-D array of shape (n_observations, n_variables).
///
/// Returns
/// -------
/// numpy.ndarray
///     Symmetric correlation matrix of shape (n_variables, n_variables).
#[pyfunction]
fn correlation_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = py_to_array2(data);
    let result = to_pyresult(mlfinance::core::stats::correlation_matrix(&arr))?;
    Ok(array2_to_py(py, result))
}

/// Sample covariance matrix.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     2-D array of shape (n_observations, n_variables).
///
/// Returns
/// -------
/// numpy.ndarray
///     Covariance matrix of shape (n_variables, n_variables).
#[pyfunction]
fn covariance_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = py_to_array2(data);
    let result = to_pyresult(mlfinance::core::stats::covariance_matrix(&arr))?;
    Ok(array2_to_py(py, result))
}

/// Eigendecomposition via power iteration.
///
/// Computes the top ``num_components`` eigenvalues and eigenvectors of a
/// symmetric matrix using deflated power iteration.
///
/// Parameters
/// ----------
/// matrix : numpy.ndarray
///     Symmetric square matrix.
/// num_components : int
///     Number of leading eigenvalues/vectors to compute.
/// max_iter : int, default 1000
///     Maximum iterations per component.
/// tol : float, default 1e-10
///     Convergence tolerance.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     ``(eigenvalues, eigenvectors)`` where eigenvalues has shape
///     ``(num_components,)`` and eigenvectors has shape
///     ``(n, num_components)`` with columns as eigenvectors.
#[pyfunction]
#[pyo3(signature = (matrix, num_components, max_iter=1000, tol=1e-10))]
fn power_iteration_eig(
    py: Python<'_>,
    matrix: PyReadonlyArray2<'_, f64>,
    num_components: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let m = py_to_array2(matrix);
    let (vals, vecs) = to_pyresult(mlfinance::core::matrix::power_iteration_eig(
        &m,
        num_components,
        max_iter,
        tol,
    ))?;
    Ok((array1_to_py(py, vals), array2_to_py(py, vecs)))
}

/// Invert a square matrix using LU decomposition.
///
/// Parameters
/// ----------
/// matrix : numpy.ndarray
///     Square matrix of shape (n, n).
///
/// Returns
/// -------
/// numpy.ndarray
///     Inverse matrix of shape (n, n).
///
/// Raises
/// ------
/// ValueError
///     If the matrix is singular or not square.
#[pyfunction]
fn matrix_inverse(
    py: Python<'_>,
    matrix: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let m = py_to_array2(matrix);
    let result = to_pyresult(mlfinance::core::matrix::matrix_inverse(&m))?;
    Ok(array2_to_py(py, result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        ewma,
        ewma_std,
        cumsum,
        log_returns,
        simple_returns,
        mean,
        variance,
        std_dev,
        skewness,
        kurtosis,
        weighted_mean,
        correlation_matrix,
        covariance_matrix,
        power_iteration_eig,
        matrix_inverse,
    );
    Ok(())
}
