use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;

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

#[pyfunction]
fn cumsum(py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let v = py_to_vec(values);
    vec_to_py_array(py, mlfinance::core::math::cumsum(&v))
}

#[pyfunction]
fn log_returns(py: Python<'_>, prices: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let v = py_to_vec(prices);
    vec_to_py_array(py, mlfinance::core::math::log_returns(&v))
}

#[pyfunction]
fn simple_returns(py: Python<'_>, prices: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let v = py_to_vec(prices);
    vec_to_py_array(py, mlfinance::core::math::simple_returns(&v))
}

#[pyfunction]
fn mean(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::mean(&v))
}

#[pyfunction]
#[pyo3(signature = (values, ddof=1))]
fn variance(values: PyReadonlyArray1<'_, f64>, ddof: usize) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::variance(&v, ddof))
}

#[pyfunction]
#[pyo3(signature = (values, ddof=1))]
fn std_dev(values: PyReadonlyArray1<'_, f64>, ddof: usize) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::std_dev(&v, ddof))
}

#[pyfunction]
fn skewness(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::skewness(&v))
}

#[pyfunction]
fn kurtosis(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = py_to_vec(values);
    to_pyresult(mlfinance::core::stats::kurtosis(&v))
}

#[pyfunction]
fn weighted_mean(
    values: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let v = py_to_vec(values);
    let w = py_to_vec(weights);
    to_pyresult(mlfinance::core::stats::weighted_mean(&v, &w))
}

#[pyfunction]
fn correlation_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = py_to_array2(data);
    let result = to_pyresult(mlfinance::core::stats::correlation_matrix(&arr))?;
    Ok(array2_to_py(py, result))
}

#[pyfunction]
fn covariance_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = py_to_array2(data);
    let result = to_pyresult(mlfinance::core::stats::covariance_matrix(&arr))?;
    Ok(array2_to_py(py, result))
}

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
    m.add_function(wrap_pyfunction!(ewma, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_std, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(simple_returns, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(power_iteration_eig, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_inverse, m)?)?;
    Ok(())
}
