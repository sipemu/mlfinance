use chrono::{DateTime, TimeZone, Utc};
use mlfinance::core::MlFinanceError;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ndarray → numpy

pub fn array1_to_py(py: Python<'_>, arr: Array1<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_array(py, &arr).unbind()
}

pub fn array2_to_py(py: Python<'_>, arr: Array2<f64>) -> Py<PyArray2<f64>> {
    PyArray2::from_array(py, &arr).unbind()
}

pub fn vec_to_py_array(py: Python<'_>, v: Vec<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, v).unbind()
}

pub fn vec_i32_to_py_array(py: Python<'_>, v: Vec<i32>) -> Py<PyArray1<i32>> {
    PyArray1::from_vec(py, v).unbind()
}

pub fn vec_usize_to_list(py: Python<'_>, v: Vec<usize>) -> PyResult<Py<PyAny>> {
    Ok(v.into_pyobject(py)?.into_any().unbind())
}

pub fn vec_bool_to_list(py: Python<'_>, v: Vec<bool>) -> PyResult<Py<PyAny>> {
    Ok(v.into_pyobject(py)?.into_any().unbind())
}

// numpy → ndarray

pub fn py_to_array1(arr: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    arr.as_array().to_owned()
}

pub fn py_to_array2(arr: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    arr.as_array().to_owned()
}

pub fn py_to_vec(arr: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    arr.as_array().to_vec()
}

// Timestamp conversion

pub fn timestamp_from_f64(ts: f64) -> DateTime<Utc> {
    let secs = ts as i64;
    let nsecs = ((ts - secs as f64) * 1e9) as u32;
    Utc.timestamp_opt(secs, nsecs).unwrap()
}

pub fn timestamp_to_f64(ts: DateTime<Utc>) -> f64 {
    ts.timestamp() as f64 + ts.timestamp_subsec_nanos() as f64 / 1e9
}

// Error mapping

pub fn to_pyresult<T>(r: Result<T, MlFinanceError>) -> PyResult<T> {
    r.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
