use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;
use crate::types::*;

// ── Fractional differentiation ──────────────────────────────────────────────

#[pyfunction]
fn get_weights(py: Python<'_>, d: f64, size: usize) -> Py<PyArray1<f64>> {
    let result = mlfinance::sampling::fracdiff::weights::get_weights(d, size);
    vec_to_py_array(py, result)
}

#[pyfunction]
fn get_weights_ffd(py: Python<'_>, d: f64, threshold: f64) -> Py<PyArray1<f64>> {
    let result = mlfinance::sampling::fracdiff::weights::get_weights_ffd(d, threshold);
    vec_to_py_array(py, result)
}

#[pyfunction]
fn frac_diff_ffd(
    py: Python<'_>,
    series: PyReadonlyArray1<'_, f64>,
    d: f64,
    threshold: f64,
) -> Py<PyArray1<f64>> {
    let s = py_to_vec(series);
    let result = mlfinance::sampling::fracdiff::ffd::frac_diff_ffd(&s, d, threshold);
    vec_to_py_array(py, result)
}

#[pyfunction]
fn frac_diff_expanding(
    py: Python<'_>,
    series: PyReadonlyArray1<'_, f64>,
    d: f64,
    threshold: f64,
) -> Py<PyArray1<f64>> {
    let s = py_to_vec(series);
    let result = mlfinance::sampling::fracdiff::expanding::frac_diff_expanding(&s, d, threshold);
    vec_to_py_array(py, result)
}

#[pyfunction]
fn find_min_d(
    series: PyReadonlyArray1<'_, f64>,
    max_d: f64,
    step_size: f64,
    threshold: f64,
) -> f64 {
    let s = py_to_vec(series);
    mlfinance::sampling::fracdiff::min_d::find_min_d(&s, max_d, step_size, threshold)
}

// ── Bootstrap ───────────────────────────────────────────────────────────────

#[pyfunction]
fn standard_bootstrap(
    py: Python<'_>,
    num_observations: usize,
    num_samples: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let result = mlfinance::sampling::bootstrap::standard::standard_bootstrap(
        num_observations,
        num_samples,
        seed,
    );
    vec_usize_to_list(py, result)
}

#[pyfunction]
fn seq_bootstrap(
    py: Python<'_>,
    ind_matrix: PyReadonlyArray2<'_, f64>,
    num_samples: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let m = py_to_array2(ind_matrix);
    let result = mlfinance::sampling::bootstrap::sequential::seq_bootstrap(&m, num_samples, seed);
    vec_usize_to_list(py, result)
}

#[pyfunction]
fn compare_bootstraps(
    ind_matrix: PyReadonlyArray2<'_, f64>,
    num_samples: usize,
    num_trials: usize,
    seed: u64,
) -> PyBootstrapComparison {
    let m = py_to_array2(ind_matrix);
    let result = mlfinance::sampling::bootstrap::monte_carlo::compare_bootstraps(
        &m,
        num_samples,
        num_trials,
        seed,
    );
    PyBootstrapComparison {
        seq_uniqueness: result.seq_uniqueness,
        std_uniqueness: result.std_uniqueness,
    }
}

// ── Concurrency ─────────────────────────────────────────────────────────────

#[pyfunction]
fn num_co_events(
    py: Python<'_>,
    events: Vec<(usize, usize)>,
    num_bars: usize,
) -> PyResult<Py<PyAny>> {
    let result = mlfinance::sampling::concurrency::num_co_events::num_co_events(&events, num_bars);
    vec_usize_to_list(py, result)
}

#[pyfunction]
fn get_indicator_matrix(
    py: Python<'_>,
    events: Vec<(usize, usize)>,
    num_bars: usize,
) -> Py<PyArray2<f64>> {
    let result =
        mlfinance::sampling::concurrency::indicator_matrix::get_indicator_matrix(&events, num_bars);
    array2_to_py(py, result)
}

#[pyfunction]
fn average_uniqueness(
    py: Python<'_>,
    events: Vec<(usize, usize)>,
    num_bars: usize,
) -> Py<PyArray1<f64>> {
    let result =
        mlfinance::sampling::concurrency::average_uniqueness::average_uniqueness(&events, num_bars);
    vec_to_py_array(py, result)
}

// ── Weights ─────────────────────────────────────────────────────────────────

#[pyfunction]
fn balanced_class_weights(py: Python<'_>, labels: Vec<i32>) -> PyResult<Py<PyAny>> {
    let result = mlfinance::sampling::weights::class_weights::balanced_class_weights(&labels);
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

#[pyfunction]
fn return_attribution_weights(
    py: Python<'_>,
    events: Vec<(usize, usize)>,
    returns: PyReadonlyArray1<'_, f64>,
    num_bars: usize,
) -> Py<PyArray1<f64>> {
    let r = py_to_vec(returns);
    let result = mlfinance::sampling::weights::return_attribution::return_attribution_weights(
        &events, &r, num_bars,
    );
    vec_to_py_array(py, result)
}

#[pyfunction]
fn time_decay(
    py: Python<'_>,
    weights: PyReadonlyArray1<'_, f64>,
    oldest_weight: f64,
) -> Py<PyArray1<f64>> {
    let w = py_to_vec(weights);
    let result = mlfinance::sampling::weights::time_decay::time_decay(&w, oldest_weight);
    vec_to_py_array(py, result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_weights, m)?)?;
    m.add_function(wrap_pyfunction!(get_weights_ffd, m)?)?;
    m.add_function(wrap_pyfunction!(frac_diff_ffd, m)?)?;
    m.add_function(wrap_pyfunction!(frac_diff_expanding, m)?)?;
    m.add_function(wrap_pyfunction!(find_min_d, m)?)?;
    m.add_function(wrap_pyfunction!(standard_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(seq_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(compare_bootstraps, m)?)?;
    m.add_function(wrap_pyfunction!(num_co_events, m)?)?;
    m.add_function(wrap_pyfunction!(get_indicator_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(average_uniqueness, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_class_weights, m)?)?;
    m.add_function(wrap_pyfunction!(return_attribution_weights, m)?)?;
    m.add_function(wrap_pyfunction!(time_decay, m)?)?;
    Ok(())
}
