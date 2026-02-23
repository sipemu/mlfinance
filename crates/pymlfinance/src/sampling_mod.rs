use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;
use crate::types::*;

// ── Fractional differentiation ──────────────────────────────────────────────

/// Compute fractional differentiation weights (AFML Ch. 5).
///
/// Returns the weight vector for a given differentiation order ``d``.
/// Weights decay with lag; the series ``size`` controls truncation.
///
/// Parameters
/// ----------
/// d : float
///     Fractional differentiation order (0 < d < 1 for stationarity).
/// size : int
///     Number of weights to compute.
///
/// Returns
/// -------
/// numpy.ndarray
///     Weight vector of length ``size``.
#[pyfunction]
fn get_weights(py: Python<'_>, d: f64, size: usize) -> Py<PyArray1<f64>> {
    let result = mlfinance::sampling::fracdiff::weights::get_weights(d, size);
    vec_to_py_array(py, result)
}

/// Compute FFD (Fixed-width window Fractional Differentiation) weights.
///
/// Truncates weights below a threshold to create a fixed-width kernel.
///
/// Parameters
/// ----------
/// d : float
///     Fractional differentiation order.
/// threshold : float
///     Minimum absolute weight to keep (e.g. 1e-4).
///
/// Returns
/// -------
/// numpy.ndarray
///     Truncated weight vector.
#[pyfunction]
fn get_weights_ffd(py: Python<'_>, d: f64, threshold: f64) -> Py<PyArray1<f64>> {
    let result = mlfinance::sampling::fracdiff::weights::get_weights_ffd(d, threshold);
    vec_to_py_array(py, result)
}

/// Apply FFD (fixed-width window fractional differentiation) to a series.
///
/// Produces a stationary series that retains memory, using a truncated
/// weight kernel (AFML Ch. 5).
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Input time series (e.g. log prices).
/// d : float
///     Differentiation order (typically 0.3-0.7).
/// threshold : float
///     Weight truncation threshold (e.g. 1e-4).
///
/// Returns
/// -------
/// numpy.ndarray
///     Fractionally differenced series.
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

/// Expanding-window fractional differentiation.
///
/// Uses all available history at each point (no truncation), producing
/// a more accurate but slower computation.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Input time series.
/// d : float
///     Differentiation order.
/// threshold : float
///     Minimum weight for inclusion.
///
/// Returns
/// -------
/// numpy.ndarray
///     Fractionally differenced series.
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

/// Find the minimum fractional differentiation order that makes a series stationary.
///
/// Performs a grid search over d values, applying FFD and testing stationarity
/// with ADF at each step (AFML Ch. 5).
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Input time series.
/// max_d : float
///     Maximum d to test.
/// step_size : float
///     Increment between d values.
/// threshold : float
///     Weight truncation threshold for FFD.
///
/// Returns
/// -------
/// float
///     Minimum d for stationarity (returns max_d if none found).
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

/// Standard IID bootstrap sampling.
///
/// Parameters
/// ----------
/// num_observations : int
///     Total number of observations to sample from.
/// num_samples : int
///     Number of bootstrap samples to draw.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// list[int]
///     Sampled indices (with replacement).
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

/// Sequential bootstrap with uniqueness-aware sampling (AFML Ch. 4).
///
/// Draws samples with probability proportional to their average uniqueness,
/// reducing redundancy from overlapping labels.
///
/// Parameters
/// ----------
/// ind_matrix : numpy.ndarray
///     Indicator matrix (n_events x n_bars) from ``get_indicator_matrix``.
/// num_samples : int
///     Number of samples to draw.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// list[int]
///     Sampled event indices.
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

/// Monte Carlo comparison of sequential vs. standard bootstrap uniqueness.
///
/// Parameters
/// ----------
/// ind_matrix : numpy.ndarray
///     Indicator matrix (n_events x n_bars).
/// num_samples : int
///     Samples per trial.
/// num_trials : int
///     Number of Monte Carlo repetitions.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// BootstrapComparison
///     Average uniqueness for sequential and standard methods.
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

/// Count concurrent events at each bar (AFML Ch. 4).
///
/// Parameters
/// ----------
/// events : list[tuple[int, int]]
///     List of (entry_idx, exit_idx) pairs.
/// num_bars : int
///     Total number of bars in the series.
///
/// Returns
/// -------
/// list[int]
///     Number of active events at each bar index.
#[pyfunction]
fn num_co_events(
    py: Python<'_>,
    events: Vec<(usize, usize)>,
    num_bars: usize,
) -> PyResult<Py<PyAny>> {
    let result = mlfinance::sampling::concurrency::num_co_events::num_co_events(&events, num_bars);
    vec_usize_to_list(py, result)
}

/// Build an indicator matrix mapping events to bars.
///
/// Parameters
/// ----------
/// events : list[tuple[int, int]]
///     List of (entry_idx, exit_idx) pairs.
/// num_bars : int
///     Total number of bars.
///
/// Returns
/// -------
/// numpy.ndarray
///     Binary matrix of shape (n_events, num_bars) where entry (i, j) = 1
///     if event i is active at bar j.
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

/// Compute average uniqueness of each event (AFML Ch. 4).
///
/// Uniqueness at bar j for event i is 1 / (number of concurrent events at j).
/// Average uniqueness is the mean across all bars spanned by the event.
///
/// Parameters
/// ----------
/// events : list[tuple[int, int]]
///     List of (entry_idx, exit_idx) pairs.
/// num_bars : int
///     Total number of bars.
///
/// Returns
/// -------
/// numpy.ndarray
///     Average uniqueness per event.
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

/// Compute balanced class weights inversely proportional to class frequency.
///
/// Parameters
/// ----------
/// labels : list[int]
///     Label vector (e.g. [-1, 0, 1]).
///
/// Returns
/// -------
/// dict[int, float]
///     Mapping from label value to weight.
#[pyfunction]
fn balanced_class_weights(py: Python<'_>, labels: Vec<i32>) -> PyResult<Py<PyAny>> {
    let result = mlfinance::sampling::weights::class_weights::balanced_class_weights(&labels);
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

/// Compute return-attribution sample weights (AFML Ch. 4).
///
/// Weights each event proportionally to its return contribution,
/// adjusted for concurrency.
///
/// Parameters
/// ----------
/// events : list[tuple[int, int]]
///     List of (entry_idx, exit_idx) pairs.
/// returns : numpy.ndarray
///     Per-bar return series.
/// num_bars : int
///     Total number of bars.
///
/// Returns
/// -------
/// numpy.ndarray
///     Sample weights (one per event).
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

/// Apply time-decay to sample weights (AFML Ch. 4).
///
/// Linearly decays weights from 1.0 (most recent) to ``oldest_weight``
/// (least recent).
///
/// Parameters
/// ----------
/// weights : numpy.ndarray
///     Input weights (typically from return attribution).
/// oldest_weight : float
///     Weight for the oldest observation. Use 0 for full linear decay,
///     1 for no decay.
///
/// Returns
/// -------
/// numpy.ndarray
///     Time-decayed weights.
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
