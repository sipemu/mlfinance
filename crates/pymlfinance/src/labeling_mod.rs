use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use mlfinance::labeling::events::Event;

use crate::convert::*;
use crate::types::*;

// ── Events & labels ─────────────────────────────────────────────────────────

#[pyfunction]
fn get_events(
    prices: PyReadonlyArray1<'_, f64>,
    entry_indices: Vec<usize>,
    config: &PyTripleBarrierConfig,
    daily_vols: PyReadonlyArray1<'_, f64>,
) -> Vec<PyEvent> {
    let p = py_to_vec(prices);
    let dv = py_to_vec(daily_vols);
    let rust_config = to_rust_bar_config(config);
    let events = mlfinance::labeling::events::get_events(&p, &entry_indices, &rust_config, &dv);
    events.iter().map(from_rust_event).collect()
}

#[pyfunction]
fn get_bins(py: Python<'_>, events: Vec<PyRef<'_, PyEvent>>) -> Py<PyArray1<i32>> {
    let rust_events = py_events_to_rust(&events);
    let bins = mlfinance::labeling::labels::get_bins(&rust_events);
    vec_i32_to_py_array(py, bins)
}

#[pyfunction]
fn get_meta_bins(
    py: Python<'_>,
    events: Vec<PyRef<'_, PyEvent>>,
    primary_predictions: Vec<i32>,
) -> Py<PyArray1<i32>> {
    let rust_events = py_events_to_rust(&events);
    let bins = mlfinance::labeling::labels::get_meta_bins(&rust_events, &primary_predictions);
    vec_i32_to_py_array(py, bins)
}

#[pyfunction]
fn drop_rare_labels(py: Python<'_>, labels: Vec<i32>, min_pct: f64) -> PyResult<Py<PyAny>> {
    let mask = mlfinance::labeling::labels::drop_rare_labels(&labels, min_pct);
    vec_bool_to_list(py, mask)
}

// ── Volatility ──────────────────────────────────────────────────────────────

#[pyfunction]
fn daily_volatility(
    py: Python<'_>,
    prices: PyReadonlyArray1<'_, f64>,
    timestamps: Vec<f64>,
    span: usize,
) -> Py<PyArray1<f64>> {
    let p = py_to_vec(prices);
    let ts: Vec<_> = timestamps.iter().map(|&t| timestamp_from_f64(t)).collect();
    let result = mlfinance::labeling::volatility::daily_volatility(&p, &ts, span);
    vec_to_py_array(py, result)
}

#[pyfunction]
fn parkinson_volatility(
    py: Python<'_>,
    bars: Vec<PyRef<'_, PyOhlcvBar>>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let rust_bars = py_bars_to_rust(&bars);
    let result = to_pyresult(mlfinance::labeling::volatility::parkinson_volatility(
        &rust_bars, window,
    ))?;
    Ok(vec_to_py_array(py, result))
}

#[pyfunction]
fn garman_klass_volatility(
    py: Python<'_>,
    bars: Vec<PyRef<'_, PyOhlcvBar>>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let rust_bars = py_bars_to_rust(&bars);
    let result = to_pyresult(mlfinance::labeling::volatility::garman_klass_volatility(
        &rust_bars, window,
    ))?;
    Ok(vec_to_py_array(py, result))
}

#[pyfunction]
fn yang_zhang_volatility(
    py: Python<'_>,
    bars: Vec<PyRef<'_, PyOhlcvBar>>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let rust_bars = py_bars_to_rust(&bars);
    let result = to_pyresult(mlfinance::labeling::volatility::yang_zhang_volatility(
        &rust_bars, window,
    ))?;
    Ok(vec_to_py_array(py, result))
}

// ── MetaLabeler class ───────────────────────────────────────────────────────

#[pyclass(name = "MetaLabeler")]
pub struct PyMetaLabeler {
    inner: mlfinance::labeling::meta_labeling::MetaLabeler,
}

#[pymethods]
impl PyMetaLabeler {
    #[new]
    fn new(min_probability: f64) -> Self {
        Self {
            inner: mlfinance::labeling::meta_labeling::MetaLabeler::new(min_probability),
        }
    }

    fn generate_labels(
        &self,
        py: Python<'_>,
        events: Vec<PyRef<'_, PyEvent>>,
        primary_predictions: Vec<i32>,
    ) -> Py<PyArray1<i32>> {
        let rust_events = py_events_to_rust(&events);
        let labels = self
            .inner
            .generate_labels(&rust_events, &primary_predictions);
        vec_i32_to_py_array(py, labels)
    }

    fn bet_size(&self, probability: f64) -> f64 {
        self.inner.bet_size(probability)
    }
}

// ── Trend scanning ──────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (prices, t_events=None, max_window=20, min_window=None))]
fn trend_scanning_labels(
    prices: PyReadonlyArray1<'_, f64>,
    t_events: Option<Vec<usize>>,
    max_window: usize,
    min_window: Option<usize>,
) -> PyResult<Vec<PyTrendScanResult>> {
    let p = py_to_vec(prices);
    let events_ref = t_events.as_deref();
    let results = to_pyresult(mlfinance::labeling::trend_scanning::trend_scanning_labels(
        &p, events_ref, max_window, min_window,
    ))?;
    Ok(results.iter().map(from_rust_trend_scan).collect())
}

#[pyfunction]
fn trend_scanning_label_series(
    py: Python<'_>,
    prices: PyReadonlyArray1<'_, f64>,
    max_window: usize,
) -> PyResult<Py<PyArray1<i32>>> {
    let p = py_to_vec(prices);
    let result = to_pyresult(
        mlfinance::labeling::trend_scanning::trend_scanning_label_series(&p, max_window),
    )?;
    Ok(vec_i32_to_py_array(py, result))
}

// ── Vertical barrier ────────────────────────────────────────────────────────

#[pyfunction]
fn add_vertical_barrier(
    py: Python<'_>,
    entry_indices: Vec<usize>,
    max_holding: usize,
    series_len: usize,
) -> PyResult<Py<PyAny>> {
    let result = mlfinance::labeling::vertical_barrier::add_vertical_barrier(
        &entry_indices,
        max_holding,
        series_len,
    );
    vec_usize_to_list(py, result)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn py_events_to_rust(events: &[PyRef<'_, PyEvent>]) -> Vec<Event> {
    use mlfinance::labeling::barriers::BarrierTouchType;
    events
        .iter()
        .map(|e| Event {
            entry_idx: e.entry_idx,
            exit_idx: e.exit_idx,
            touch_type: match e.touch_type.as_str() {
                "upper" => BarrierTouchType::Upper,
                "lower" => BarrierTouchType::Lower,
                _ => BarrierTouchType::Vertical,
            },
            return_value: e.return_value,
        })
        .collect()
}

fn py_bars_to_rust(bars: &[PyRef<'_, PyOhlcvBar>]) -> Vec<mlfinance::core::OhlcvBar> {
    bars.iter()
        .map(|b| mlfinance::core::OhlcvBar {
            timestamp: timestamp_from_f64(b.timestamp),
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            volume: b.volume,
            vwap: b.vwap,
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_events, m)?)?;
    m.add_function(wrap_pyfunction!(get_bins, m)?)?;
    m.add_function(wrap_pyfunction!(get_meta_bins, m)?)?;
    m.add_function(wrap_pyfunction!(drop_rare_labels, m)?)?;
    m.add_function(wrap_pyfunction!(daily_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(parkinson_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(garman_klass_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(yang_zhang_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(trend_scanning_labels, m)?)?;
    m.add_function(wrap_pyfunction!(trend_scanning_label_series, m)?)?;
    m.add_function(wrap_pyfunction!(add_vertical_barrier, m)?)?;
    m.add_class::<PyMetaLabeler>()?;
    Ok(())
}
