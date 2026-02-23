use pyo3::prelude::*;

use crate::convert::{timestamp_from_f64, timestamp_to_f64};
use mlfinance::core::{OhlcvBar, TickData};
use mlfinance::labeling::barriers::{BarrierTouchType, TripleBarrierConfig};
use mlfinance::labeling::events::Event;
use mlfinance::labeling::trend_scanning::TrendScanResult;

// ── Core types ──────────────────────────────────────────────────────────────

#[pyclass(name = "TickData", from_py_object)]
#[derive(Clone)]
pub struct PyTickData {
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub volume: f64,
}

#[pymethods]
impl PyTickData {
    #[new]
    fn new(timestamp: f64, price: f64, volume: f64) -> Self {
        Self {
            timestamp,
            price,
            volume,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TickData(timestamp={}, price={}, volume={})",
            self.timestamp, self.price, self.volume
        )
    }
}

#[pyclass(name = "OhlcvBar", from_py_object)]
#[derive(Clone)]
pub struct PyOhlcvBar {
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub open: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub vwap: f64,
}

#[pymethods]
impl PyOhlcvBar {
    #[new]
    #[pyo3(signature = (timestamp, open, high, low, close, volume, vwap=None))]
    fn new(
        timestamp: f64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        vwap: Option<f64>,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            vwap: vwap.unwrap_or(0.0),
        }
    }

    fn mid_price(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    fn dollar_volume(&self) -> f64 {
        self.vwap * self.volume
    }

    fn __repr__(&self) -> String {
        format!(
            "OhlcvBar(ts={}, o={}, h={}, l={}, c={}, v={}, vwap={})",
            self.timestamp, self.open, self.high, self.low, self.close, self.volume, self.vwap
        )
    }
}

// ── Labeling types ──────────────────────────────────────────────────────────

#[pyclass(name = "TripleBarrierConfig", from_py_object)]
#[derive(Clone)]
pub struct PyTripleBarrierConfig {
    #[pyo3(get)]
    pub upper_barrier: Option<f64>,
    #[pyo3(get)]
    pub lower_barrier: Option<f64>,
    #[pyo3(get)]
    pub max_holding_period: Option<usize>,
}

#[pymethods]
impl PyTripleBarrierConfig {
    #[new]
    #[pyo3(signature = (upper_barrier=None, lower_barrier=None, max_holding_period=None))]
    fn new(
        upper_barrier: Option<f64>,
        lower_barrier: Option<f64>,
        max_holding_period: Option<usize>,
    ) -> Self {
        Self {
            upper_barrier,
            lower_barrier,
            max_holding_period,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TripleBarrierConfig(upper={:?}, lower={:?}, max_hold={:?})",
            self.upper_barrier, self.lower_barrier, self.max_holding_period
        )
    }
}

#[pyclass(name = "Event", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyEvent {
    #[pyo3(get)]
    pub entry_idx: usize,
    #[pyo3(get)]
    pub exit_idx: usize,
    #[pyo3(get)]
    pub touch_type: String,
    #[pyo3(get)]
    pub return_value: f64,
}

#[pymethods]
impl PyEvent {
    fn __repr__(&self) -> String {
        format!(
            "Event(entry={}, exit={}, touch={}, ret={:.6})",
            self.entry_idx, self.exit_idx, self.touch_type, self.return_value
        )
    }
}

#[pyclass(name = "TrendScanResult", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTrendScanResult {
    #[pyo3(get)]
    pub t_stat: f64,
    #[pyo3(get)]
    pub label: i32,
    #[pyo3(get)]
    pub best_window: usize,
    #[pyo3(get)]
    pub r_squared: f64,
}

#[pymethods]
impl PyTrendScanResult {
    fn __repr__(&self) -> String {
        format!(
            "TrendScanResult(t={:.4}, label={}, window={}, r2={:.4})",
            self.t_stat, self.label, self.best_window, self.r_squared
        )
    }
}

// ── Backtesting types ───────────────────────────────────────────────────────

#[pyclass(name = "DrawdownResult", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyDrawdownResult {
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub max_drawdown_duration: usize,
    #[pyo3(get)]
    pub drawdown_series: Vec<f64>,
    #[pyo3(get)]
    pub time_under_water: Vec<usize>,
}

#[pymethods]
impl PyDrawdownResult {
    fn __repr__(&self) -> String {
        format!(
            "DrawdownResult(max_dd={:.4}, max_dd_dur={})",
            self.max_drawdown, self.max_drawdown_duration
        )
    }
}

#[pyclass(name = "CscvResult", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCscvResult {
    #[pyo3(get)]
    pub pbo: f64,
    #[pyo3(get)]
    pub rank_logits: Vec<f64>,
}

#[pymethods]
impl PyCscvResult {
    fn __repr__(&self) -> String {
        format!("CscvResult(pbo={:.4})", self.pbo)
    }
}

// ── Features types ──────────────────────────────────────────────────────────

#[pyclass(name = "KMeansResult", frozen)]
pub struct PyKMeansResult {
    #[pyo3(get)]
    pub labels: Vec<usize>,
    #[pyo3(get)]
    pub n_iterations: usize,
    pub(crate) centroids_data: Vec<Vec<f64>>,
}

#[pymethods]
impl PyKMeansResult {
    #[getter]
    fn centroids(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self
            .centroids_data
            .clone()
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "KMeansResult(k={}, iters={})",
            self.centroids_data.len(),
            self.n_iterations
        )
    }
}

#[pyclass(name = "OncResult", frozen)]
pub struct PyOncResult {
    #[pyo3(get)]
    pub labels: Vec<usize>,
    #[pyo3(get)]
    pub silhouette: f64,
    #[pyo3(get)]
    pub n_clusters: usize,
}

#[pymethods]
impl PyOncResult {
    fn __repr__(&self) -> String {
        format!(
            "OncResult(k={}, sil={:.4})",
            self.n_clusters, self.silhouette
        )
    }
}

#[pyclass(name = "AllocationComparison", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyAllocationComparison {
    #[pyo3(get)]
    pub hrp_sharpe: f64,
    #[pyo3(get)]
    pub cla_sharpe: f64,
    #[pyo3(get)]
    pub ivp_sharpe: f64,
    #[pyo3(get)]
    pub hrp_variance: f64,
    #[pyo3(get)]
    pub cla_variance: f64,
    #[pyo3(get)]
    pub ivp_variance: f64,
}

#[pymethods]
impl PyAllocationComparison {
    fn __repr__(&self) -> String {
        format!(
            "AllocationComparison(hrp_sr={:.4}, cla_sr={:.4}, ivp_sr={:.4})",
            self.hrp_sharpe, self.cla_sharpe, self.ivp_sharpe
        )
    }
}

#[pyclass(name = "BootstrapComparison", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyBootstrapComparison {
    #[pyo3(get)]
    pub seq_uniqueness: f64,
    #[pyo3(get)]
    pub std_uniqueness: f64,
}

#[pymethods]
impl PyBootstrapComparison {
    fn __repr__(&self) -> String {
        format!(
            "BootstrapComparison(seq={:.4}, std={:.4})",
            self.seq_uniqueness, self.std_uniqueness
        )
    }
}

// ── Sampling types ──────────────────────────────────────────────────────────

#[pyclass(name = "FoldIndices", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFoldIndices {
    #[pyo3(get)]
    pub train: Vec<usize>,
    #[pyo3(get)]
    pub test: Vec<usize>,
}

#[pymethods]
impl PyFoldIndices {
    fn __repr__(&self) -> String {
        format!(
            "FoldIndices(train_len={}, test_len={})",
            self.train.len(),
            self.test.len()
        )
    }
}

// ── Enums as string converters ──────────────────────────────────────────────

pub fn barrier_touch_type_to_str(t: &BarrierTouchType) -> &'static str {
    match t {
        BarrierTouchType::Upper => "upper",
        BarrierTouchType::Lower => "lower",
        BarrierTouchType::Vertical => "vertical",
    }
}

// ── Conversion functions ────────────────────────────────────────────────────

pub fn to_rust_tick(py_tick: &PyTickData) -> TickData {
    TickData {
        timestamp: timestamp_from_f64(py_tick.timestamp),
        price: py_tick.price,
        volume: py_tick.volume,
    }
}

pub fn to_rust_bar_config(cfg: &PyTripleBarrierConfig) -> TripleBarrierConfig {
    TripleBarrierConfig {
        upper_barrier: cfg.upper_barrier,
        lower_barrier: cfg.lower_barrier,
        max_holding_period: cfg.max_holding_period,
    }
}

pub fn from_rust_event(event: &Event) -> PyEvent {
    PyEvent {
        entry_idx: event.entry_idx,
        exit_idx: event.exit_idx,
        touch_type: barrier_touch_type_to_str(&event.touch_type).to_string(),
        return_value: event.return_value,
    }
}

pub fn from_rust_bar(bar: &OhlcvBar) -> PyOhlcvBar {
    PyOhlcvBar {
        timestamp: timestamp_to_f64(bar.timestamp),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume,
        vwap: bar.vwap,
    }
}

pub fn from_rust_trend_scan(r: &TrendScanResult) -> PyTrendScanResult {
    PyTrendScanResult {
        t_stat: r.t_stat,
        label: r.label,
        best_window: r.best_window,
        r_squared: r.r_squared,
    }
}
