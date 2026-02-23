use pyo3::prelude::*;

use crate::convert::{timestamp_from_f64, timestamp_to_f64};
use mlfinance::core::{OhlcvBar, TickData};
use mlfinance::labeling::barriers::{BarrierTouchType, TripleBarrierConfig};
use mlfinance::labeling::events::Event;
use mlfinance::labeling::trend_scanning::TrendScanResult;

// ── Core types ──────────────────────────────────────────────────────────────

/// A single market tick with timestamp, price, and volume.
///
/// Parameters
/// ----------
/// timestamp : float
///     Unix timestamp in seconds (fractional seconds supported).
/// price : float
///     Trade price.
/// volume : float
///     Trade volume.
///
/// Examples
/// --------
/// >>> tick = TickData(1609459200.0, 100.5, 10.0)
/// >>> tick.price
/// 100.5
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

/// An OHLCV bar (Open, High, Low, Close, Volume) with VWAP.
///
/// Parameters
/// ----------
/// timestamp : float
///     Unix timestamp of the bar open.
/// open : float
///     Opening price.
/// high : float
///     Highest price during the bar.
/// low : float
///     Lowest price during the bar.
/// close : float
///     Closing price.
/// volume : float
///     Total volume traded.
/// vwap : float, optional
///     Volume-weighted average price (default 0.0).
///
/// Attributes
/// ----------
/// mid_price() : float
///     (high + low) / 2
/// typical_price() : float
///     (high + low + close) / 3
/// dollar_volume() : float
///     vwap * volume
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

    /// Return the mid price: (high + low) / 2.
    fn mid_price(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Return the typical price: (high + low + close) / 3.
    fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Return the dollar volume: vwap * volume.
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

/// Configuration for the triple-barrier labeling method.
///
/// Defines the upper profit-taking barrier, lower stop-loss barrier,
/// and maximum holding period for event labeling (AFML Ch. 3).
///
/// Parameters
/// ----------
/// upper_barrier : float, optional
///     Profit-taking threshold as a fraction of daily volatility.
/// lower_barrier : float, optional
///     Stop-loss threshold as a fraction of daily volatility.
/// max_holding_period : int, optional
///     Maximum number of bars before forced exit (vertical barrier).
///
/// Examples
/// --------
/// >>> config = TripleBarrierConfig(upper_barrier=2.0, lower_barrier=2.0, max_holding_period=10)
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

/// A labeled event from the triple-barrier method.
///
/// Attributes
/// ----------
/// entry_idx : int
///     Index of the entry bar.
/// exit_idx : int
///     Index of the exit bar.
/// touch_type : str
///     Which barrier was touched first: ``"upper"``, ``"lower"``, or ``"vertical"``.
/// return_value : float
///     Return from entry to exit.
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

/// Result of trend scanning label detection.
///
/// Attributes
/// ----------
/// t_stat : float
///     t-statistic of the best linear fit.
/// label : int
///     Trend direction: +1 (up), -1 (down), or 0 (no trend).
/// best_window : int
///     Look-ahead window that produced the highest |t-stat|.
/// r_squared : float
///     R-squared of the best linear fit.
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

/// Result of drawdown analysis on a return series.
///
/// Attributes
/// ----------
/// max_drawdown : float
///     Maximum peak-to-trough decline.
/// max_drawdown_duration : int
///     Longest drawdown duration in bars.
/// drawdown_series : list[float]
///     Per-bar drawdown values.
/// time_under_water : list[int]
///     Per-bar count of consecutive bars in drawdown.
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

/// Result of Combinatorially Symmetric Cross-Validation (CSCV).
///
/// Attributes
/// ----------
/// pbo : float
///     Probability of Backtest Overfitting.
/// rank_logits : list[float]
///     Log-odds of each strategy's rank degradation.
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

/// Result of K-means clustering.
///
/// Attributes
/// ----------
/// labels : list[int]
///     Cluster assignment for each data point.
/// n_iterations : int
///     Number of iterations until convergence.
/// centroids : list[list[float]]
///     Cluster centroids (k x n_features).
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

/// Result of Optimal Number of Clusters (ONC) analysis.
///
/// Attributes
/// ----------
/// labels : list[int]
///     Cluster assignment for each feature/variable.
/// silhouette : float
///     Silhouette score of the optimal clustering.
/// n_clusters : int
///     Optimal number of clusters found.
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

/// Comparison of HRP, CLA, and IVP portfolio allocations.
///
/// Attributes
/// ----------
/// hrp_sharpe : float
///     Out-of-sample Sharpe ratio for HRP.
/// cla_sharpe : float
///     Out-of-sample Sharpe ratio for CLA (min-variance).
/// ivp_sharpe : float
///     Out-of-sample Sharpe ratio for Inverse Variance.
/// hrp_variance : float
///     Out-of-sample portfolio variance for HRP.
/// cla_variance : float
///     Out-of-sample portfolio variance for CLA.
/// ivp_variance : float
///     Out-of-sample portfolio variance for IVP.
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

/// Comparison of sequential vs. standard bootstrap uniqueness.
///
/// Attributes
/// ----------
/// seq_uniqueness : float
///     Average uniqueness from sequential bootstrap.
/// std_uniqueness : float
///     Average uniqueness from standard (IID) bootstrap.
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

/// Train/test split indices for a single cross-validation fold.
///
/// Attributes
/// ----------
/// train : list[int]
///     Indices of training samples.
/// test : list[int]
///     Indices of test samples.
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
