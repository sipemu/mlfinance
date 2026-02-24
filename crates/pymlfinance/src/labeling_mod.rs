use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use mlfinance::labeling::events::Event;

use crate::convert::*;
use crate::types::*;

// ── Events & labels ─────────────────────────────────────────────────────────

/// Generate triple-barrier labeled events from a price series (AFML Ch. 3).
///
/// For each entry index, finds the first barrier touch (upper profit-take,
/// lower stop-loss, or vertical max-holding) and records the event.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series.
/// entry_indices : list[int]
///     Indices at which to enter positions.
/// config : TripleBarrierConfig
///     Barrier configuration (upper, lower, max holding period).
/// daily_vols : numpy.ndarray
///     Daily volatility estimates (same length as prices), used to scale barriers.
///
/// Returns
/// -------
/// list[Event]
///     Labeled events with entry/exit indices, touch type, and return.
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

/// Convert events to directional labels (-1, 0, +1).
///
/// Parameters
/// ----------
/// events : list[Event]
///     Labeled events from ``get_events``.
///
/// Returns
/// -------
/// numpy.ndarray[int32]
///     Labels: +1 (upper touch), -1 (lower touch), 0 (vertical).
#[pyfunction]
fn get_bins(py: Python<'_>, events: Vec<PyRef<'_, PyEvent>>) -> Py<PyArray1<i32>> {
    let rust_events = py_events_to_rust(&events);
    let bins = mlfinance::labeling::labels::get_bins(&rust_events);
    vec_i32_to_py_array(py, bins)
}

/// Generate meta-labels that correct a primary model's predictions (AFML Ch. 3).
///
/// A meta-label is 1 if the primary prediction's direction matches the
/// actual outcome, 0 otherwise.
///
/// Parameters
/// ----------
/// events : list[Event]
///     Labeled events from ``get_events``.
/// primary_predictions : list[int]
///     Primary model's directional predictions (+1 or -1).
///
/// Returns
/// -------
/// numpy.ndarray[int32]
///     Meta-labels (0 or 1).
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

/// Generate a boolean mask to drop labels below a minimum frequency.
///
/// Parameters
/// ----------
/// labels : list[int]
///     Label vector.
/// min_pct : float
///     Minimum fraction (0 to 1) a label must represent to be kept.
///
/// Returns
/// -------
/// list[bool]
///     Mask where True means the sample's label is frequent enough.
#[pyfunction]
fn drop_rare_labels(py: Python<'_>, labels: Vec<i32>, min_pct: f64) -> PyResult<Py<PyAny>> {
    let mask = mlfinance::labeling::labels::drop_rare_labels(&labels, min_pct);
    vec_bool_to_list(py, mask)
}

// ── Volatility ──────────────────────────────────────────────────────────────

/// Compute daily volatility as the EWMA standard deviation of returns.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series.
/// timestamps : list[float]
///     Unix timestamps corresponding to prices.
/// span : int
///     EWMA span for volatility estimation.
///
/// Returns
/// -------
/// numpy.ndarray
///     Daily volatility estimates.
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

/// Parkinson volatility estimator using high-low range.
///
/// More efficient than close-to-close volatility since it uses
/// intra-bar price range information.
///
/// Parameters
/// ----------
/// bars : list[OhlcvBar]
///     OHLCV bars.
/// window : int
///     Rolling window size.
///
/// Returns
/// -------
/// numpy.ndarray
///     Rolling Parkinson volatility estimates.
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

/// Garman-Klass volatility estimator using OHLC prices.
///
/// Uses open, high, low, and close prices for a more efficient estimate
/// than Parkinson (accounts for opening jumps).
///
/// Parameters
/// ----------
/// bars : list[OhlcvBar]
///     OHLCV bars.
/// window : int
///     Rolling window size.
///
/// Returns
/// -------
/// numpy.ndarray
///     Rolling Garman-Klass volatility estimates.
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

/// Yang-Zhang volatility estimator.
///
/// Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
/// components for a minimum-variance, drift-independent estimator.
///
/// Parameters
/// ----------
/// bars : list[OhlcvBar]
///     OHLCV bars.
/// window : int
///     Rolling window size.
///
/// Returns
/// -------
/// numpy.ndarray
///     Rolling Yang-Zhang volatility estimates.
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

/// Meta-labeling model for bet sizing (AFML Ch. 3).
///
/// Wraps a primary model's predictions: generates meta-labels that
/// indicate whether to act on the primary signal, and sizes bets
/// based on prediction probability.
///
/// Parameters
/// ----------
/// min_probability : float
///     Minimum probability threshold for a positive meta-label.
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

    /// Generate binary meta-labels from events and primary predictions.
    ///
    /// Parameters
    /// ----------
    /// events : list[Event]
    ///     Labeled events.
    /// primary_predictions : list[int]
    ///     Primary model's directional predictions (+1 or -1).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[int32]
    ///     Meta-labels (0 = do not trade, 1 = trade).
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

    /// Compute bet size from a predicted probability.
    ///
    /// Parameters
    /// ----------
    /// probability : float
    ///     Predicted probability of the positive class.
    ///
    /// Returns
    /// -------
    /// float
    ///     Bet size in [0, 1]. Returns 0 if below ``min_probability``.
    fn bet_size(&self, probability: f64) -> f64 {
        self.inner.bet_size(probability)
    }
}

// ── Trend scanning ──────────────────────────────────────────────────────────

/// Trend scanning labels using t-statistic regression (AFML Ch. 3.5).
///
/// For each event index, fits linear regressions over multiple forward-looking
/// windows and selects the window with the highest |t-statistic|.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series.
/// t_events : list[int], optional
///     Indices at which to compute labels. If None, uses all indices.
/// max_window : int, default 20
///     Maximum look-ahead window.
/// min_window : int, optional
///     Minimum look-ahead window (default: 3).
///
/// Returns
/// -------
/// list[TrendScanResult]
///     Per-event results with t-stat, label, best window, and R-squared.
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

/// Trend scanning label series (+1, -1, 0) for an entire price series.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series.
/// max_window : int
///     Maximum look-ahead window.
///
/// Returns
/// -------
/// numpy.ndarray[int32]
///     Labels for each index where a label could be computed.
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

/// Add vertical barriers (maximum holding period) to entry indices.
///
/// Parameters
/// ----------
/// entry_indices : list[int]
///     Entry bar indices.
/// max_holding : int
///     Maximum holding period in bars.
/// series_len : int
///     Total length of the price series (for clamping).
///
/// Returns
/// -------
/// list[int]
///     Exit indices corresponding to each entry (clamped to series_len - 1).
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
    register_functions!(
        m,
        get_events,
        get_bins,
        get_meta_bins,
        drop_rare_labels,
        daily_volatility,
        parkinson_volatility,
        garman_klass_volatility,
        yang_zhang_volatility,
        trend_scanning_labels,
        trend_scanning_label_series,
        add_vertical_barrier,
    );
    register_classes!(m, PyMetaLabeler);
    Ok(())
}
