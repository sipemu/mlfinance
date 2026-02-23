use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use mlfinance::core::traits::BarAggregator;
use mlfinance::data::bars::{
    dollar_bars::DollarBarAggregator, imbalance_bars::*, runs_bars::*,
    tick_bars::TickBarAggregator, time_bars::TimeBarAggregator, volume_bars::VolumeBarAggregator,
};

use crate::convert::*;
use crate::types::*;

// ── Macro for bar aggregator wrappers (single-arg threshold) ────────────────

macro_rules! py_bar_aggregator_single {
    ($py_name:ident, $py_str:literal, $rust_type:ty, $arg_name:ident, $arg_type:ty, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(name = $py_str)]
        pub struct $py_name {
            inner: $rust_type,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new($arg_name: $arg_type) -> Self {
                Self {
                    inner: <$rust_type>::new($arg_name),
                }
            }

            /// Process a single tick and return a completed bar, if any.
            ///
            /// Parameters
            /// ----------
            /// tick : TickData
            ///     A single market tick.
            ///
            /// Returns
            /// -------
            /// OhlcvBar or None
            ///     A completed bar if the threshold was reached, otherwise None.
            fn process_tick(&mut self, tick: &PyTickData) -> Option<PyOhlcvBar> {
                let rust_tick = to_rust_tick(tick);
                self.inner
                    .process_tick(&rust_tick)
                    .map(|b| from_rust_bar(&b))
            }

            /// Process a batch of ticks and return all completed bars.
            ///
            /// Parameters
            /// ----------
            /// ticks : list[TickData]
            ///     Sequence of market ticks.
            ///
            /// Returns
            /// -------
            /// list[OhlcvBar]
            ///     All bars completed during the batch.
            fn process_ticks(&mut self, ticks: Vec<PyRef<'_, PyTickData>>) -> Vec<PyOhlcvBar> {
                let rust_ticks: Vec<_> = ticks.iter().map(|t| to_rust_tick(t)).collect();
                self.inner
                    .process_ticks(&rust_ticks)
                    .iter()
                    .map(from_rust_bar)
                    .collect()
            }
        }
    };
}

// ── Macro for bar aggregator wrappers (two-arg: expected + ewma_span) ───────

macro_rules! py_bar_aggregator_dual {
    ($py_name:ident, $py_str:literal, $rust_type:ty, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(name = $py_str)]
        pub struct $py_name {
            inner: $rust_type,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new(initial_expected: f64, ewma_span: f64) -> Self {
                Self {
                    inner: <$rust_type>::new(initial_expected, ewma_span),
                }
            }

            /// Process a single tick and return a completed bar, if any.
            fn process_tick(&mut self, tick: &PyTickData) -> Option<PyOhlcvBar> {
                let rust_tick = to_rust_tick(tick);
                self.inner
                    .process_tick(&rust_tick)
                    .map(|b| from_rust_bar(&b))
            }

            /// Process a batch of ticks and return all completed bars.
            fn process_ticks(&mut self, ticks: Vec<PyRef<'_, PyTickData>>) -> Vec<PyOhlcvBar> {
                let rust_ticks: Vec<_> = ticks.iter().map(|t| to_rust_tick(t)).collect();
                self.inner
                    .process_ticks(&rust_ticks)
                    .iter()
                    .map(from_rust_bar)
                    .collect()
            }
        }
    };
}

// ── Bar aggregator types ────────────────────────────────────────────────────

py_bar_aggregator_single!(
    PyTickBarAggregator,
    "TickBarAggregator",
    TickBarAggregator,
    bar_size,
    usize,
    "Aggregate ticks into bars with a fixed number of ticks per bar.\n\nParameters\n----------\nbar_size : int\n    Number of ticks per bar."
);
py_bar_aggregator_single!(
    PyVolumeBarAggregator,
    "VolumeBarAggregator",
    VolumeBarAggregator,
    volume_threshold,
    f64,
    "Aggregate ticks into bars when cumulative volume reaches a threshold.\n\nParameters\n----------\nvolume_threshold : float\n    Volume threshold per bar."
);
py_bar_aggregator_single!(
    PyDollarBarAggregator,
    "DollarBarAggregator",
    DollarBarAggregator,
    dollar_threshold,
    f64,
    "Aggregate ticks into bars when cumulative dollar volume reaches a threshold.\n\nParameters\n----------\ndollar_threshold : float\n    Dollar volume threshold per bar."
);

/// Aggregate ticks into bars at fixed time intervals.
///
/// Parameters
/// ----------
/// interval_seconds : int
///     Bar duration in seconds.
#[pyclass(name = "TimeBarAggregator")]
pub struct PyTimeBarAggregator {
    inner: TimeBarAggregator,
}

#[pymethods]
impl PyTimeBarAggregator {
    #[new]
    fn new(interval_seconds: i64) -> Self {
        Self {
            inner: TimeBarAggregator::new(chrono::Duration::seconds(interval_seconds)),
        }
    }

    /// Process a single tick and return a completed bar, if any.
    fn process_tick(&mut self, tick: &PyTickData) -> Option<PyOhlcvBar> {
        let rust_tick = to_rust_tick(tick);
        self.inner
            .process_tick(&rust_tick)
            .map(|b| from_rust_bar(&b))
    }

    /// Process a batch of ticks and return all completed bars.
    fn process_ticks(&mut self, ticks: Vec<PyRef<'_, PyTickData>>) -> Vec<PyOhlcvBar> {
        let rust_ticks: Vec<_> = ticks.iter().map(|t| to_rust_tick(t)).collect();
        self.inner
            .process_ticks(&rust_ticks)
            .iter()
            .map(from_rust_bar)
            .collect()
    }
}

// Imbalance bars
py_bar_aggregator_dual!(
    PyTickImbalanceBarAggregator,
    "TickImbalanceBarAggregator",
    TickImbalanceBarAggregator,
    "Tick imbalance bars (TIB) — sample when tick direction imbalance exceeds an EWMA threshold.\n\nParameters\n----------\ninitial_expected : float\n    Initial expected imbalance.\newma_span : float\n    EWMA decay span for threshold adaptation."
);
py_bar_aggregator_dual!(
    PyVolumeImbalanceBarAggregator,
    "VolumeImbalanceBarAggregator",
    VolumeImbalanceBarAggregator,
    "Volume imbalance bars (VIB) — sample when signed volume imbalance exceeds an EWMA threshold.\n\nParameters\n----------\ninitial_expected : float\n    Initial expected imbalance.\newma_span : float\n    EWMA decay span for threshold adaptation."
);
py_bar_aggregator_dual!(
    PyDollarImbalanceBarAggregator,
    "DollarImbalanceBarAggregator",
    DollarImbalanceBarAggregator,
    "Dollar imbalance bars (DIB) — sample when signed dollar volume imbalance exceeds an EWMA threshold.\n\nParameters\n----------\ninitial_expected : float\n    Initial expected imbalance.\newma_span : float\n    EWMA decay span for threshold adaptation."
);

// Runs bars
py_bar_aggregator_dual!(
    PyTickRunsBarAggregator,
    "TickRunsBarAggregator",
    TickRunsBarAggregator,
    "Tick runs bars — sample when the longest run of same-sign ticks exceeds an EWMA threshold.\n\nParameters\n----------\ninitial_expected : float\n    Initial expected run length.\newma_span : float\n    EWMA decay span for threshold adaptation."
);
py_bar_aggregator_dual!(
    PyVolumeRunsBarAggregator,
    "VolumeRunsBarAggregator",
    VolumeRunsBarAggregator,
    "Volume runs bars — sample when volume of the dominant direction exceeds an EWMA threshold.\n\nParameters\n----------\ninitial_expected : float\n    Initial expected run volume.\newma_span : float\n    EWMA decay span for threshold adaptation."
);
py_bar_aggregator_dual!(
    PyDollarRunsBarAggregator,
    "DollarRunsBarAggregator",
    DollarRunsBarAggregator,
    "Dollar runs bars — sample when dollar volume of the dominant direction exceeds an EWMA threshold.\n\nParameters\n----------\ninitial_expected : float\n    Initial expected run dollar volume.\newma_span : float\n    EWMA decay span for threshold adaptation."
);

// ── Standalone data functions ───────────────────────────────────────────────

/// CUSUM event filter for detecting structural shifts (AFML Ch. 2).
///
/// Detects indices where the cumulative sum of deviations from the
/// running mean exceeds a symmetric threshold.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input series (e.g. log returns or price differences).
/// threshold : float
///     Symmetric threshold for positive and negative CUSUM.
///
/// Returns
/// -------
/// list[int]
///     Indices where CUSUM events are detected.
#[pyfunction]
fn cusum_filter(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    threshold: f64,
) -> PyResult<Py<PyAny>> {
    let v = py_to_vec(values);
    let indices = mlfinance::data::sampling::cusum_filter::cusum_filter(&v, threshold);
    vec_usize_to_list(py, indices)
}

/// Sample n evenly spaced indices in a range.
///
/// Parameters
/// ----------
/// start : int
///     Start index (inclusive).
/// end : int
///     End index (exclusive).
/// n : int
///     Number of samples.
///
/// Returns
/// -------
/// list[int]
///     Evenly spaced indices.
#[pyfunction]
fn linspace_sample(py: Python<'_>, start: usize, end: usize, n: usize) -> PyResult<Py<PyAny>> {
    let indices = mlfinance::data::sampling::event_sampling::linspace_sample(start, end, n);
    vec_usize_to_list(py, indices)
}

/// Sample n random indices from a range.
///
/// Parameters
/// ----------
/// n : int
///     Number of samples.
/// total : int
///     Upper bound of the range (exclusive).
/// seed : int
///     Random seed for reproducibility.
///
/// Returns
/// -------
/// list[int]
///     Randomly sampled indices (sorted, with replacement).
#[pyfunction]
fn uniform_sample(py: Python<'_>, n: usize, total: usize, seed: u64) -> PyResult<Py<PyAny>> {
    let indices = mlfinance::data::sampling::event_sampling::uniform_sample(n, total, seed);
    vec_usize_to_list(py, indices)
}

/// ETF trick for combining multiple product series into a single tradeable index.
///
/// Parameters
/// ----------
/// prices : list[list[float]]
///     Per-product price series (products x time steps).
/// weights : list[list[float]]
///     Per-product allocation weights (products x time steps).
///
/// Returns
/// -------
/// numpy.ndarray
///     Synthetic ETF price series.
#[pyfunction]
fn etf_trick(py: Python<'_>, prices: Vec<Vec<f64>>, weights: Vec<Vec<f64>>) -> Py<PyArray1<f64>> {
    let result = mlfinance::data::multi_product::etf_trick::etf_trick(&prices, &weights);
    vec_to_py_array(py, result)
}

/// PCA-based portfolio weights from a covariance matrix.
///
/// Allocates risk proportionally to principal components. Optionally
/// targets a specific risk distribution.
///
/// Parameters
/// ----------
/// cov_matrix : numpy.ndarray
///     Covariance matrix (n x n).
/// risk_target : float, optional
///     Target risk fraction for the first component. If None, uses
///     equal risk allocation across all components.
///
/// Returns
/// -------
/// numpy.ndarray
///     Portfolio weights (length n, sums to 1).
#[pyfunction]
#[pyo3(signature = (cov_matrix, risk_target=None))]
fn pca_weights(
    py: Python<'_>,
    cov_matrix: PyReadonlyArray2<'_, f64>,
    risk_target: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let m = py_to_array2(cov_matrix);
    let result = to_pyresult(mlfinance::data::multi_product::pca_weights::pca_weights(
        &m,
        risk_target,
    ))?;
    Ok(array1_to_py(py, result))
}

/// Compute roll gaps for a single-future continuous series.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Raw futures prices.
/// roll_dates : list[int]
///     Indices where contract rolls occur.
///
/// Returns
/// -------
/// numpy.ndarray
///     Cumulative roll gap adjustments.
#[pyfunction]
fn roll_gaps(
    py: Python<'_>,
    prices: PyReadonlyArray1<'_, f64>,
    roll_dates: Vec<usize>,
) -> Py<PyArray1<f64>> {
    let p = py_to_vec(prices);
    let result = mlfinance::data::multi_product::single_future_roll::roll_gaps(&p, &roll_dates);
    vec_to_py_array(py, result)
}

/// Build a non-negative rolled price series by adjusting for roll gaps.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Raw futures prices.
/// roll_dates : list[int]
///     Indices where contract rolls occur.
///
/// Returns
/// -------
/// numpy.ndarray
///     Adjusted non-negative price series.
#[pyfunction]
fn non_negative_rolled(
    py: Python<'_>,
    prices: PyReadonlyArray1<'_, f64>,
    roll_dates: Vec<usize>,
) -> Py<PyArray1<f64>> {
    let p = py_to_vec(prices);
    let result =
        mlfinance::data::multi_product::single_future_roll::non_negative_rolled(&p, &roll_dates);
    vec_to_py_array(py, result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTickBarAggregator>()?;
    m.add_class::<PyVolumeBarAggregator>()?;
    m.add_class::<PyDollarBarAggregator>()?;
    m.add_class::<PyTimeBarAggregator>()?;
    m.add_class::<PyTickImbalanceBarAggregator>()?;
    m.add_class::<PyVolumeImbalanceBarAggregator>()?;
    m.add_class::<PyDollarImbalanceBarAggregator>()?;
    m.add_class::<PyTickRunsBarAggregator>()?;
    m.add_class::<PyVolumeRunsBarAggregator>()?;
    m.add_class::<PyDollarRunsBarAggregator>()?;
    m.add_function(wrap_pyfunction!(cusum_filter, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_sample, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_sample, m)?)?;
    m.add_function(wrap_pyfunction!(etf_trick, m)?)?;
    m.add_function(wrap_pyfunction!(pca_weights, m)?)?;
    m.add_function(wrap_pyfunction!(roll_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(non_negative_rolled, m)?)?;
    Ok(())
}
