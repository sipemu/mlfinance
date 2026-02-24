#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Register multiple `#[pyfunction]`s on a module in one call.
macro_rules! register_functions {
    ($m:expr, $($func:ident),* $(,)?) => {
        $(
            $m.add_function(pyo3::wrap_pyfunction!($func, $m)?)?;
        )*
    };
}

/// Register multiple `#[pyclass]` types on a module in one call.
macro_rules! register_classes {
    ($m:expr, $($class:ty),* $(,)?) => {
        $(
            $m.add_class::<$class>()?;
        )*
    };
}

mod backtesting_mod;
mod classifier;
mod convert;
mod core_mod;
mod data_mod;
mod features_mod;
mod labeling_mod;
mod modeling_mod;
mod sampling_mod;
mod types;

// Polars expression plugin modules
mod polars_backtesting_exprs;
mod polars_convert;
mod polars_core_exprs;
mod polars_features_exprs;
mod polars_kwargs;
mod polars_labeling_exprs;
mod polars_microstructure_exprs;
mod polars_sampling_exprs;
mod polars_volatility_exprs;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core: math, stats, matrix
    let core = PyModule::new(m.py(), "core")?;
    core_mod::register(&core)?;
    m.add_submodule(&core)?;

    // Data: bars, sampling, multi_product
    let data = PyModule::new(m.py(), "data")?;
    data_mod::register(&data)?;
    m.add_submodule(&data)?;

    // Labeling: barriers, events, labels, volatility, meta_labeling, trend_scanning
    let labeling = PyModule::new(m.py(), "labeling")?;
    labeling_mod::register(&labeling)?;
    m.add_submodule(&labeling)?;

    // Sampling: fracdiff, bootstrap, concurrency, weights
    let sampling = PyModule::new(m.py(), "sampling")?;
    sampling_mod::register(&sampling)?;
    m.add_submodule(&sampling)?;

    // Features: structural_breaks, entropy, microstructure, denoising, allocation, clustering, codependence
    let features = PyModule::new(m.py(), "features")?;
    features_mod::register(&features)?;
    m.add_submodule(&features)?;

    // Modeling: cross_validation, feature_importance, hyperparams, ensemble, scoring
    let modeling = PyModule::new(m.py(), "modeling")?;
    modeling_mod::register(&modeling)?;
    m.add_submodule(&modeling)?;

    // Backtesting: statistics, overfitting, strategy_risk, bet_sizing, synthetic
    let backtesting = PyModule::new(m.py(), "backtesting")?;
    backtesting_mod::register(&backtesting)?;
    m.add_submodule(&backtesting)?;

    // Register shared types at the root level too
    register_classes!(
        m,
        types::PyTickData,
        types::PyOhlcvBar,
        types::PyTripleBarrierConfig,
        types::PyEvent,
        types::PyTrendScanResult,
        types::PyDrawdownResult,
        types::PyCscvResult,
        types::PyKMeansResult,
        types::PyOncResult,
        types::PyAllocationComparison,
        types::PyBootstrapComparison,
        types::PyFoldIndices,
    );

    Ok(())
}
