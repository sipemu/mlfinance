#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use pyo3::prelude::*;
use pyo3::types::PyModule;

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

#[pymodule]
fn pymlfinance(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<types::PyTickData>()?;
    m.add_class::<types::PyOhlcvBar>()?;
    m.add_class::<types::PyTripleBarrierConfig>()?;
    m.add_class::<types::PyEvent>()?;
    m.add_class::<types::PyTrendScanResult>()?;
    m.add_class::<types::PyDrawdownResult>()?;
    m.add_class::<types::PyCscvResult>()?;
    m.add_class::<types::PyKMeansResult>()?;
    m.add_class::<types::PyOncResult>()?;
    m.add_class::<types::PyAllocationComparison>()?;
    m.add_class::<types::PyBootstrapComparison>()?;
    m.add_class::<types::PyFoldIndices>()?;

    Ok(())
}
