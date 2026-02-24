#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use pyo3::prelude::*;

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

/// Create a submodule, register its contents, and attach it to a parent module.
macro_rules! register_submodule {
    ($parent:expr, $name:expr, $register_fn:path) => {{
        let sub = pyo3::types::PyModule::new($parent.py(), $name)?;
        $register_fn(&sub)?;
        $parent.add_submodule(&sub)?;
    }};
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
    // Submodules
    register_submodule!(m, "core", core_mod::register);
    register_submodule!(m, "data", data_mod::register);
    register_submodule!(m, "labeling", labeling_mod::register);
    register_submodule!(m, "sampling", sampling_mod::register);
    register_submodule!(m, "features", features_mod::register);
    register_submodule!(m, "modeling", modeling_mod::register);
    register_submodule!(m, "backtesting", backtesting_mod::register);

    // Shared types at the root level
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
