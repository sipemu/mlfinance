//! Unified re-export facade for the mlfinance workspace.
//!
//! This crate provides convenient access to all mlfinance sub-crates through
//! a single dependency. Import individual crates as needed:
//!
//! - [`core`] -- Core types, traits, and utilities
//! - [`parallel`] -- Parallel computation and partitioning
//! - [`data`] -- Alternative bar types and data structures
//! - [`labeling`] -- Triple-barrier labeling and meta-labeling
//! - [`sampling`] -- Bootstrap, fractional differentiation, and sample weights
//! - [`modeling`] -- Cross-validation and feature importance
//! - [`backtesting`] -- Strategy backtesting framework
//! - [`features`] -- Feature engineering, portfolio allocation, and analysis

pub use mlfinance_backtesting as backtesting;
pub use mlfinance_core as core;
pub use mlfinance_data as data;
pub use mlfinance_features as features;
pub use mlfinance_labeling as labeling;
pub use mlfinance_modeling as modeling;
pub use mlfinance_parallel as parallel;
pub use mlfinance_sampling as sampling;
