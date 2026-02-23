//! Combinatorial Purged Cross-Validation (CPCV) -- Chapter 12.
//!
//! Standard k-fold cross-validation produces a single backtest path.
//! CPCV generates all C(N, k) train/test splits, yielding many more
//! independent paths for statistical analysis of backtest results.
//!
//! - [`combinatorial`] -- configuration and split generation.
//! - [`path_builder`] -- assembling complete backtest paths from splits.

pub mod combinatorial;
pub mod path_builder;
