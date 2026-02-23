#![warn(missing_docs)]

//! Core types, traits, and utilities for the mlfinance workspace.
//!
//! This crate provides the foundational building blocks shared across all mlfinance sub-crates:
//!
//! - **[`types`]** -- Market data primitives ([`OhlcvBar`], [`TickData`], [`Side`], [`Timestamp`]).
//! - **[`error`]** -- Unified error type ([`MlFinanceError`]) used throughout the workspace.
//! - **[`traits`]** -- Extension points for bar aggregation, sampling, weighting, and classification.
//! - **[`series`]** -- Timestamped series container with returns, diffs, and rolling windows.
//! - **[`math`]** -- EWMA, cumulative sums, log/simple returns, and other scalar helpers.
//! - **[`stats`]** -- Descriptive statistics (mean, variance, skewness, kurtosis) and
//!   correlation/covariance matrices.
//! - **[`matrix`]** -- Linear algebra utilities (eigendecomposition via power iteration,
//!   matrix inversion via Gauss-Jordan) that avoid an external BLAS dependency.
//!
//! # Quick start
//!
//! ```rust
//! use mlfinance_core::{OhlcvBar, MlFinanceError};
//! use mlfinance_core::math::ewma;
//! use mlfinance_core::stats::mean;
//! ```

pub mod error;
pub mod math;
pub mod matrix;
pub mod series;
pub mod stats;
pub mod traits;
pub mod types;

pub use error::MlFinanceError;
pub use types::{OhlcvBar, Side, TickData, Timestamp};
