//! Multi-product data transformations for portfolio construction.
//!
//! These utilities address the challenges of working with multiple
//! correlated instruments, as described in AFML Chapter 3:
//!
//! - [`etf_trick`] -- Combine multiple price series into a single
//!   continuous ETF-like series using time-varying weights.
//! - [`pca_weights`] -- Compute PCA-based risk-parity portfolio weights
//!   from a covariance matrix.
//! - [`single_future_roll`] -- Compute roll-gap adjustments and build
//!   continuous non-negative price series from rolling futures contracts.

pub mod etf_trick;
pub mod pca_weights;
pub mod single_future_roll;
