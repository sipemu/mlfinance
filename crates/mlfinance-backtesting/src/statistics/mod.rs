//! Strategy performance statistics (Chapter 14).
//!
//! A collection of metrics for evaluating strategy quality:
//!
//! - [`sharpe`] -- annualized Sharpe Ratio.
//! - [`psr`] -- Probabilistic Sharpe Ratio, accounting for non-normality.
//! - [`dsr`] -- Deflated Sharpe Ratio, adjusting PSR for multiple testing.
//! - [`drawdown`] -- maximum drawdown, drawdown duration, time under water.
//! - [`hhi`] -- Herfindahl-Hirschman Index for return concentration.
//! - [`general`] -- hit ratio, average holding period, and portfolio turnover.

pub mod drawdown;
pub mod dsr;
pub mod general;
pub mod hhi;
pub mod psr;
pub mod sharpe;
