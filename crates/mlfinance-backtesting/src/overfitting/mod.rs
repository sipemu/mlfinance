//! Backtest overfitting detection (Chapter 11).
//!
//! Tools to quantify how likely a strategy's in-sample performance is
//! the result of overfitting rather than genuine skill:
//!
//! - [`pbo`] -- Probability of Backtest Overfitting using combinatorial
//!   partitioning of the performance matrix.
//! - [`cscv`] -- Combinatorially Symmetric Cross-Validation with rank logit
//!   distribution.
//! - [`multiple_testing`] -- Bonferroni and Holm-Bonferroni corrections for
//!   family-wise error rate control when testing many strategies.

pub mod cscv;
pub mod multiple_testing;
pub mod pbo;
