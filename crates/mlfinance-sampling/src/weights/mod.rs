//! Sample weight computation for financial labels (Chapter 4).
//!
//! Proper sample weighting is critical for training ML models on financial data
//! where labels overlap in time. This module provides three complementary schemes:
//!
//! - [`class_weights`] -- balanced class weights inversely proportional to class
//!   frequency, analogous to scikit-learn's `class_weight="balanced"`.
//! - [`return_attribution`] -- weights proportional to uniqueness times absolute
//!   return, ensuring that samples with higher information content receive more
//!   weight (Snippet 4.10).
//! - [`time_decay`] -- piecewise-linear time decay that down-weights older samples
//!   relative to newer ones (Snippet 4.11).

/// Balanced class weights inversely proportional to class frequency.
pub mod class_weights;
/// Weights proportional to uniqueness times absolute return.
pub mod return_attribution;
/// Piecewise-linear time decay for down-weighting older samples.
pub mod time_decay;
