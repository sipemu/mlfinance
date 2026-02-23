//! Hyperparameter tuning and scoring utilities.
//!
//! Provides grid search, randomised search, log-uniform sampling, and
//! common scoring functions (accuracy, F1, negative log loss).

pub mod grid_search;
pub mod log_uniform;
pub mod random_search;
pub mod scoring;
