//! Bootstrap sampling methods for financial labels (Chapter 4).
//!
//! Standard IID bootstrap ignores the temporal structure of financial labels,
//! leading to inflated sample uniqueness and overfitting. This module provides:
//!
//! - [`sequential`] -- sequential bootstrap that draws samples weighted by their
//!   average uniqueness, respecting label concurrency (Snippets 4.5--4.6).
//! - [`standard`] -- classical IID bootstrap (uniform sampling with replacement),
//!   useful as a baseline for comparison.
//! - [`monte_carlo`] -- Monte Carlo comparison of sequential vs. standard bootstrap
//!   to quantify the uniqueness improvement (Snippet 4.9).

/// Monte Carlo comparison of sequential vs. standard bootstrap.
pub mod monte_carlo;
/// Sequential bootstrap weighted by average uniqueness.
pub mod sequential;
/// Classical IID bootstrap (uniform sampling with replacement).
pub mod standard;
