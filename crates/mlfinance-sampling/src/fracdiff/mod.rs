//! Fractional differentiation of time series (Chapter 5).
//!
//! Fractional differentiation allows a non-stationary time series (e.g., prices)
//! to be made stationary while preserving as much memory as possible. This module
//! provides:
//!
//! - [`weights`] -- weight computation for the fractional difference operator.
//! - [`expanding`] -- expanding-window fractional differentiation (Snippet 5.2).
//! - [`ffd`] -- fixed-width window fractional differentiation (Snippet 5.3).
//! - [`min_d`] -- search for the minimum differentiation order `d` that achieves
//!   stationarity.

/// Expanding-window fractional differentiation.
pub mod expanding;
/// Fixed-width window fractional differentiation (FFD).
pub mod ffd;
/// Search for minimum differentiation order for stationarity.
pub mod min_d;
/// Weight computation for the fractional difference operator.
pub mod weights;
