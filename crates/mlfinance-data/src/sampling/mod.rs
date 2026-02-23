//! Event-driven sampling and structural-break detection.
//!
//! This module provides filters and sampling utilities for selecting
//! meaningful observations from a time series, rather than sampling at
//! fixed intervals.
//!
//! - [`cusum_filter`] -- Symmetric CUSUM filter for detecting structural
//!   breaks (AFML Snippet 2.4).
//! - [`event_sampling`] -- Deterministic linspace and pseudo-random uniform
//!   index sampling for reproducible experiments.

pub mod cusum_filter;
pub mod event_sampling;
