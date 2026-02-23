//! Label concurrency analysis (Chapter 4).
//!
//! Financial labels typically span multiple bars, causing overlapping events that
//! share information. This module quantifies that overlap:
//!
//! - [`num_co_events`] -- count the number of labels active at each bar.
//! - [`indicator_matrix`] -- build a binary indicator matrix `I[t, i]` for events
//!   over bars.
//! - [`average_uniqueness`] -- compute the average uniqueness of each label,
//!   defined as the mean of `1 / c_t` over the bars the label spans, where `c_t`
//!   is the concurrency count.

/// Average uniqueness computation for overlapping labels.
pub mod average_uniqueness;
/// Binary indicator matrix construction for label concurrency.
pub mod indicator_matrix;
/// Count the number of concurrent labels at each bar.
pub mod num_co_events;
