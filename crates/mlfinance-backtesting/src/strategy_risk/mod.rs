//! Strategy risk decomposition (Chapter 15).
//!
//! Relates the Sharpe Ratio to a strategy's precision (hit rate),
//! betting frequency, and win/loss ratio, and provides inverse
//! computations:
//!
//! - [`sr_from_precision`] -- expected SR from precision and frequency
//!   (Snippet 15.1).
//! - [`implied_precision`] -- minimum precision for a target SR
//!   (Snippet 15.3).
//! - [`implied_frequency`] -- minimum betting frequency for a target SR
//!   (Snippet 15.4).
//! - [`failure_probability`] -- probability that true precision falls below
//!   break-even (Snippet 15.5).

pub mod failure_probability;
pub mod implied_frequency;
pub mod implied_precision;
pub mod sr_from_precision;
