//! Bet sizing: converting model signals into position sizes (Chapter 10).
//!
//! Provides several approaches to bet sizing:
//!
//! - [`probability_to_size`] -- sigmoid and power mappings from predicted
//!   probabilities to bet sizes (Snippet 10.1).
//! - [`active_bets`] -- averaging overlapping signals at each bar
//!   (Snippet 10.2).
//! - [`discretization`] -- rounding continuous signals to discrete steps
//!   (Snippet 10.3).
//! - [`dynamic_sizing`] -- computing limit prices and trade sizes that
//!   account for the current position (Snippet 10.4).

pub mod active_bets;
pub mod discretization;
pub mod dynamic_sizing;
pub mod probability_to_size;
