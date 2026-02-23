#![warn(missing_docs)]

//! Financial data structures and transformations for machine learning.
//!
//! This crate implements the data-preparation techniques from
//! *Advances in Financial Machine Learning* (de Prado), covering material
//! from Chapters 2 and 3.
//!
//! # Key modules
//!
//! - [`bars`] -- Alternative bar types (time, tick, volume, dollar, imbalance,
//!   runs) that transform raw tick data into OHLCV bars with desirable
//!   statistical properties (Chapter 2).
//! - [`sampling`] -- Event-driven sampling, including the symmetric CUSUM
//!   filter for structural-break detection (Chapter 2, Snippet 2.4).
//! - [`multi_product`] -- Multi-product helpers: the ETF trick for combining
//!   multiple price series, PCA-based risk-parity weights, and single-future
//!   roll adjustments (Chapter 3).

pub mod bars;
pub mod multi_product;
pub mod sampling;
