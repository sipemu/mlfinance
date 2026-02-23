#![warn(missing_docs)]

//! Sampling, weighting, and fractional differentiation for financial machine learning.
//!
//! This crate implements techniques from *Advances in Financial Machine Learning*
//! (Marcos Lopez de Prado), covering chapters 4 and 5:
//!
//! - **Chapter 4 -- Sample Weights**: concurrent label counting, average uniqueness,
//!   sequential bootstrap, return-attribution weights, time-decay weighting, and
//!   balanced class weights.
//! - **Chapter 5 -- Fractional Differentiation**: expanding-window and fixed-width-window
//!   (FFD) fractional differentiation, weight computation, and minimum-d search for
//!   stationarity.
//!
//! # Key modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`bootstrap`] | Sequential, standard IID, and Monte Carlo bootstrap comparison |
//! | [`concurrency`] | Indicator matrix, co-event counting, average uniqueness |
//! | [`fracdiff`] | Fractional differentiation (expanding, FFD, min-d search) |
//! | [`weights`] | Sample weight schemes (class, return-attribution, time decay) |

pub mod bootstrap;
pub mod concurrency;
pub mod fracdiff;
pub mod weights;
