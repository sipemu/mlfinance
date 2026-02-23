#![warn(missing_docs)]

//! Labeling methods for financial machine learning.
//!
//! This crate implements the labeling techniques from *Advances in Financial
//! Machine Learning* by Marcos Lopez de Prado, primarily Chapters 3 and 5.
//! Labels transform raw price data into supervised-learning targets that
//! reflect the outcome of a trading strategy.
//!
//! # Key concepts
//!
//! - **Triple-barrier method** ([`barriers`]): defines profit-taking, stop-loss,
//!   and time-expiry barriers around each trade entry. The label is determined
//!   by which barrier is touched first.
//! - **Events** ([`events`]): pairs each entry signal with its first barrier
//!   touch, producing an entry/exit record with the realized return.
//! - **Labels** ([`labels`]): converts events into discrete class labels
//!   (`{-1, 0, 1}`) or meta-labels (`{0, 1}`).
//! - **Meta-labeling** ([`meta_labeling`]): trains a secondary model to decide
//!   whether to act on a primary model's signal, enabling bet sizing.
//! - **Trend scanning** ([`trend_scanning`]): a model-free labeling method that
//!   fits linear regressions over multiple forward windows and selects the
//!   window with the highest absolute t-statistic.
//! - **Volatility estimation** ([`volatility`]): EWMA, Parkinson, Garman-Klass,
//!   and Yang-Zhang estimators used to set barrier widths and normalize returns.
//! - **Vertical barriers** ([`vertical_barrier`]): standalone helper for
//!   computing time-expiry exit indices.
//! - **Drop labels** ([`drop_labels`]): utilities for removing rare label
//!   classes to mitigate class imbalance.
//!
//! # Book chapters covered
//!
//! | Module            | AFML reference           |
//! |-------------------|--------------------------|
//! | `volatility`      | Snippet 3.1              |
//! | `barriers`        | Snippet 3.2 - 3.3        |
//! | `events`          | Snippet 3.2 - 3.3        |
//! | `labels`          | Snippet 3.5 - 3.7        |
//! | `drop_labels`     | Snippet 3.8              |
//! | `meta_labeling`   | Chapter 3.6              |
//! | `trend_scanning`  | Chapter 3.5 / 5.4        |

pub mod barriers;
pub mod drop_labels;
pub mod events;
pub mod labels;
pub mod meta_labeling;
pub mod trend_scanning;
pub mod vertical_barrier;
pub mod volatility;
