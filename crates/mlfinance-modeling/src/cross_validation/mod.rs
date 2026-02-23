//! Cross-validation with information-leakage controls.
//!
//! Provides purged K-Fold splitting (with embargo) and cross-validated scoring
//! to prevent look-ahead bias in financial time series.

pub mod cv_score;
pub mod purged_kfold;
