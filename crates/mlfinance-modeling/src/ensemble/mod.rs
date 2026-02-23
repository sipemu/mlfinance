//! Ensemble methods for classification.
//!
//! Provides bagging accuracy estimation, AdaBoost sample-weight updates,
//! and random forest configuration presets.

/// Bagging accuracy estimation for ensemble classifiers.
pub mod bagging;
/// AdaBoost sample-weight updates and meta-labeling.
pub mod boosting;
/// Random forest configuration presets and utilities.
pub mod random_forest;
