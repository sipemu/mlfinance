#![warn(missing_docs)]

//! Modeling utilities for financial machine learning.
//!
//! This crate provides cross-validation with information-leakage controls,
//! ensemble methods (bagging, boosting, random forests), feature importance
//! measures, and hyperparameter search strategies.
//!
//! # Key modules
//!
//! - [`cross_validation`] -- Purged K-Fold cross-validation with embargo, and
//!   cross-validated scoring.
//! - [`ensemble`] -- Bagging accuracy, AdaBoost weight updates, and random forest
//!   configuration presets.
//! - [`feature_importance`] -- MDI, MDA, SFI, orthogonal (PCA) features, weighted
//!   Kendall tau, and synthetic data generation.
//! - [`hyperparams`] -- Grid search, random search, log-uniform sampling, and
//!   scoring functions.

pub mod cross_validation;
pub mod ensemble;
pub mod feature_importance;
pub mod hyperparams;
