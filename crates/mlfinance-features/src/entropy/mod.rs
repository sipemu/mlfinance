//! Entropy estimators and encoding schemes for financial time series.
//!
//! Provides Shannon, plug-in, Lempel-Ziv, Kontoyiannis, and Gaussian entropy
//! estimators, along with binary, quantile, and sigma encoding schemes for
//! discretizing continuous data.

/// Encoding schemes for discretizing continuous data (binary, quantile, sigma).
pub mod encoding;
/// Gaussian entropy estimator.
pub mod gaussian_entropy;
/// Kontoyiannis entropy estimator.
pub mod kontoyiannis;
/// Lempel-Ziv entropy estimator.
pub mod lempel_ziv;
/// Plug-in entropy estimator.
pub mod plugin;
/// Shannon entropy estimator.
pub mod shannon;
