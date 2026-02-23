//! Structural break detection and stationarity tests.
//!
//! Provides Augmented Dickey-Fuller (ADF) unit root tests, Supremum ADF (SADF)
//! and Generalized SADF (GSADF) for bubble detection, Brown-Durbin-Evans and
//! Chu-Stinchcombe-White CUSUM tests, and sub/super-martingale trend tests.

/// Augmented Dickey-Fuller (ADF) unit root test.
pub mod adf;
/// Brown-Durbin-Evans and Chu-Stinchcombe-White CUSUM tests.
pub mod cusum_tests;
/// Generalized Supremum ADF (GSADF) for bubble detection.
pub mod gsadf;
/// Supremum ADF (SADF) for bubble detection.
pub mod sadf;
/// Sub-martingale and super-martingale trend tests.
pub mod sub_super_martingale;
