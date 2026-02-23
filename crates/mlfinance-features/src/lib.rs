#![warn(missing_docs)]

//! Feature engineering and portfolio allocation for financial machine learning.
//!
//! This crate implements techniques from *Advances in Financial Machine Learning*
//! (Marcos Lopez de Prado) covering Chapters 2-20. It provides tools for:
//!
//! - **Portfolio allocation**: Hierarchical Risk Parity (HRP), Critical Line Algorithm (CLA),
//!   Inverse Variance Portfolio (IVP), and Monte Carlo allocation comparison.
//! - **Clustering**: Optimal Number of Clusters (ONC) using K-Means with silhouette scoring.
//! - **Codependence**: Correlation-based, information-theoretic, and distance-based
//!   dependence measures between time series.
//! - **Denoising**: Random Matrix Theory (RMT) based denoising and detoning of
//!   correlation and covariance matrices.
//! - **Entropy**: Shannon, plug-in, Lempel-Ziv, Kontoyiannis, and Gaussian entropy
//!   estimators with binary, quantile, and sigma encodings.
//! - **Microstructure**: Market microstructure features including Amihud, Kyle, Roll,
//!   Hasbrouck lambda estimators, VPIN, tick rule, Corwin-Schultz spread, and order features.
//! - **Structural breaks**: ADF, SADF, GSADF bubble detection tests, CUSUM tests,
//!   and sub/super-martingale trend tests.
//!
//! # Key types
//!
//! - [`allocation::hrp::hrp::hrp_weights`] -- full HRP pipeline
//! - [`allocation::cla::cla_min_variance`] -- minimum-variance portfolio
//! - [`clustering::onc::OncResult`] -- optimal clustering result
//! - [`codependence::codependence_matrix::DependenceMethod`] -- dependence method selector
//! - [`denoising::rmt::denoise_corr`] -- RMT correlation denoising
//! - [`microstructure::vpin::vpin`] -- Volume-Synchronized Probability of Informed Trading

pub mod allocation;
pub mod clustering;
pub mod codependence;
pub mod denoising;
pub mod entropy;
pub mod microstructure;
pub mod structural_breaks;
