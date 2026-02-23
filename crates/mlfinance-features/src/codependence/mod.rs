//! Codependence and distance measures between time series.
//!
//! Provides correlation-based measures (angular, distance correlation, Kullback-Leibler),
//! information-theoretic measures (mutual information, variation of information),
//! rank-based measures (GNPR distance), optimal transport dependence, and
//! utilities for building pairwise dependence and distance matrices.

/// Utilities for building pairwise dependence and distance matrices.
pub mod codependence_matrix;
/// Correlation-based distance measures (angular, distance correlation, KL divergence).
pub mod correlation;
/// GNPR distance for rank-based dependence.
pub mod gnpr_distance;
/// Information-theoretic measures (mutual information, variation of information).
pub mod information;
/// Optimal transport dependence measure.
pub mod optimal_transport;
