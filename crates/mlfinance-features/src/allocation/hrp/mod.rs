//! Hierarchical Risk Parity (HRP) portfolio allocation.
//!
//! Implements the full HRP pipeline: correlation distance, single-linkage clustering,
//! quasi-diagonalization, and recursive bisection weight allocation.

#[allow(clippy::module_inception)]
/// Full HRP pipeline implementation.
pub mod hrp;
/// Quasi-diagonalization for reordering assets based on clustering.
pub mod quasi_diag;
/// Recursive bisection weight allocation.
pub mod recursive_bisection;
/// Correlation distance and single-linkage clustering.
pub mod tree_clustering;
