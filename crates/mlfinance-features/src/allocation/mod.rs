//! Portfolio allocation methods.
//!
//! Provides Hierarchical Risk Parity (HRP), Critical Line Algorithm (CLA),
//! Inverse Variance Portfolio (IVP), and Monte Carlo comparison of allocation methods.

/// Critical Line Algorithm (CLA) for minimum variance portfolios.
pub mod cla;
/// Hierarchical Risk Parity (HRP) portfolio allocation.
pub mod hrp;
/// Inverse Variance Portfolio (IVP) allocation.
pub mod ivp;
/// Monte Carlo comparison of allocation methods.
pub mod monte_carlo;
