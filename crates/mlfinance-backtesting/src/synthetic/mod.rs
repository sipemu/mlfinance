//! Synthetic data generation and visualization (Chapter 13).
//!
//! Tools for generating synthetic financial data and computing
//! performance meshes:
//!
//! - [`ornstein_uhlenbeck`] -- simulation and parameter estimation of the
//!   mean-reverting O-U process.
//! - [`optimal_trading_rule`] -- Monte Carlo mesh of Sharpe ratios across
//!   profit-taking / stop-loss thresholds (Snippets 13.1-13.2).
//! - [`sharpe_mesh`] -- Sharpe ratio heatmaps from a grid of returns.

pub mod optimal_trading_rule;
pub mod ornstein_uhlenbeck;
pub mod sharpe_mesh;
