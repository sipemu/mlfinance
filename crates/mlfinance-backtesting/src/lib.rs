#![warn(missing_docs)]

//! Backtesting infrastructure for quantitative finance strategies.
//!
//! This crate implements backtesting tools from *Advances in Financial Machine
//! Learning* (Chapters 10, 12-15) and related literature. It covers:
//!
//! - **Bet sizing** -- mapping classifier probabilities to position sizes,
//!   averaging overlapping signals, discretizing signals, and dynamic
//!   sizing with limit prices (Chapter 10).
//! - **Combinatorial Purged Cross-Validation (CPCV)** -- generating all
//!   C(N, k) train/test splits and assembling complete backtest paths
//!   (Chapter 12).
//! - **Overfitting detection** -- Probability of Backtest Overfitting (PBO),
//!   Combinatorially Symmetric Cross-Validation (CSCV), and multiple testing
//!   corrections (Chapter 11).
//! - **Strategy statistics** -- Sharpe ratio, drawdowns, hit ratio, holding
//!   period, turnover, HHI concentration, Probabilistic Sharpe Ratio (PSR),
//!   and Deflated Sharpe Ratio (DSR) (Chapters 14).
//! - **Strategy risk** -- deriving Sharpe from precision/frequency, implied
//!   precision, implied frequency, and strategy failure probability
//!   (Chapter 15).
//! - **Synthetic data** -- Ornstein-Uhlenbeck simulation/estimation, optimal
//!   trading rule meshes, and Sharpe ratio meshes for visualization
//!   (Chapter 13).
//!
//! # Key types
//!
//! | Module | Key type / function |
//! |---|---|
//! | [`bet_sizing`] | [`bet_sizing::probability_to_size::sigmoid_bet_size`], [`bet_sizing::dynamic_sizing::DynamicBetSize`] |
//! | [`cpcv`] | [`cpcv::combinatorial::CpcvConfig`], [`cpcv::path_builder::build_paths`] |
//! | [`overfitting`] | [`overfitting::pbo::probability_of_backtest_overfitting`], [`overfitting::cscv::cscv`] |
//! | [`statistics`] | [`statistics::sharpe::sharpe_ratio`], [`statistics::psr::probabilistic_sharpe_ratio`], [`statistics::dsr::deflated_sharpe_ratio`] |
//! | [`strategy_risk`] | [`strategy_risk::sr_from_precision::sr_from_precision`], [`strategy_risk::failure_probability::strategy_failure_probability`] |
//! | [`synthetic`] | [`synthetic::ornstein_uhlenbeck::simulate_ou`], [`synthetic::optimal_trading_rule::otr_mesh`] |

pub mod bet_sizing;
pub mod cpcv;
pub mod overfitting;
pub mod statistics;
pub mod strategy_risk;
pub mod synthetic;
