//! Market microstructure features and estimators.
//!
//! Provides liquidity, spread, and price impact estimators including Amihud illiquidity,
//! Kyle's lambda, Roll spread, Hasbrouck's lambda, VPIN, tick rule classification,
//! Corwin-Schultz spread, and order size distribution features.

/// Amihud illiquidity measure.
pub mod amihud_lambda;
/// Corwin-Schultz spread estimator.
pub mod corwin_schultz;
/// Hasbrouck's lambda (price impact measure).
pub mod hasbrouck_lambda;
/// Kyle's lambda (price impact from volume).
pub mod kyle_lambda;
/// Order size distribution features.
pub mod order_features;
/// Roll spread model for bid-ask spread estimation.
pub mod roll_model;
/// Tick rule classification for trades.
pub mod tick_rule;
/// VPIN (Volume-Synchronized Probability of Informed Trading).
pub mod vpin;
