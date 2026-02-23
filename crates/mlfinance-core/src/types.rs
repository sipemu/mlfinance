//! Market data primitives used throughout the mlfinance workspace.
//!
//! This module defines the canonical representations for ticks, OHLCV bars,
//! trade sides, and timestamps.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// UTC timestamp used for all time-indexed data.
pub type Timestamp = DateTime<Utc>;

/// A single market tick containing price, volume, and a timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    /// The point in time at which this tick was observed.
    pub timestamp: Timestamp,
    /// The trade or quote price.
    pub price: f64,
    /// The traded volume (shares, contracts, or base-currency units).
    pub volume: f64,
}

/// An OHLCV bar with an additional volume-weighted average price (VWAP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    /// The opening timestamp of the bar.
    pub timestamp: Timestamp,
    /// Opening price.
    pub open: f64,
    /// Highest price during the bar.
    pub high: f64,
    /// Lowest price during the bar.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Total volume traded during the bar.
    pub volume: f64,
    /// Volume-weighted average price over the bar.
    pub vwap: f64,
}

/// Direction of a trade or position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    /// Long / buy direction.
    Buy,
    /// Short / sell direction.
    Sell,
}

impl Side {
    /// Return the numeric sign for this side: `+1.0` for [`Buy`](Side::Buy),
    /// `-1.0` for [`Sell`](Side::Sell).
    pub fn sign(&self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
}

impl OhlcvBar {
    /// Compute the mid-price as `(high + low) / 2`.
    pub fn mid_price(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Compute the typical price as `(high + low + close) / 3`.
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Compute the dollar volume as `vwap * volume`.
    pub fn dollar_volume(&self) -> f64 {
        self.vwap * self.volume
    }
}
