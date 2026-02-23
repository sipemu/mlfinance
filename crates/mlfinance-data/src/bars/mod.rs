//! Alternative bar types for converting raw tick data into OHLCV bars.
//!
//! Standard time bars sample at fixed clock intervals, which produces bars
//! with non-stationary statistical properties. This module provides
//! information-driven alternatives that sample according to market activity:
//!
//! - [`time_bars`] -- Fixed-interval time bars (the traditional approach).
//! - [`tick_bars`] -- Bars emitted every N ticks.
//! - [`volume_bars`] -- Bars emitted when cumulative volume crosses a threshold.
//! - [`dollar_bars`] -- Bars emitted when cumulative dollar volume crosses a threshold.
//! - [`imbalance_bars`] -- Tick/volume/dollar imbalance bars driven by
//!   signed-order-flow imbalance (AFML Chapter 2).
//! - [`runs_bars`] -- Tick/volume/dollar runs bars driven by the longest
//!   buy or sell run (AFML Chapter 2).
//! - [`tick_rule`] -- The tick rule for classifying trades as buyer- or
//!   seller-initiated.
//!
//! All aggregator types implement the [`mlfinance_core::traits::BarAggregator`]
//! trait, which provides a streaming `process_tick` interface.

/// Dollar bars (emit when cumulative dollar volume exceeds threshold).
pub mod dollar_bars;
/// Imbalance bars driven by signed-order-flow imbalance.
pub mod imbalance_bars;
/// Runs bars driven by the longest buy or sell run.
pub mod runs_bars;
/// Tick-count bars (emit every N ticks).
pub mod tick_bars;
/// The tick rule for classifying trades as buyer- or seller-initiated.
pub mod tick_rule;
/// Fixed-interval time bars.
pub mod time_bars;
/// Volume bars (emit when cumulative volume exceeds threshold).
pub mod volume_bars;

use mlfinance_core::types::{OhlcvBar, TickData, Timestamp};

/// Accumulates tick data and builds an OhlcvBar when finalized.
#[derive(Debug, Clone)]
pub(crate) struct BarBuilder {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    dollar_volume: f64,
    timestamp: Timestamp,
    tick_count: usize,
}

impl BarBuilder {
    pub fn new() -> Self {
        Self {
            open: 0.0,
            high: f64::NEG_INFINITY,
            low: f64::INFINITY,
            close: 0.0,
            volume: 0.0,
            dollar_volume: 0.0,
            timestamp: chrono::DateTime::<chrono::Utc>::MIN_UTC,
            tick_count: 0,
        }
    }

    pub fn update(&mut self, tick: &TickData) {
        if self.tick_count == 0 {
            self.open = tick.price;
            self.timestamp = tick.timestamp;
        }
        if tick.price > self.high {
            self.high = tick.price;
        }
        if tick.price < self.low {
            self.low = tick.price;
        }
        self.close = tick.price;
        self.volume += tick.volume;
        self.dollar_volume += tick.price * tick.volume;
        self.tick_count += 1;
    }

    pub fn build(&self) -> OhlcvBar {
        let vwap = if self.volume > 0.0 {
            self.dollar_volume / self.volume
        } else {
            self.close
        };
        OhlcvBar {
            timestamp: self.timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            vwap,
        }
    }

    pub fn reset(&mut self) {
        self.open = 0.0;
        self.high = f64::NEG_INFINITY;
        self.low = f64::INFINITY;
        self.close = 0.0;
        self.volume = 0.0;
        self.dollar_volume = 0.0;
        self.timestamp = chrono::DateTime::<chrono::Utc>::MIN_UTC;
        self.tick_count = 0;
    }

    pub fn tick_count(&self) -> usize {
        self.tick_count
    }

    pub fn cumulative_volume(&self) -> f64 {
        self.volume
    }

    pub fn cumulative_dollar_volume(&self) -> f64 {
        self.dollar_volume
    }

    pub fn is_empty(&self) -> bool {
        self.tick_count == 0
    }
}
