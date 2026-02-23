use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::{OhlcvBar, TickData};

use super::BarBuilder;

/// Aggregates ticks into OHLCV bars every N ticks.
///
/// A new bar is emitted after exactly `bar_size` ticks have been accumulated.
pub struct TickBarAggregator {
    bar_size: usize,
    builder: BarBuilder,
}

impl TickBarAggregator {
    /// Create a new `TickBarAggregator` that emits a bar every `bar_size` ticks.
    pub fn new(bar_size: usize) -> Self {
        Self {
            bar_size,
            builder: BarBuilder::new(),
        }
    }
}

impl BarAggregator for TickBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        self.builder.update(tick);

        if self.builder.tick_count() >= self.bar_size {
            let bar = self.builder.build();
            self.builder.reset();
            Some(bar)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn make_tick(price: f64, volume: f64) -> TickData {
        TickData {
            timestamp: Utc.timestamp_opt(0, 0).unwrap(),
            price,
            volume,
        }
    }

    #[test]
    fn test_tick_bars_basic() {
        let mut agg = TickBarAggregator::new(3);

        assert!(agg.process_tick(&make_tick(100.0, 1.0)).is_none());
        assert!(agg.process_tick(&make_tick(101.0, 1.0)).is_none());

        let bar = agg.process_tick(&make_tick(99.0, 1.0));
        assert!(bar.is_some());
        let bar = bar.unwrap();
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 101.0);
        assert_eq!(bar.low, 99.0);
        assert_eq!(bar.close, 99.0);
        assert_eq!(bar.volume, 3.0);
    }

    #[test]
    fn test_tick_bars_multiple() {
        let mut agg = TickBarAggregator::new(2);
        let ticks: Vec<_> = (0..6).map(|i| make_tick(100.0 + i as f64, 1.0)).collect();
        let bars = agg.process_ticks(&ticks);
        assert_eq!(bars.len(), 3);
    }

    #[test]
    fn test_tick_bars_vwap() {
        let mut agg = TickBarAggregator::new(2);
        agg.process_tick(&make_tick(100.0, 10.0));
        let bar = agg.process_tick(&make_tick(200.0, 10.0)).unwrap();
        // VWAP = (100*10 + 200*10) / 20 = 150
        assert!((bar.vwap - 150.0).abs() < 1e-10);
    }
}
