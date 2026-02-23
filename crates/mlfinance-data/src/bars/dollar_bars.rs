use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::{OhlcvBar, TickData};

use super::BarBuilder;

/// Aggregates ticks into OHLCV bars when cumulative dollar volume exceeds a threshold.
///
/// Dollar volume for each tick is computed as `price * volume`. A new bar is
/// emitted whenever the accumulated dollar volume reaches or exceeds the threshold.
pub struct DollarBarAggregator {
    dollar_threshold: f64,
    builder: BarBuilder,
}

impl DollarBarAggregator {
    /// Create a new `DollarBarAggregator` with the given dollar volume threshold.
    pub fn new(dollar_threshold: f64) -> Self {
        Self {
            dollar_threshold,
            builder: BarBuilder::new(),
        }
    }
}

impl BarAggregator for DollarBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        self.builder.update(tick);

        if self.builder.cumulative_dollar_volume() >= self.dollar_threshold {
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
    fn test_dollar_bars_basic() {
        let mut agg = DollarBarAggregator::new(10_000.0);

        // 100 * 30 = 3000
        assert!(agg.process_tick(&make_tick(100.0, 30.0)).is_none());
        // 3000 + 100 * 40 = 7000
        assert!(agg.process_tick(&make_tick(100.0, 40.0)).is_none());
        // 7000 + 100 * 50 = 12000 >= 10000
        let bar = agg.process_tick(&make_tick(100.0, 50.0));
        assert!(bar.is_some());
        let bar = bar.unwrap();
        assert_eq!(bar.volume, 120.0);
    }

    #[test]
    fn test_dollar_bars_price_weighted() {
        let mut agg = DollarBarAggregator::new(500.0);
        // 200 * 1 = 200
        assert!(agg.process_tick(&make_tick(200.0, 1.0)).is_none());
        // 200 + 300 * 1 = 500 >= 500
        let bar = agg.process_tick(&make_tick(300.0, 1.0));
        assert!(bar.is_some());
    }
}
