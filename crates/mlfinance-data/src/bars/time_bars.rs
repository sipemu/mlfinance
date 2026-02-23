use chrono::Duration;
use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::{OhlcvBar, TickData, Timestamp};

use super::BarBuilder;

/// Aggregates ticks into OHLCV bars at fixed time intervals.
///
/// A new bar is emitted whenever a tick arrives whose timestamp exceeds
/// the current bar's start time plus the configured interval.
pub struct TimeBarAggregator {
    interval: Duration,
    builder: BarBuilder,
    bar_end: Option<Timestamp>,
}

impl TimeBarAggregator {
    /// Create a new `TimeBarAggregator` with the given time interval.
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            builder: BarBuilder::new(),
            bar_end: None,
        }
    }
}

impl BarAggregator for TimeBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        // Initialize bar_end on first tick
        if self.bar_end.is_none() {
            self.bar_end = Some(tick.timestamp + self.interval);
            self.builder.update(tick);
            return None;
        }

        let end = self.bar_end.unwrap();

        if tick.timestamp >= end {
            // Emit the current bar
            let bar = if !self.builder.is_empty() {
                Some(self.builder.build())
            } else {
                None
            };

            // Reset and start a new bar
            self.builder.reset();
            self.builder.update(tick);
            self.bar_end = Some(tick.timestamp + self.interval);

            bar
        } else {
            self.builder.update(tick);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn make_tick(secs: i64, price: f64, volume: f64) -> TickData {
        TickData {
            timestamp: Utc.timestamp_opt(secs, 0).unwrap(),
            price,
            volume,
        }
    }

    #[test]
    fn test_time_bars_basic() {
        let mut agg = TimeBarAggregator::new(Duration::seconds(10));

        // Ticks within first 10-second window
        assert!(agg.process_tick(&make_tick(0, 100.0, 10.0)).is_none());
        assert!(agg.process_tick(&make_tick(3, 101.0, 5.0)).is_none());
        assert!(agg.process_tick(&make_tick(7, 99.0, 8.0)).is_none());

        // This tick crosses the boundary, emits the bar
        let bar = agg.process_tick(&make_tick(10, 102.0, 12.0));
        assert!(bar.is_some());
        let bar = bar.unwrap();
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 101.0);
        assert_eq!(bar.low, 99.0);
        assert_eq!(bar.close, 99.0);
        assert_eq!(bar.volume, 23.0);
    }

    #[test]
    fn test_time_bars_multiple() {
        let mut agg = TimeBarAggregator::new(Duration::seconds(5));
        let ticks = vec![
            make_tick(0, 100.0, 1.0),
            make_tick(2, 101.0, 1.0),
            make_tick(5, 102.0, 1.0),
            make_tick(7, 103.0, 1.0),
            make_tick(10, 104.0, 1.0),
        ];
        let bars = agg.process_ticks(&ticks);
        assert_eq!(bars.len(), 2);
    }
}
