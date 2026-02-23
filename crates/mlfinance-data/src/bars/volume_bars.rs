use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::{OhlcvBar, TickData};

use super::BarBuilder;

/// Aggregates ticks into OHLCV bars when cumulative volume exceeds a threshold.
///
/// A new bar is emitted whenever the accumulated volume reaches or exceeds
/// the configured `volume_threshold`.
pub struct VolumeBarAggregator {
    volume_threshold: f64,
    builder: BarBuilder,
}

impl VolumeBarAggregator {
    /// Create a new `VolumeBarAggregator` with the given volume threshold.
    pub fn new(volume_threshold: f64) -> Self {
        Self {
            volume_threshold,
            builder: BarBuilder::new(),
        }
    }
}

impl BarAggregator for VolumeBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        self.builder.update(tick);

        if self.builder.cumulative_volume() >= self.volume_threshold {
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
    fn test_volume_bars_basic() {
        let mut agg = VolumeBarAggregator::new(100.0);

        assert!(agg.process_tick(&make_tick(50.0, 30.0)).is_none());
        assert!(agg.process_tick(&make_tick(51.0, 30.0)).is_none());

        // cumulative volume = 30 + 30 + 50 = 110 >= 100
        let bar = agg.process_tick(&make_tick(52.0, 50.0));
        assert!(bar.is_some());
        let bar = bar.unwrap();
        assert_eq!(bar.volume, 110.0);
        assert_eq!(bar.open, 50.0);
        assert_eq!(bar.close, 52.0);
    }

    #[test]
    fn test_volume_bars_multiple() {
        let mut agg = VolumeBarAggregator::new(50.0);
        let ticks: Vec<_> = (0..10).map(|i| make_tick(100.0 + i as f64, 20.0)).collect();
        let bars = agg.process_ticks(&ticks);
        // Each bar needs 3 ticks (60 >= 50), so we get 3 bars (9 ticks used, 1 leftover)
        assert_eq!(bars.len(), 3);
    }
}
