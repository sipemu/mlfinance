//! Information-driven imbalance bars (Tick, Volume, Dollar).
//!
//! Imbalance bars dynamically set the bar size based on the running imbalance
//! of signed ticks/volumes/dollar amounts. An exponentially weighted moving
//! average (EWMA) of past bar imbalances is used as the expected imbalance.
//! When the absolute running imbalance exceeds this expectation, a bar is emitted.

use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::{OhlcvBar, TickData};

use super::tick_rule::classify_tick;
use super::BarBuilder;

/// EWMA helper that tracks a running exponentially weighted moving average.
#[derive(Debug, Clone)]
struct Ewma {
    span: f64,
    value: Option<f64>,
}

impl Ewma {
    fn new(span: f64) -> Self {
        Self { span, value: None }
    }

    fn alpha(&self) -> f64 {
        2.0 / (self.span + 1.0)
    }

    fn update(&mut self, x: f64) {
        match self.value {
            Some(prev) => {
                let a = self.alpha();
                self.value = Some(a * x + (1.0 - a) * prev);
            }
            None => {
                self.value = Some(x);
            }
        }
    }

    fn get(&self) -> f64 {
        self.value.unwrap_or(0.0)
    }
}

/// Tick Imbalance Bars (TIB).
///
/// Tracks the cumulative signed tick imbalance `theta = sum(b_t)` where
/// `b_t` is the tick rule classification. Emits a bar when `|theta|` exceeds
/// the EWMA of historical expected imbalances.
pub struct TickImbalanceBarAggregator {
    builder: BarBuilder,
    theta: f64,
    prev_price: f64,
    prev_bt: f64,
    ewma_theta: Ewma,
    ewma_tick_count: Ewma,
    initial_expected: f64,
    initialized: bool,
}

impl TickImbalanceBarAggregator {
    /// Create a new TIB aggregator.
    ///
    /// - `initial_expected`: initial expected imbalance threshold before EWMA kicks in
    /// - `ewma_span`: the span for the EWMA window (e.g., 20)
    pub fn new(initial_expected: f64, ewma_span: f64) -> Self {
        Self {
            builder: BarBuilder::new(),
            theta: 0.0,
            prev_price: 0.0,
            prev_bt: 1.0,
            ewma_theta: Ewma::new(ewma_span),
            ewma_tick_count: Ewma::new(ewma_span),
            initial_expected,
            initialized: false,
        }
    }

    fn expected_imbalance(&self) -> f64 {
        if self.ewma_theta.value.is_some() {
            self.ewma_tick_count.get() * self.ewma_theta.get().abs()
        } else {
            self.initial_expected
        }
    }
}

impl BarAggregator for TickImbalanceBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        if !self.initialized {
            self.prev_price = tick.price;
            self.initialized = true;
            self.builder.update(tick);
            return None;
        }

        let bt = classify_tick(tick.price, self.prev_price, self.prev_bt);
        self.prev_price = tick.price;
        self.prev_bt = bt;

        self.theta += bt;
        self.builder.update(tick);

        let expected = self.expected_imbalance();

        if self.theta.abs() >= expected && expected > 0.0 {
            let bar = self.builder.build();

            // Update EWMAs with current bar stats
            self.ewma_theta.update(self.theta.abs());
            self.ewma_tick_count
                .update(self.builder.tick_count() as f64);

            self.builder.reset();
            self.theta = 0.0;

            Some(bar)
        } else {
            None
        }
    }
}

/// Volume Imbalance Bars (VIB).
///
/// Tracks `theta = sum(b_t * v_t)` where `v_t` is the tick volume.
/// Emits a bar when `|theta|` exceeds the EWMA of historical expected imbalances.
pub struct VolumeImbalanceBarAggregator {
    builder: BarBuilder,
    theta: f64,
    prev_price: f64,
    prev_bt: f64,
    ewma_theta: Ewma,
    ewma_tick_count: Ewma,
    initial_expected: f64,
    initialized: bool,
}

impl VolumeImbalanceBarAggregator {
    /// Create a new VIB aggregator.
    ///
    /// # Arguments
    ///
    /// * `initial_expected` - Initial expected volume imbalance threshold before
    ///   the EWMA has accumulated enough history.
    /// * `ewma_span` - Span parameter for the EWMA windows (e.g., 20).
    pub fn new(initial_expected: f64, ewma_span: f64) -> Self {
        Self {
            builder: BarBuilder::new(),
            theta: 0.0,
            prev_price: 0.0,
            prev_bt: 1.0,
            ewma_theta: Ewma::new(ewma_span),
            ewma_tick_count: Ewma::new(ewma_span),
            initial_expected,
            initialized: false,
        }
    }

    fn expected_imbalance(&self) -> f64 {
        if self.ewma_theta.value.is_some() {
            self.ewma_tick_count.get() * self.ewma_theta.get().abs()
        } else {
            self.initial_expected
        }
    }
}

impl BarAggregator for VolumeImbalanceBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        if !self.initialized {
            self.prev_price = tick.price;
            self.initialized = true;
            self.builder.update(tick);
            return None;
        }

        let bt = classify_tick(tick.price, self.prev_price, self.prev_bt);
        self.prev_price = tick.price;
        self.prev_bt = bt;

        self.theta += bt * tick.volume;
        self.builder.update(tick);

        let expected = self.expected_imbalance();

        if self.theta.abs() >= expected && expected > 0.0 {
            let bar = self.builder.build();

            self.ewma_theta.update(self.theta.abs());
            self.ewma_tick_count
                .update(self.builder.tick_count() as f64);

            self.builder.reset();
            self.theta = 0.0;

            Some(bar)
        } else {
            None
        }
    }
}

/// Dollar Imbalance Bars (DIB).
///
/// Tracks `theta = sum(b_t * v_t * p_t)` where `v_t * p_t` is the dollar volume.
/// Emits a bar when `|theta|` exceeds the EWMA of historical expected imbalances.
pub struct DollarImbalanceBarAggregator {
    builder: BarBuilder,
    theta: f64,
    prev_price: f64,
    prev_bt: f64,
    ewma_theta: Ewma,
    ewma_tick_count: Ewma,
    initial_expected: f64,
    initialized: bool,
}

impl DollarImbalanceBarAggregator {
    /// Create a new DIB aggregator.
    ///
    /// # Arguments
    ///
    /// * `initial_expected` - Initial expected dollar imbalance threshold before
    ///   the EWMA has accumulated enough history.
    /// * `ewma_span` - Span parameter for the EWMA windows (e.g., 20).
    pub fn new(initial_expected: f64, ewma_span: f64) -> Self {
        Self {
            builder: BarBuilder::new(),
            theta: 0.0,
            prev_price: 0.0,
            prev_bt: 1.0,
            ewma_theta: Ewma::new(ewma_span),
            ewma_tick_count: Ewma::new(ewma_span),
            initial_expected,
            initialized: false,
        }
    }

    fn expected_imbalance(&self) -> f64 {
        if self.ewma_theta.value.is_some() {
            self.ewma_tick_count.get() * self.ewma_theta.get().abs()
        } else {
            self.initial_expected
        }
    }
}

impl BarAggregator for DollarImbalanceBarAggregator {
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar> {
        if !self.initialized {
            self.prev_price = tick.price;
            self.initialized = true;
            self.builder.update(tick);
            return None;
        }

        let bt = classify_tick(tick.price, self.prev_price, self.prev_bt);
        self.prev_price = tick.price;
        self.prev_bt = bt;

        self.theta += bt * tick.price * tick.volume;
        self.builder.update(tick);

        let expected = self.expected_imbalance();

        if self.theta.abs() >= expected && expected > 0.0 {
            let bar = self.builder.build();

            self.ewma_theta.update(self.theta.abs());
            self.ewma_tick_count
                .update(self.builder.tick_count() as f64);

            self.builder.reset();
            self.theta = 0.0;

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
    fn test_tick_imbalance_bars() {
        // With a low initial expected imbalance, bars should be emitted quickly
        let mut agg = TickImbalanceBarAggregator::new(2.0, 10.0);
        let ticks = vec![
            make_tick(100.0, 1.0),
            make_tick(101.0, 1.0), // bt=+1, theta=1
            make_tick(102.0, 1.0), // bt=+1, theta=2 >= 2.0 -> emit
            make_tick(103.0, 1.0), // bt=+1, theta=1
            make_tick(104.0, 1.0), // bt=+1, theta=2 -> might emit
        ];
        let bars = agg.process_ticks(&ticks);
        assert!(!bars.is_empty(), "Expected at least one bar from TIB");
        assert_eq!(bars[0].open, 100.0);
    }

    #[test]
    fn test_volume_imbalance_bars() {
        let mut agg = VolumeImbalanceBarAggregator::new(50.0, 10.0);
        let ticks = vec![
            make_tick(100.0, 10.0),
            make_tick(101.0, 20.0), // bt=+1, theta=20
            make_tick(102.0, 30.0), // bt=+1, theta=50 >= 50.0 -> emit
        ];
        let bars = agg.process_ticks(&ticks);
        assert!(!bars.is_empty());
    }

    #[test]
    fn test_dollar_imbalance_bars() {
        let mut agg = DollarImbalanceBarAggregator::new(5000.0, 10.0);
        let ticks = vec![
            make_tick(100.0, 10.0),
            make_tick(101.0, 20.0), // bt=+1, theta=101*20=2020
            make_tick(102.0, 30.0), // bt=+1, theta=2020+102*30=5080 >= 5000 -> emit
        ];
        let bars = agg.process_ticks(&ticks);
        assert!(!bars.is_empty());
    }

    #[test]
    fn test_ewma() {
        let mut ewma = Ewma::new(10.0);
        assert_eq!(ewma.get(), 0.0);
        ewma.update(10.0);
        assert_eq!(ewma.get(), 10.0);
        ewma.update(20.0);
        // alpha = 2/11 ~= 0.1818
        let expected = 0.1818181818 * 20.0 + (1.0 - 0.1818181818) * 10.0;
        assert!((ewma.get() - expected).abs() < 1e-6);
    }
}
