//! Information-driven runs bars (Tick, Volume, Dollar).
//!
//! Runs bars are similar to imbalance bars but track the longest run of
//! consecutive buy or sell ticks. A bar is emitted when the maximum of the
//! buy-run or sell-run metrics exceeds the EWMA expectation.

use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::{OhlcvBar, TickData};

use super::tick_rule::classify_tick;
use super::BarBuilder;

/// EWMA helper for runs bars.
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

/// Tick Runs Bars (TRB).
///
/// Tracks the number of buy ticks and sell ticks separately, then
/// emits a bar when `max(buy_count, sell_count)` exceeds the EWMA expectation.
pub struct TickRunsBarAggregator {
    builder: BarBuilder,
    buy_count: f64,
    sell_count: f64,
    prev_price: f64,
    prev_bt: f64,
    ewma_buy: Ewma,
    ewma_sell: Ewma,
    ewma_tick_count: Ewma,
    initial_expected: f64,
    initialized: bool,
}

impl TickRunsBarAggregator {
    /// Create a new TRB aggregator.
    ///
    /// # Arguments
    ///
    /// * `initial_expected` - Initial expected run-length threshold before
    ///   the EWMA has accumulated enough history.
    /// * `ewma_span` - Span parameter for the EWMA windows (e.g., 20).
    pub fn new(initial_expected: f64, ewma_span: f64) -> Self {
        Self {
            builder: BarBuilder::new(),
            buy_count: 0.0,
            sell_count: 0.0,
            prev_price: 0.0,
            prev_bt: 1.0,
            ewma_buy: Ewma::new(ewma_span),
            ewma_sell: Ewma::new(ewma_span),
            ewma_tick_count: Ewma::new(ewma_span),
            initial_expected,
            initialized: false,
        }
    }

    fn expected_runs(&self) -> f64 {
        if self.ewma_buy.value.is_some() {
            let p_buy = self.ewma_buy.get();
            let p_sell = self.ewma_sell.get();
            let max_p = p_buy.max(p_sell);
            self.ewma_tick_count.get() * max_p
        } else {
            self.initial_expected
        }
    }
}

impl BarAggregator for TickRunsBarAggregator {
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

        if bt > 0.0 {
            self.buy_count += 1.0;
        } else {
            self.sell_count += 1.0;
        }
        self.builder.update(tick);

        let max_run = self.buy_count.max(self.sell_count);
        let expected = self.expected_runs();

        if max_run >= expected && expected > 0.0 {
            let bar = self.builder.build();
            let total = self.buy_count + self.sell_count;

            // Update EWMAs: proportion of buy/sell
            if total > 0.0 {
                self.ewma_buy.update(self.buy_count / total);
                self.ewma_sell.update(self.sell_count / total);
            }
            self.ewma_tick_count
                .update(self.builder.tick_count() as f64);

            self.builder.reset();
            self.buy_count = 0.0;
            self.sell_count = 0.0;

            Some(bar)
        } else {
            None
        }
    }
}

/// Volume Runs Bars (VRB).
///
/// Tracks cumulative buy-volume and sell-volume separately.
/// Emits a bar when `max(buy_vol, sell_vol)` exceeds the EWMA expectation.
pub struct VolumeRunsBarAggregator {
    builder: BarBuilder,
    buy_volume: f64,
    sell_volume: f64,
    prev_price: f64,
    prev_bt: f64,
    ewma_buy: Ewma,
    ewma_sell: Ewma,
    ewma_tick_count: Ewma,
    initial_expected: f64,
    initialized: bool,
}

impl VolumeRunsBarAggregator {
    /// Create a new VRB aggregator.
    ///
    /// # Arguments
    ///
    /// * `initial_expected` - Initial expected volume-run threshold before
    ///   the EWMA has accumulated enough history.
    /// * `ewma_span` - Span parameter for the EWMA windows (e.g., 20).
    pub fn new(initial_expected: f64, ewma_span: f64) -> Self {
        Self {
            builder: BarBuilder::new(),
            buy_volume: 0.0,
            sell_volume: 0.0,
            prev_price: 0.0,
            prev_bt: 1.0,
            ewma_buy: Ewma::new(ewma_span),
            ewma_sell: Ewma::new(ewma_span),
            ewma_tick_count: Ewma::new(ewma_span),
            initial_expected,
            initialized: false,
        }
    }

    fn expected_runs(&self) -> f64 {
        if self.ewma_buy.value.is_some() {
            let max_vol = self.ewma_buy.get().max(self.ewma_sell.get());
            self.ewma_tick_count.get() * max_vol
        } else {
            self.initial_expected
        }
    }
}

impl BarAggregator for VolumeRunsBarAggregator {
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

        if bt > 0.0 {
            self.buy_volume += tick.volume;
        } else {
            self.sell_volume += tick.volume;
        }
        self.builder.update(tick);

        let max_vol = self.buy_volume.max(self.sell_volume);
        let expected = self.expected_runs();

        if max_vol >= expected && expected > 0.0 {
            let bar = self.builder.build();
            let total = self.buy_volume + self.sell_volume;

            if total > 0.0 {
                self.ewma_buy.update(self.buy_volume / total);
                self.ewma_sell.update(self.sell_volume / total);
            }
            self.ewma_tick_count
                .update(self.builder.tick_count() as f64);

            self.builder.reset();
            self.buy_volume = 0.0;
            self.sell_volume = 0.0;

            Some(bar)
        } else {
            None
        }
    }
}

/// Dollar Runs Bars (DRB).
///
/// Tracks cumulative buy-dollar-volume and sell-dollar-volume separately.
/// Emits a bar when `max(buy_dv, sell_dv)` exceeds the EWMA expectation.
pub struct DollarRunsBarAggregator {
    builder: BarBuilder,
    buy_dollar: f64,
    sell_dollar: f64,
    prev_price: f64,
    prev_bt: f64,
    ewma_buy: Ewma,
    ewma_sell: Ewma,
    ewma_tick_count: Ewma,
    initial_expected: f64,
    initialized: bool,
}

impl DollarRunsBarAggregator {
    /// Create a new DRB aggregator.
    ///
    /// # Arguments
    ///
    /// * `initial_expected` - Initial expected dollar-run threshold before
    ///   the EWMA has accumulated enough history.
    /// * `ewma_span` - Span parameter for the EWMA windows (e.g., 20).
    pub fn new(initial_expected: f64, ewma_span: f64) -> Self {
        Self {
            builder: BarBuilder::new(),
            buy_dollar: 0.0,
            sell_dollar: 0.0,
            prev_price: 0.0,
            prev_bt: 1.0,
            ewma_buy: Ewma::new(ewma_span),
            ewma_sell: Ewma::new(ewma_span),
            ewma_tick_count: Ewma::new(ewma_span),
            initial_expected,
            initialized: false,
        }
    }

    fn expected_runs(&self) -> f64 {
        if self.ewma_buy.value.is_some() {
            let max_dv = self.ewma_buy.get().max(self.ewma_sell.get());
            self.ewma_tick_count.get() * max_dv
        } else {
            self.initial_expected
        }
    }
}

impl BarAggregator for DollarRunsBarAggregator {
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

        let dv = tick.price * tick.volume;
        if bt > 0.0 {
            self.buy_dollar += dv;
        } else {
            self.sell_dollar += dv;
        }
        self.builder.update(tick);

        let max_dv = self.buy_dollar.max(self.sell_dollar);
        let expected = self.expected_runs();

        if max_dv >= expected && expected > 0.0 {
            let bar = self.builder.build();
            let total = self.buy_dollar + self.sell_dollar;

            if total > 0.0 {
                self.ewma_buy.update(self.buy_dollar / total);
                self.ewma_sell.update(self.sell_dollar / total);
            }
            self.ewma_tick_count
                .update(self.builder.tick_count() as f64);

            self.builder.reset();
            self.buy_dollar = 0.0;
            self.sell_dollar = 0.0;

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
    fn test_tick_runs_bars() {
        let mut agg = TickRunsBarAggregator::new(3.0, 10.0);
        // All upticks: buy_count will grow
        let ticks = vec![
            make_tick(100.0, 1.0),
            make_tick(101.0, 1.0), // buy_count=1
            make_tick(102.0, 1.0), // buy_count=2
            make_tick(103.0, 1.0), // buy_count=3 >= 3.0 -> emit
            make_tick(104.0, 1.0),
        ];
        let bars = agg.process_ticks(&ticks);
        assert!(!bars.is_empty(), "Expected at least one bar from TRB");
    }

    #[test]
    fn test_volume_runs_bars() {
        let mut agg = VolumeRunsBarAggregator::new(50.0, 10.0);
        let ticks = vec![
            make_tick(100.0, 10.0),
            make_tick(101.0, 20.0), // buy_vol=20
            make_tick(102.0, 30.0), // buy_vol=50 >= 50.0 -> emit
        ];
        let bars = agg.process_ticks(&ticks);
        assert!(!bars.is_empty());
    }

    #[test]
    fn test_dollar_runs_bars() {
        let mut agg = DollarRunsBarAggregator::new(5000.0, 10.0);
        let ticks = vec![
            make_tick(100.0, 10.0),
            make_tick(101.0, 20.0), // buy_dv=101*20=2020
            make_tick(102.0, 30.0), // buy_dv=2020+102*30=5080 >= 5000 -> emit
        ];
        let bars = agg.process_ticks(&ticks);
        assert!(!bars.is_empty());
    }
}
