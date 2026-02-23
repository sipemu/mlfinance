//! Triple barrier configuration and touch detection.
//!
//! Implements the triple-barrier method from Advances in Financial Machine
//! Learning by Marcos Lopez de Prado. Three barriers are defined for each
//! trade: an upper (profit-taking), lower (stop-loss), and vertical (time)
//! barrier. The label is determined by which barrier is touched first.

/// Configuration for the triple-barrier labeling method.
#[derive(Debug, Clone)]
pub struct TripleBarrierConfig {
    /// Profit-taking threshold in units of daily volatility.
    /// If `None`, no upper barrier is set.
    pub upper_barrier: Option<f64>,
    /// Stop-loss threshold in units of daily volatility.
    /// If `None`, no lower barrier is set.
    pub lower_barrier: Option<f64>,
    /// Maximum holding period in number of bars (vertical barrier).
    /// If `None`, no vertical barrier is set.
    pub max_holding_period: Option<usize>,
}

/// Describes which barrier was touched first and the associated return.
#[derive(Debug, Clone)]
pub struct BarrierTouch {
    /// Index in the price series where the barrier was touched.
    pub timestamp_index: usize,
    /// Which barrier type was touched.
    pub touch_type: BarrierTouchType,
    /// The return at the point of touch: `(price[touch] - price[entry]) / price[entry]`.
    pub return_value: f64,
}

/// The type of barrier that was touched first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierTouchType {
    /// Upper barrier: profit-taking level was reached.
    Upper,
    /// Lower barrier: stop-loss level was reached.
    Lower,
    /// Vertical barrier: maximum holding period was reached.
    Vertical,
}

/// Check if a return triggers either barrier.
fn check_barriers(
    ret: f64,
    upper_threshold: Option<f64>,
    lower_threshold: Option<f64>,
) -> Option<BarrierTouchType> {
    if let Some(ut) = upper_threshold {
        if ret >= ut {
            return Some(BarrierTouchType::Upper);
        }
    }
    if let Some(lt) = lower_threshold {
        if ret <= -lt {
            return Some(BarrierTouchType::Lower);
        }
    }
    None
}

/// Compute the maximum index to scan based on holding period.
fn max_scan_index(entry_idx: usize, max_holding_period: Option<usize>, series_len: usize) -> usize {
    match max_holding_period {
        Some(hp) => (entry_idx + hp).min(series_len - 1),
        None => series_len - 1,
    }
}

/// Create a vertical barrier touch at the max holding index.
fn vertical_touch(prices: &[f64], entry_price: f64, max_idx: usize) -> BarrierTouch {
    let ret = (prices[max_idx] - entry_price) / entry_price;
    BarrierTouch {
        timestamp_index: max_idx,
        touch_type: BarrierTouchType::Vertical,
        return_value: ret,
    }
}

/// Find which barrier is touched first for a given entry point.
///
/// Starting from `entry_idx`, walks forward through `prices` and checks
/// at each bar whether the return has breached the upper or lower barrier.
/// Barriers are set at `daily_vol * config.upper_barrier` (for upper) and
/// `-daily_vol * config.lower_barrier` (for lower). If neither horizontal
/// barrier is touched within `config.max_holding_period` bars, the vertical
/// barrier is triggered.
///
/// # Arguments
///
/// * `prices` - Full price series.
/// * `entry_idx` - Index of the entry bar in `prices`.
/// * `config` - Triple barrier configuration.
/// * `daily_vol` - Daily volatility at the entry point.
///
/// # Returns
///
/// `Some(BarrierTouch)` if a barrier was touched; `None` if no barrier can
/// be evaluated (e.g., entry is at the end of the series with no holding
/// period set, or the config has no barriers at all).
pub fn find_first_touch(
    prices: &[f64],
    entry_idx: usize,
    config: &TripleBarrierConfig,
    daily_vol: f64,
) -> Option<BarrierTouch> {
    if entry_idx >= prices.len() || prices[entry_idx] == 0.0 {
        return None;
    }

    let entry_price = prices[entry_idx];
    let upper_threshold = config.upper_barrier.map(|u| u * daily_vol);
    let lower_threshold = config.lower_barrier.map(|l| l * daily_vol);
    let max_idx = max_scan_index(entry_idx, config.max_holding_period, prices.len());

    for (offset, &price) in prices[(entry_idx + 1)..=max_idx].iter().enumerate() {
        let ret = (price - entry_price) / entry_price;
        if let Some(touch_type) = check_barriers(ret, upper_threshold, lower_threshold) {
            return Some(BarrierTouch {
                timestamp_index: entry_idx + 1 + offset,
                touch_type,
                return_value: ret,
            });
        }
    }

    if config.max_holding_period.is_some() && max_idx > entry_idx {
        return Some(vertical_touch(prices, entry_price, max_idx));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upper_barrier_touch() {
        // Prices rise steadily
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0), // 2 * daily_vol = 2 * 0.01 = 0.02
            lower_barrier: Some(2.0),
            max_holding_period: Some(10),
        };
        let daily_vol = 0.01; // 1%
        let touch = find_first_touch(&prices, 0, &config, daily_vol).unwrap();
        assert_eq!(touch.touch_type, BarrierTouchType::Upper);
        // 2% return is at index 2 (102/100 - 1 = 0.02)
        assert_eq!(touch.timestamp_index, 2);
        assert!((touch.return_value - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_lower_barrier_touch() {
        // Prices drop
        let prices = vec![100.0, 99.0, 98.0, 97.0, 96.0];
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0), // 2 * 0.01 = 0.02
            max_holding_period: Some(10),
        };
        let daily_vol = 0.01;
        let touch = find_first_touch(&prices, 0, &config, daily_vol).unwrap();
        assert_eq!(touch.touch_type, BarrierTouchType::Lower);
        // -2% at index 2 (98/100 - 1 = -0.02)
        assert_eq!(touch.timestamp_index, 2);
        assert!((touch.return_value - (-0.02)).abs() < 1e-10);
    }

    #[test]
    fn test_vertical_barrier_touch() {
        // Prices stay flat - neither barrier touched
        let prices = vec![100.0, 100.05, 99.95, 100.02, 99.98];
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0), // 2 * 0.01 = 2% - won't be reached
            lower_barrier: Some(2.0),
            max_holding_period: Some(3),
        };
        let daily_vol = 0.01;
        let touch = find_first_touch(&prices, 0, &config, daily_vol).unwrap();
        assert_eq!(touch.touch_type, BarrierTouchType::Vertical);
        assert_eq!(touch.timestamp_index, 3);
    }

    #[test]
    fn test_no_barriers_configured() {
        let prices = vec![100.0, 101.0, 102.0];
        let config = TripleBarrierConfig {
            upper_barrier: None,
            lower_barrier: None,
            max_holding_period: None,
        };
        let touch = find_first_touch(&prices, 0, &config, 0.01);
        assert!(touch.is_none());
    }

    #[test]
    fn test_entry_at_end() {
        let prices = vec![100.0, 101.0];
        let config = TripleBarrierConfig {
            upper_barrier: Some(1.0),
            lower_barrier: Some(1.0),
            max_holding_period: None,
        };
        let touch = find_first_touch(&prices, 1, &config, 0.01);
        assert!(touch.is_none());
    }

    #[test]
    fn test_only_vertical_barrier() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let config = TripleBarrierConfig {
            upper_barrier: None,
            lower_barrier: None,
            max_holding_period: Some(2),
        };
        let touch = find_first_touch(&prices, 0, &config, 0.01).unwrap();
        assert_eq!(touch.touch_type, BarrierTouchType::Vertical);
        assert_eq!(touch.timestamp_index, 2);
    }

    #[test]
    fn test_max_holding_clamped_to_series_end() {
        let prices = vec![100.0, 100.5, 101.0];
        let config = TripleBarrierConfig {
            upper_barrier: None,
            lower_barrier: None,
            max_holding_period: Some(100), // much longer than the series
        };
        let touch = find_first_touch(&prices, 0, &config, 0.01).unwrap();
        assert_eq!(touch.touch_type, BarrierTouchType::Vertical);
        assert_eq!(touch.timestamp_index, 2);
    }

    #[test]
    fn test_entry_out_of_bounds() {
        let prices = vec![100.0, 101.0];
        let config = TripleBarrierConfig {
            upper_barrier: Some(1.0),
            lower_barrier: Some(1.0),
            max_holding_period: Some(5),
        };
        let touch = find_first_touch(&prices, 10, &config, 0.01);
        assert!(touch.is_none());
    }
}
