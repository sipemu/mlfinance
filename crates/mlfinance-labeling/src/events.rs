//! getEvents: determine first barrier touch for each event (Snippets 3.2-3.3).
//!
//! For each entry signal, finds the first barrier touch using the triple-barrier
//! method and records the resulting event (entry, exit, touch type, return).

use crate::barriers::{find_first_touch, BarrierTouchType, TripleBarrierConfig};

/// An event representing a trade from entry to exit via a barrier touch.
#[derive(Debug, Clone)]
pub struct Event {
    /// Index of the entry bar in the price series.
    pub entry_idx: usize,
    /// Index of the exit bar (where a barrier was touched).
    pub exit_idx: usize,
    /// Which barrier was touched first.
    pub touch_type: BarrierTouchType,
    /// The return at the point of exit: `(price[exit] - price[entry]) / price[entry]`.
    pub return_value: f64,
}

/// Convert a BarrierTouch to an Event for a given entry index.
fn touch_to_event(entry_idx: usize, touch: crate::barriers::BarrierTouch) -> Event {
    Event {
        entry_idx,
        exit_idx: touch.timestamp_index,
        touch_type: touch.touch_type,
        return_value: touch.return_value,
    }
}

/// Generate events by finding the first barrier touch for each entry signal.
///
/// For each entry index, looks up the daily volatility at that point and
/// applies the triple-barrier method to determine which barrier is touched
/// first. Entry indices for which no barrier touch occurs (e.g., at the
/// end of the series) are silently skipped.
///
/// # Arguments
///
/// * `prices` - Full price series.
/// * `entry_indices` - Indices of the entry bars (signals from a primary model).
/// * `config` - Triple barrier configuration.
/// * `daily_vols` - Daily volatility for each bar in `prices`. Must have
///   the same length as `prices`.
///
/// # Returns
///
/// A vector of [`Event`] structs, one for each entry that resulted in a
/// barrier touch.
pub fn get_events(
    prices: &[f64],
    entry_indices: &[usize],
    config: &TripleBarrierConfig,
    daily_vols: &[f64],
) -> Vec<Event> {
    let mut events = Vec::with_capacity(entry_indices.len());

    for &entry_idx in entry_indices {
        if entry_idx >= prices.len() || entry_idx >= daily_vols.len() {
            continue;
        }

        let vol = if daily_vols[entry_idx] <= 0.0 {
            1e-10
        } else {
            daily_vols[entry_idx]
        };

        if let Some(touch) = find_first_touch(prices, entry_idx, config, vol) {
            events.push(touch_to_event(entry_idx, touch));
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> TripleBarrierConfig {
        TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0),
            max_holding_period: Some(5),
        }
    }

    #[test]
    fn test_get_events_basic() {
        // Prices: 100, then rise to 105 over 5 bars
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let daily_vols = vec![0.01; 8]; // 1% daily vol
        let config = make_config(); // upper/lower at 2%, holding 5 bars
        let entry_indices = vec![0, 3];

        let events = get_events(&prices, &entry_indices, &config, &daily_vols);
        assert_eq!(events.len(), 2);

        // First event: entry at 0, upper barrier at 2% => price 102 at idx 2
        assert_eq!(events[0].entry_idx, 0);
        assert_eq!(events[0].exit_idx, 2);
        assert_eq!(events[0].touch_type, BarrierTouchType::Upper);

        // Second event: entry at 3 (price=103), upper at 2% => 103*1.02=105.06
        // idx 4: 104/103-1 = 0.0097 < 0.02
        // idx 5: 105/103-1 = 0.0194 < 0.02
        // idx 6: 106/103-1 = 0.0291 >= 0.02 -> upper touch
        assert_eq!(events[1].entry_idx, 3);
        assert_eq!(events[1].touch_type, BarrierTouchType::Upper);
    }

    #[test]
    fn test_get_events_skip_out_of_bounds() {
        let prices = vec![100.0, 101.0, 102.0];
        let daily_vols = vec![0.01; 3];
        let config = make_config();
        let entry_indices = vec![0, 10]; // 10 is out of bounds

        let events = get_events(&prices, &entry_indices, &config, &daily_vols);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].entry_idx, 0);
    }

    #[test]
    fn test_get_events_empty_entries() {
        let prices = vec![100.0, 101.0, 102.0];
        let daily_vols = vec![0.01; 3];
        let config = make_config();
        let entry_indices: Vec<usize> = vec![];

        let events = get_events(&prices, &entry_indices, &config, &daily_vols);
        assert!(events.is_empty());
    }

    #[test]
    fn test_get_events_with_zero_vol() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let daily_vols = vec![0.0; 4]; // zero volatility
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0),
            max_holding_period: Some(3),
        };
        let entry_indices = vec![0];

        let events = get_events(&prices, &entry_indices, &config, &daily_vols);
        // Should still produce an event (vertical barrier or touch with tiny vol)
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_get_events_vertical_barrier() {
        // Flat prices, neither upper nor lower touched
        let prices = vec![100.0, 100.001, 99.999, 100.0005, 99.9995, 100.0];
        let daily_vols = vec![0.01; 6];
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0),
            max_holding_period: Some(3),
        };
        let entry_indices = vec![0];

        let events = get_events(&prices, &entry_indices, &config, &daily_vols);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].touch_type, BarrierTouchType::Vertical);
        assert_eq!(events[0].exit_idx, 3);
    }
}
