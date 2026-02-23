//! getBins: generate labels from events (Snippets 3.5-3.7).
//!
//! Provides standard labeling ({-1, 0, 1}), meta-labeling ({0, 1}),
//! and rare-label filtering based on the events produced by the
//! triple-barrier method.

use crate::events::Event;

/// Standard labels: {-1, 0, 1} based on return sign.
///
/// For each event, assigns:
/// * `1` if `return_value > 0` (positive return / upper barrier touch)
/// * `-1` if `return_value < 0` (negative return / lower barrier touch)
/// * `0` if `return_value == 0` (zero return / vertical barrier with no move)
///
/// # Arguments
/// * `events` - Slice of events produced by `get_events`.
///
/// # Returns
/// A vector of labels, one per event.
pub fn get_bins(events: &[Event]) -> Vec<i32> {
    events
        .iter()
        .map(|e| {
            if e.return_value > 0.0 {
                1
            } else if e.return_value < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// Meta-labels: {0, 1} based on whether the primary model's direction was correct.
///
/// For each event, compares the primary model's predicted direction with the
/// actual return. If the primary prediction matches the sign of the return
/// (or the return is zero), assigns `1` (correct). Otherwise assigns `0`
/// (incorrect).
///
/// # Arguments
/// * `events` - Slice of events produced by `get_events`.
/// * `primary_predictions` - Slice of primary model predictions ({-1, 1}).
///   Must have the same length as `events`.
///
/// # Returns
/// A vector of meta-labels, one per event. If lengths differ, only
/// processes up to the shorter length.
pub fn get_meta_bins(events: &[Event], primary_predictions: &[i32]) -> Vec<i32> {
    events
        .iter()
        .zip(primary_predictions.iter())
        .map(|(event, &pred)| {
            let actual_sign = if event.return_value > 0.0 {
                1
            } else if event.return_value < 0.0 {
                -1
            } else {
                0
            };
            // Meta-label is 1 if the primary model's bet direction matches
            // the actual return sign (or if return is exactly zero).
            if actual_sign == 0 || (pred > 0 && actual_sign > 0) || (pred < 0 && actual_sign < 0) {
                1
            } else {
                0
            }
        })
        .collect()
}

/// Drop rare labels that occur less than `min_pct` of total.
///
/// Returns a boolean mask indicating which events should be kept.
/// An event is kept if its label (computed via `get_bins`) belongs to a
/// class that represents at least `min_pct` fraction of all labels.
///
/// # Arguments
/// * `labels` - Slice of labels (e.g., output of `get_bins`).
/// * `min_pct` - Minimum fraction (0.0 to 1.0) of total labels a class
///   must represent to be retained.
///
/// # Returns
/// A boolean vector of the same length as `labels`. `true` means keep,
/// `false` means drop.
pub fn drop_rare_labels(labels: &[i32], min_pct: f64) -> Vec<bool> {
    if labels.is_empty() {
        return vec![];
    }

    let total = labels.len() as f64;
    let mut counts = std::collections::HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0usize) += 1;
    }

    labels
        .iter()
        .map(|l| {
            let count = counts.get(l).copied().unwrap_or(0);
            (count as f64 / total) >= min_pct
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barriers::BarrierTouchType;

    fn make_event(return_value: f64, touch_type: BarrierTouchType) -> Event {
        Event {
            entry_idx: 0,
            exit_idx: 1,
            touch_type,
            return_value,
        }
    }

    #[test]
    fn test_get_bins_positive() {
        let events = vec![make_event(0.05, BarrierTouchType::Upper)];
        let bins = get_bins(&events);
        assert_eq!(bins, vec![1]);
    }

    #[test]
    fn test_get_bins_negative() {
        let events = vec![make_event(-0.03, BarrierTouchType::Lower)];
        let bins = get_bins(&events);
        assert_eq!(bins, vec![-1]);
    }

    #[test]
    fn test_get_bins_zero() {
        let events = vec![make_event(0.0, BarrierTouchType::Vertical)];
        let bins = get_bins(&events);
        assert_eq!(bins, vec![0]);
    }

    #[test]
    fn test_get_bins_mixed() {
        let events = vec![
            make_event(0.02, BarrierTouchType::Upper),
            make_event(-0.01, BarrierTouchType::Lower),
            make_event(0.0, BarrierTouchType::Vertical),
            make_event(0.03, BarrierTouchType::Vertical),
        ];
        let bins = get_bins(&events);
        assert_eq!(bins, vec![1, -1, 0, 1]);
    }

    #[test]
    fn test_get_meta_bins_correct_predictions() {
        let events = vec![
            make_event(0.05, BarrierTouchType::Upper),
            make_event(-0.03, BarrierTouchType::Lower),
        ];
        let predictions = vec![1, -1]; // Both correct
        let meta = get_meta_bins(&events, &predictions);
        assert_eq!(meta, vec![1, 1]);
    }

    #[test]
    fn test_get_meta_bins_incorrect_predictions() {
        let events = vec![
            make_event(0.05, BarrierTouchType::Upper),
            make_event(-0.03, BarrierTouchType::Lower),
        ];
        let predictions = vec![-1, 1]; // Both incorrect
        let meta = get_meta_bins(&events, &predictions);
        assert_eq!(meta, vec![0, 0]);
    }

    #[test]
    fn test_get_meta_bins_zero_return() {
        let events = vec![make_event(0.0, BarrierTouchType::Vertical)];
        let predictions = vec![1];
        let meta = get_meta_bins(&events, &predictions);
        // Zero return is considered "correct" regardless of prediction
        assert_eq!(meta, vec![1]);
    }

    #[test]
    fn test_drop_rare_labels_basic() {
        // 10 labels: 5 ones, 4 minus-ones, 1 zero
        let labels = vec![1, 1, 1, 1, 1, -1, -1, -1, -1, 0];
        // min_pct = 0.15 means a class needs at least 15% representation
        // 1 => 50%, -1 => 40%, 0 => 10%
        let mask = drop_rare_labels(&labels, 0.15);
        // 0 should be dropped (10% < 15%)
        assert_eq!(
            mask,
            vec![true, true, true, true, true, true, true, true, true, false]
        );
    }

    #[test]
    fn test_drop_rare_labels_all_same() {
        let labels = vec![1, 1, 1, 1];
        let mask = drop_rare_labels(&labels, 0.5);
        assert_eq!(mask, vec![true, true, true, true]);
    }

    #[test]
    fn test_drop_rare_labels_empty() {
        let labels: Vec<i32> = vec![];
        let mask = drop_rare_labels(&labels, 0.1);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_drop_rare_labels_none_dropped() {
        let labels = vec![1, -1, 1, -1];
        // Both classes at 50%, min_pct = 0.4 => both kept
        let mask = drop_rare_labels(&labels, 0.4);
        assert_eq!(mask, vec![true, true, true, true]);
    }
}
