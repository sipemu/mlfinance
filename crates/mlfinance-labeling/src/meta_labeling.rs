//! Meta-labeling: secondary model for bet sizing.
//!
//! The meta-labeling approach trains a secondary model to decide whether to
//! act on a primary model's signal. The meta-label is 1 if the primary
//! model's predicted direction matches the actual outcome, and 0 otherwise.
//! The secondary model's predicted probability can then be used for bet sizing.

use crate::events::Event;

/// A meta-labeler that generates binary labels and computes bet sizes
/// from a primary model's predictions.
#[derive(Debug, Clone)]
pub struct MetaLabeler {
    /// Minimum probability threshold for placing a bet.
    /// If the secondary model's predicted probability is below this,
    /// the bet size is zero.
    pub min_probability: f64,
}

impl MetaLabeler {
    /// Create a new `MetaLabeler` with the given minimum probability threshold.
    ///
    /// # Arguments
    /// * `min_probability` - Minimum probability (0.0 to 1.0) to place a bet.
    pub fn new(min_probability: f64) -> Self {
        Self { min_probability }
    }

    /// Generate meta-labels: 1 if the primary model is correct, 0 otherwise.
    ///
    /// For each event, compares the primary model's predicted direction with
    /// the actual return:
    /// * `1` if `primary_prediction > 0` and `return_value > 0`
    /// * `1` if `primary_prediction < 0` and `return_value < 0`
    /// * `1` if `return_value == 0` (no clear direction)
    /// * `0` otherwise
    ///
    /// # Arguments
    /// * `events` - Slice of events.
    /// * `primary_predictions` - Primary model's directional predictions ({-1, 1}).
    ///   Must have the same length as `events`.
    ///
    /// # Returns
    /// A vector of meta-labels, one per event. Processes up to the shorter
    /// of the two input lengths.
    pub fn generate_labels(&self, events: &[Event], primary_predictions: &[i32]) -> Vec<i32> {
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

                if actual_sign == 0
                    || (pred > 0 && actual_sign > 0)
                    || (pred < 0 && actual_sign < 0)
                {
                    1
                } else {
                    0
                }
            })
            .collect()
    }

    /// Compute bet size from meta-label probability.
    ///
    /// The bet size is derived from the secondary model's predicted probability
    /// that the primary model is correct. If the probability is below
    /// `min_probability`, the bet size is 0.
    ///
    /// The bet size formula maps probability to [-1, 1]:
    /// `bet_size = 2 * probability - 1` (clamped to [0, 1]).
    ///
    /// # Arguments
    /// * `probability` - Predicted probability (0.0 to 1.0) from the secondary model.
    ///
    /// # Returns
    /// Bet size in [0.0, 1.0]. Returns 0.0 if probability is below the threshold.
    pub fn bet_size(&self, probability: f64) -> f64 {
        if probability < self.min_probability {
            return 0.0;
        }
        // Map [0.5, 1.0] -> [0.0, 1.0]; values below 0.5 map to negative but
        // we clamp to 0 since meta-label probabilities should be >= 0.5 for a bet.
        (2.0 * probability - 1.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barriers::BarrierTouchType;

    fn make_event(return_value: f64) -> Event {
        Event {
            entry_idx: 0,
            exit_idx: 1,
            touch_type: if return_value > 0.0 {
                BarrierTouchType::Upper
            } else if return_value < 0.0 {
                BarrierTouchType::Lower
            } else {
                BarrierTouchType::Vertical
            },
            return_value,
        }
    }

    #[test]
    fn test_meta_labeler_new() {
        let ml = MetaLabeler::new(0.5);
        assert!((ml.min_probability - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_generate_labels_correct() {
        let ml = MetaLabeler::new(0.5);
        let events = vec![make_event(0.05), make_event(-0.03)];
        let preds = vec![1, -1];
        let labels = ml.generate_labels(&events, &preds);
        assert_eq!(labels, vec![1, 1]);
    }

    #[test]
    fn test_generate_labels_incorrect() {
        let ml = MetaLabeler::new(0.5);
        let events = vec![make_event(0.05), make_event(-0.03)];
        let preds = vec![-1, 1];
        let labels = ml.generate_labels(&events, &preds);
        assert_eq!(labels, vec![0, 0]);
    }

    #[test]
    fn test_generate_labels_zero_return() {
        let ml = MetaLabeler::new(0.5);
        let events = vec![make_event(0.0)];
        let preds = vec![1];
        let labels = ml.generate_labels(&events, &preds);
        assert_eq!(labels, vec![1]); // zero return => considered correct
    }

    #[test]
    fn test_generate_labels_mixed() {
        let ml = MetaLabeler::new(0.5);
        let events = vec![
            make_event(0.05),
            make_event(-0.03),
            make_event(0.01),
            make_event(-0.02),
        ];
        let preds = vec![1, 1, -1, -1];
        let labels = ml.generate_labels(&events, &preds);
        // pred=1 vs ret>0 => 1, pred=1 vs ret<0 => 0,
        // pred=-1 vs ret>0 => 0, pred=-1 vs ret<0 => 1
        assert_eq!(labels, vec![1, 0, 0, 1]);
    }

    #[test]
    fn test_bet_size_above_threshold() {
        let ml = MetaLabeler::new(0.5);
        let size = ml.bet_size(0.8);
        // 2 * 0.8 - 1 = 0.6
        assert!((size - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_bet_size_below_threshold() {
        let ml = MetaLabeler::new(0.6);
        let size = ml.bet_size(0.5);
        assert_eq!(size, 0.0);
    }

    #[test]
    fn test_bet_size_at_threshold() {
        let ml = MetaLabeler::new(0.5);
        let size = ml.bet_size(0.5);
        // 2 * 0.5 - 1 = 0.0 => exactly at boundary
        assert!((size - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bet_size_full_confidence() {
        let ml = MetaLabeler::new(0.5);
        let size = ml.bet_size(1.0);
        // 2 * 1.0 - 1 = 1.0
        assert!((size - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bet_size_clamped() {
        let ml = MetaLabeler::new(0.3);
        let size = ml.bet_size(0.4);
        // 2 * 0.4 - 1 = -0.2, clamped to 0.0
        assert_eq!(size, 0.0);
    }
}
