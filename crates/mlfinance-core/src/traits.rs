//! Traits that define the primary extension points of the mlfinance pipeline.
//!
//! Users implement these traits to plug custom bar aggregation strategies, sampling
//! filters, weight generators, and ML classifiers into the framework.

use crate::types::{OhlcvBar, TickData};
use ndarray::{Array1, Array2};

/// Trait for bar aggregation: processes ticks and emits bars when a condition is met.
pub trait BarAggregator: Send {
    /// Ingest a single tick and optionally emit a completed bar.
    ///
    /// # Arguments
    ///
    /// * `tick` - The incoming tick data to process.
    ///
    /// # Returns
    ///
    /// `Some(OhlcvBar)` when the aggregation threshold is reached, otherwise `None`.
    fn process_tick(&mut self, tick: &TickData) -> Option<OhlcvBar>;

    /// Convenience method that processes a slice of ticks and collects all emitted bars.
    ///
    /// # Arguments
    ///
    /// * `ticks` - Slice of tick data to process sequentially.
    ///
    /// # Returns
    ///
    /// A vector of all bars emitted during processing.
    fn process_ticks(&mut self, ticks: &[TickData]) -> Vec<OhlcvBar> {
        ticks.iter().filter_map(|t| self.process_tick(t)).collect()
    }
}

/// Trait for event-based sampling filters.
pub trait Filter: Send {
    /// Decide whether the given value triggers a sampling event.
    ///
    /// # Arguments
    ///
    /// * `value` - The observation to evaluate against the filter condition.
    ///
    /// # Returns
    ///
    /// `true` if the value should be sampled, `false` otherwise.
    fn should_sample(&mut self, value: f64) -> bool;
}

/// Trait for generating sample weights.
pub trait WeightGenerator: Send {
    /// Produce a weight vector of length `num_samples`.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - The number of weights to generate.
    ///
    /// # Returns
    ///
    /// A 1-D array of non-negative weights.
    fn generate_weights(&self, num_samples: usize) -> Array1<f64>;
}

/// Generic classifier interface. Users implement this trait
/// to plug their own ML models into the mlfinance pipeline.
pub trait Classifier: Send + Sync {
    /// Train the model on labelled data with optional per-sample weights.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape `(n_samples, n_features)`.
    /// * `y` - Label vector of length `n_samples`.
    /// * `sample_weight` - Optional per-sample weights of length `n_samples`.
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, sample_weight: Option<&Array1<f64>>);

    /// Predict class labels for the given feature matrix.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape `(n_samples, n_features)`.
    ///
    /// # Returns
    ///
    /// Predicted labels as a 1-D array of length `n_samples`.
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;

    /// Predict class probabilities for the given feature matrix.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape `(n_samples, n_features)`.
    ///
    /// # Returns
    ///
    /// A matrix of shape `(n_samples, n_classes)` with probability estimates.
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;

    /// Return feature importances learned during fitting, if available.
    ///
    /// # Returns
    ///
    /// `Some(Array1)` of length `n_features` if the model exposes importances,
    /// otherwise `None`.
    fn feature_importances(&self) -> Option<Array1<f64>>;
}
