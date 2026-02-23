//! Generic time-indexed series container with convenience methods for financial returns.
//!
//! [`TimeSeries<T>`] pairs a vector of [`Timestamp`]s with a vector of values of type `T`.
//! When `T = f64`, additional methods for returns, diffs, rolling windows, and cumulative
//! sums become available.

use crate::error::{MlFinanceError, Result};
use crate::types::Timestamp;
use serde::{Deserialize, Serialize};

/// A time-indexed series of values.
///
/// Timestamps and values are stored as parallel vectors of equal length.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries<T> {
    /// Monotonic timestamps for each observation.
    pub timestamps: Vec<Timestamp>,
    /// The data values corresponding to each timestamp.
    pub values: Vec<T>,
}

impl<T: Clone> TimeSeries<T> {
    /// Create a new `TimeSeries` from parallel timestamp and value vectors.
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Timestamps for each observation.
    /// * `values` - Data values; must be the same length as `timestamps`.
    ///
    /// # Errors
    ///
    /// Returns [`MlFinanceError::DimensionMismatch`] if the two vectors differ in length.
    pub fn new(timestamps: Vec<Timestamp>, values: Vec<T>) -> Result<Self> {
        if timestamps.len() != values.len() {
            return Err(MlFinanceError::DimensionMismatch {
                msg: format!(
                    "timestamps length {} != values length {}",
                    timestamps.len(),
                    values.len()
                ),
            });
        }
        Ok(Self { timestamps, values })
    }

    /// Return the number of observations in the series.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return `true` if the series contains no observations.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Extract a contiguous sub-series from index `start` (inclusive) to `end` (exclusive).
    ///
    /// # Arguments
    ///
    /// * `start` - First index to include.
    /// * `end` - One past the last index to include.
    ///
    /// # Errors
    ///
    /// Returns [`MlFinanceError::IndexOutOfBounds`] if `end > len()` or `start > end`.
    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if end > self.len() || start > end {
            return Err(MlFinanceError::IndexOutOfBounds {
                index: end,
                length: self.len(),
            });
        }
        Ok(Self {
            timestamps: self.timestamps[start..end].to_vec(),
            values: self.values[start..end].to_vec(),
        })
    }
}

impl TimeSeries<f64> {
    /// Return overlapping windows of the given size over the values slice.
    ///
    /// # Arguments
    ///
    /// * `window` - The window size (must be > 0 and <= series length).
    ///
    /// # Returns
    ///
    /// A vector of slices, each of length `window`.
    ///
    /// # Errors
    ///
    /// Returns [`MlFinanceError::InvalidParameter`] if `window` is 0 or larger than the series.
    pub fn rolling_window(&self, window: usize) -> Result<Vec<&[f64]>> {
        if window == 0 || window > self.len() {
            return Err(MlFinanceError::InvalidParameter {
                msg: format!("window {} invalid for series length {}", window, self.len()),
            });
        }
        Ok(self.values.windows(window).collect())
    }

    /// Compute simple (arithmetic) percentage changes: `(v[i] - v[i-1]) / v[i-1]`.
    ///
    /// # Returns
    ///
    /// A vector of length `len() - 1`, or empty if the series has fewer than 2 elements.
    pub fn pct_change(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return vec![];
        }
        self.values
            .windows(2)
            .map(|w| {
                if w[0] == 0.0 {
                    0.0
                } else {
                    (w[1] - w[0]) / w[0]
                }
            })
            .collect()
    }

    /// Compute log returns: `ln(v[i] / v[i-1])`.
    ///
    /// Non-positive prices produce a return of `0.0`.
    ///
    /// # Returns
    ///
    /// A vector of length `len() - 1`, or empty if the series has fewer than 2 elements.
    pub fn log_returns(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return vec![];
        }
        self.values
            .windows(2)
            .map(|w| {
                if w[0] <= 0.0 || w[1] <= 0.0 {
                    0.0
                } else {
                    (w[1] / w[0]).ln()
                }
            })
            .collect()
    }

    /// Compute first differences: `v[i] - v[i-1]`.
    ///
    /// # Returns
    ///
    /// A vector of length `len() - 1`, or empty if the series has fewer than 2 elements.
    pub fn diff(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return vec![];
        }
        self.values.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Lag the series by `n` positions, filling the first `n` entries with `None`.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of positions to shift.
    ///
    /// # Returns
    ///
    /// A vector of `Option<f64>` with the same length as the series.
    pub fn shift(&self, n: usize) -> Vec<Option<f64>> {
        let mut result = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            if i < n {
                result.push(None);
            } else {
                result.push(Some(self.values[i - n]));
            }
        }
        result
    }

    /// Compute the cumulative sum of the values.
    ///
    /// # Returns
    ///
    /// A vector of the same length where entry `i` is `sum(values[0..=i])`.
    pub fn cumsum(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.len());
        let mut sum = 0.0;
        for &v in &self.values {
            sum += v;
            result.push(sum);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_ts(values: Vec<f64>) -> TimeSeries<f64> {
        let now = Utc::now();
        let timestamps: Vec<_> = (0..values.len())
            .map(|i| now + chrono::Duration::seconds(i as i64))
            .collect();
        TimeSeries::new(timestamps, values).unwrap()
    }

    #[test]
    fn test_pct_change() {
        let ts = make_ts(vec![100.0, 110.0, 105.0]);
        let pct = ts.pct_change();
        assert_eq!(pct.len(), 2);
        assert!((pct[0] - 0.1).abs() < 1e-10);
        assert!((pct[1] - (-5.0 / 110.0)).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns() {
        let ts = make_ts(vec![100.0, 110.0]);
        let lr = ts.log_returns();
        assert!((lr[0] - (1.1_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_cumsum() {
        let ts = make_ts(vec![1.0, 2.0, 3.0]);
        assert_eq!(ts.cumsum(), vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_diff() {
        let ts = make_ts(vec![1.0, 3.0, 6.0]);
        assert_eq!(ts.diff(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let now = Utc::now();
        let result = TimeSeries::new(vec![now], vec![1.0, 2.0]);
        assert!(result.is_err());
    }
}
