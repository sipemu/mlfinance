//! Unified error type for the mlfinance workspace.
//!
//! All fallible operations across every mlfinance sub-crate return
//! [`Result<T>`](Result), which is an alias for `std::result::Result<T, MlFinanceError>`.

use thiserror::Error;

/// Enumeration of all errors that can occur in the mlfinance workspace.
#[derive(Error, Debug)]
pub enum MlFinanceError {
    /// The input contained fewer data points than the algorithm requires.
    #[error("insufficient data: expected at least {expected}, got {actual}")]
    InsufficientData {
        /// Minimum number of elements required.
        expected: usize,
        /// Number of elements actually provided.
        actual: usize,
    },

    /// Two related dimensions (e.g., rows vs. columns) did not match.
    #[error("dimension mismatch: {msg}")]
    DimensionMismatch {
        /// Human-readable description of the mismatch.
        msg: String,
    },

    /// A caller-supplied parameter was out of range or otherwise invalid.
    #[error("invalid parameter: {msg}")]
    InvalidParameter {
        /// Human-readable description of the invalid parameter.
        msg: String,
    },

    /// An operation was attempted on an empty time series.
    #[error("empty series")]
    EmptySeries,

    /// A numerical computation failed (e.g., singular matrix).
    #[error("computation error: {msg}")]
    ComputationError {
        /// Human-readable description of the computation failure.
        msg: String,
    },

    /// An iterative algorithm did not converge within the allowed budget.
    #[error("convergence failure after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations executed before giving up.
        iterations: usize,
    },

    /// An index exceeded the valid range for the collection.
    #[error("index out of bounds: {index} (length {length})")]
    IndexOutOfBounds {
        /// The out-of-bounds index that was requested.
        index: usize,
        /// The actual length of the collection.
        length: usize,
    },

    /// A matrix expected to be positive-definite was not.
    #[error("not positive definite")]
    NotPositiveDefinite,

    /// Catch-all for errors that do not fit another variant.
    #[error("{0}")]
    Other(String),
}

/// A type alias for `std::result::Result<T, MlFinanceError>`.
pub type Result<T> = std::result::Result<T, MlFinanceError>;
