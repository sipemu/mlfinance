#![warn(missing_docs)]

//! Parallel computation utilities for financial machine learning.
//!
//! This crate provides tools for partitioning work, mapping functions in parallel,
//! processing job queues with progress tracking, and combinatoric utilities.
//!
//! # Key modules
//!
//! - [`partition`] -- Linear and nested partitioning of index ranges across threads.
//! - [`parallel_map`] -- Partition-based and element-wise parallel map over slices.
//! - [`process_jobs`] -- Parallel job processing with optional progress callbacks.
//! - [`combinatorics`] -- Partitions, random matrices, trajectories, and portfolio optimization.
//! - [`vectorization`] -- Cartesian product and index product helpers.

/// Combinatoric and optimization utilities.
pub mod combinatorics;
/// Partition-based and element-wise parallel map over slices.
pub mod parallel_map;
/// Linear and nested partitioning of index ranges across threads.
pub mod partition;
/// Parallel job processing with optional progress callbacks.
pub mod process_jobs;
/// Cartesian product and index product helpers.
pub mod vectorization;
