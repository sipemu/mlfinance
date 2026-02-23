//! Combinatoric and optimization utilities.
//!
//! Provides integer partition enumeration, random matrix generation,
//! trajectory simulation, and mean-variance portfolio optimization.

/// Mean-variance portfolio optimization.
pub mod optimization;
/// Integer partition enumeration (stars-and-bars).
pub mod partitions;
/// Random matrix generation with specified rank.
pub mod random_matrix;
/// Trajectory simulation for multi-step paths.
pub mod trajectories;
