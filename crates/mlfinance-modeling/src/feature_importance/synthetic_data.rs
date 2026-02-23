//! Synthetic data generation for testing feature importance (Snippet 8.7).
//!
//! Generates classification data with informative, redundant, and noise features.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Container for synthetic classification data.
pub struct SyntheticData {
    /// Feature matrix of shape (n_samples, n_informative + n_redundant + n_noise).
    pub x: Array2<f64>,
    /// Target vector of length n_samples (values: -1.0 or 1.0).
    pub y: Array1<f64>,
    /// Number of informative features.
    pub n_informative: usize,
    /// Number of redundant features (linear combinations of informative features).
    pub n_redundant: usize,
    /// Number of pure noise features.
    pub n_noise: usize,
}

/// Generate normalized random weights for the informative features.
fn generate_weights(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n).map(|_| sample_normal(rng)).collect();
    let w_norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
    if w_norm > 0.0 {
        for w in &mut weights {
            *w /= w_norm;
        }
    }
    weights
}

/// Generate labels from a linear combination of informative features.
fn generate_labels(
    x: &Array2<f64>,
    weights: &[f64],
    n_samples: usize,
    rng: &mut StdRng,
) -> Array1<f64> {
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let score: f64 = weights
            .iter()
            .enumerate()
            .map(|(j, &w)| x[[i, j]] * w)
            .sum::<f64>()
            + 0.1 * sample_normal(rng);
        y[i] = if score > 0.0 { 1.0 } else { -1.0 };
    }
    y
}

/// Generate redundant features as linear combinations of informative features.
fn fill_redundant_features(
    x: &mut Array2<f64>,
    n_samples: usize,
    n_informative: usize,
    n_redundant: usize,
    rng: &mut StdRng,
) {
    for j in 0..n_redundant {
        let col_idx = n_informative + j;
        let coeffs: Vec<f64> = (0..n_informative).map(|_| sample_normal(rng)).collect();
        for i in 0..n_samples {
            let val: f64 = coeffs.iter().enumerate().map(|(k, &c)| c * x[[i, k]]).sum();
            x[[i, col_idx]] = val + 0.01 * sample_normal(rng);
        }
    }
}

/// Generate synthetic classification data with informative, redundant, and noise features.
///
/// Informative features are drawn from a standard normal distribution and labels
/// are assigned by a random linear combination of these features. Redundant features
/// are random linear combinations of the informative ones, and noise features are
/// independent standard normal draws.
///
/// # Arguments
///
/// * `n_samples` - Number of samples to generate.
/// * `n_informative` - Number of truly informative features that drive the label.
/// * `n_redundant` - Number of redundant features (linear combinations of informative features).
/// * `n_noise` - Number of pure noise features (standard normal).
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
///
/// A [`SyntheticData`] struct containing the feature matrix, labels, and feature counts.
pub fn make_classification(
    n_samples: usize,
    n_informative: usize,
    n_redundant: usize,
    n_noise: usize,
    seed: u64,
) -> SyntheticData {
    let n_features = n_informative + n_redundant + n_noise;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::zeros((n_samples, n_features));

    // Fill informative features
    for i in 0..n_samples {
        for j in 0..n_informative {
            x[[i, j]] = sample_normal(&mut rng);
        }
    }

    let weights = generate_weights(n_informative, &mut rng);
    let y = generate_labels(&x, &weights, n_samples, &mut rng);

    fill_redundant_features(&mut x, n_samples, n_informative, n_redundant, &mut rng);

    // Fill noise features
    for j in 0..n_noise {
        let col_idx = n_informative + n_redundant + j;
        for i in 0..n_samples {
            x[[i, col_idx]] = sample_normal(&mut rng);
        }
    }

    SyntheticData {
        x,
        y,
        n_informative,
        n_redundant,
        n_noise,
    }
}

/// Sample from standard normal using Box-Muller transform.
fn sample_normal(rng: &mut StdRng) -> f64 {
    // Use Box-Muller transform
    let u1: f64 = rng.gen_range(1e-10..1.0);
    let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_classification_shapes() {
        let data = make_classification(100, 5, 3, 2, 42);
        assert_eq!(data.x.nrows(), 100);
        assert_eq!(data.x.ncols(), 10); // 5 + 3 + 2
        assert_eq!(data.y.len(), 100);
        assert_eq!(data.n_informative, 5);
        assert_eq!(data.n_redundant, 3);
        assert_eq!(data.n_noise, 2);
    }

    #[test]
    fn test_make_classification_labels_binary() {
        let data = make_classification(100, 5, 3, 2, 42);
        for &label in data.y.iter() {
            assert!(
                (label - 1.0).abs() < 1e-10 || (label - (-1.0)).abs() < 1e-10,
                "Label should be -1 or 1, got {}",
                label
            );
        }
    }

    #[test]
    fn test_make_classification_reproducible() {
        let data1 = make_classification(50, 3, 2, 1, 42);
        let data2 = make_classification(50, 3, 2, 1, 42);

        for i in 0..50 {
            for j in 0..6 {
                assert!(
                    (data1.x[[i, j]] - data2.x[[i, j]]).abs() < 1e-10,
                    "Data should be reproducible with same seed"
                );
            }
            assert!(
                (data1.y[i] - data2.y[i]).abs() < 1e-10,
                "Labels should be reproducible with same seed"
            );
        }
    }

    #[test]
    fn test_make_classification_balanced() {
        let data = make_classification(1000, 5, 0, 0, 42);
        let n_positive = data.y.iter().filter(|&&v| v > 0.0).count();
        let n_negative = data.y.iter().filter(|&&v| v < 0.0).count();

        // Should be roughly balanced (not perfectly, but within reason)
        let ratio = n_positive as f64 / (n_positive + n_negative) as f64;
        assert!(
            ratio > 0.3 && ratio < 0.7,
            "Data should be roughly balanced, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_make_classification_no_redundant_no_noise() {
        let data = make_classification(50, 3, 0, 0, 42);
        assert_eq!(data.x.ncols(), 3);
    }

    #[test]
    fn test_make_classification_only_noise() {
        let data = make_classification(50, 0, 0, 5, 42);
        assert_eq!(data.x.ncols(), 5);
    }
}
