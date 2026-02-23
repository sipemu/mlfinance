//! Random forest configuration (Snippet 6.2).
//!
//! Provides three standard configurations inspired by the book's RF setups.

/// How to determine the number of features to consider at each split.
#[derive(Debug, Clone)]
pub enum MaxFeatures {
    /// sqrt(n_features)
    Sqrt,
    /// log2(n_features)
    Log2,
    /// A fraction of n_features
    Fraction(f64),
    /// A fixed number of features
    Fixed(usize),
}

impl MaxFeatures {
    /// Compute the actual number of features to use given the total feature count.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Total number of available features.
    ///
    /// # Returns
    ///
    /// The resolved number of features (always at least 1).
    pub fn compute(&self, n_features: usize) -> usize {
        match self {
            MaxFeatures::Sqrt => ((n_features as f64).sqrt().ceil() as usize).max(1),
            MaxFeatures::Log2 => ((n_features as f64).log2().ceil() as usize).max(1),
            MaxFeatures::Fraction(f) => {
                let f = f.clamp(0.0, 1.0);
                ((n_features as f64 * f).ceil() as usize).max(1)
            }
            MaxFeatures::Fixed(k) => (*k).min(n_features).max(1),
        }
    }
}

/// Configuration for a random forest.
#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    /// Number of trees in the forest.
    pub n_estimators: usize,
    /// Number of features to consider at each split.
    pub max_features: MaxFeatures,
    /// Fraction of samples to use for each tree (bootstrap sampling).
    pub max_samples: f64,
    /// Minimum weighted fraction of samples required at a leaf node.
    pub min_weight_fraction_leaf: f64,
}

/// Standard random forest configuration (Setup 1).
///
/// 1000 trees, `sqrt(n_features)` features per split, 100% bootstrap
/// sampling, and no minimum weight constraint.
///
/// # Arguments
///
/// * `_n_features` - Total number of features (unused, present for API consistency).
///
/// # Returns
///
/// A [`RandomForestConfig`] with the standard setup.
pub fn rf_setup_1(_n_features: usize) -> RandomForestConfig {
    RandomForestConfig {
        n_estimators: 1000,
        max_features: MaxFeatures::Sqrt,
        max_samples: 1.0,
        min_weight_fraction_leaf: 0.0,
    }
}

/// Regularised random forest configuration (Setup 2).
///
/// 5000 trees, `log2(n_features)` features per split, 100% bootstrap
/// sampling, and a 5% minimum weight fraction for regularisation.
///
/// # Arguments
///
/// * `_n_features` - Total number of features (unused, present for API consistency).
///
/// # Returns
///
/// A [`RandomForestConfig`] with the regularised setup.
pub fn rf_setup_2(_n_features: usize) -> RandomForestConfig {
    RandomForestConfig {
        n_estimators: 5000,
        max_features: MaxFeatures::Log2,
        max_samples: 1.0,
        min_weight_fraction_leaf: 0.05,
    }
}

/// Bagging-like random forest configuration (Setup 3).
///
/// 1000 trees, all features considered at each split (no feature
/// sub-sampling), 100% bootstrap sampling, and no minimum weight constraint.
///
/// # Arguments
///
/// * `n_features` - Total number of features. Used to set `MaxFeatures::Fixed`.
///
/// # Returns
///
/// A [`RandomForestConfig`] equivalent to bagging (all features per split).
pub fn rf_setup_3(n_features: usize) -> RandomForestConfig {
    RandomForestConfig {
        n_estimators: 1000,
        max_features: MaxFeatures::Fixed(n_features),
        max_samples: 1.0,
        min_weight_fraction_leaf: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_features_sqrt() {
        let mf = MaxFeatures::Sqrt;
        assert_eq!(mf.compute(100), 10);
        assert_eq!(mf.compute(1), 1);
        assert_eq!(mf.compute(2), 2); // ceil(sqrt(2)) = 2
    }

    #[test]
    fn test_max_features_log2() {
        let mf = MaxFeatures::Log2;
        assert_eq!(mf.compute(8), 3); // log2(8)=3
        assert_eq!(mf.compute(1), 1);
        assert_eq!(mf.compute(16), 4);
    }

    #[test]
    fn test_max_features_fraction() {
        let mf = MaxFeatures::Fraction(0.5);
        assert_eq!(mf.compute(10), 5);
        assert_eq!(mf.compute(1), 1);
    }

    #[test]
    fn test_max_features_fixed() {
        let mf = MaxFeatures::Fixed(5);
        assert_eq!(mf.compute(10), 5);
        assert_eq!(mf.compute(3), 3); // clamped to n_features
    }

    #[test]
    fn test_rf_setup_1() {
        let config = rf_setup_1(100);
        assert_eq!(config.n_estimators, 1000);
        assert_eq!(config.max_features.compute(100), 10);
        assert!((config.max_samples - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rf_setup_2() {
        let config = rf_setup_2(100);
        assert_eq!(config.n_estimators, 5000);
        // log2(100) ~= 6.64, ceil = 7
        assert_eq!(config.max_features.compute(100), 7);
        assert!(config.min_weight_fraction_leaf > 0.0);
    }

    #[test]
    fn test_rf_setup_3() {
        let config = rf_setup_3(100);
        assert_eq!(config.n_estimators, 1000);
        assert_eq!(config.max_features.compute(100), 100); // all features
    }
}
