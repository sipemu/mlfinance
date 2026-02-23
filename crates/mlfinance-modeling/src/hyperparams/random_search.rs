//! Randomized search over hyperparameters (Snippet 9.3).
//!
//! Samples parameter combinations randomly and evaluates them, rather than
//! exhaustively trying all combinations as in grid search.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::hyperparams::grid_search::GridSearchResult;

/// Perform randomized search over parameter distributions.
///
/// Randomly samples `n_iter` parameter combinations from uniform distributions
/// defined by (min, max) ranges and evaluates each one.
///
/// # Arguments
/// * `param_distributions` - Slice of (name, min, max) tuples defining the search space.
/// * `n_iter` - Number of random parameter combinations to try.
/// * `score_fn` - Function that takes parameter values and returns a score.
///   Higher scores are better.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
/// A `GridSearchResult` with the best parameters, best score, and all evaluated scores.
#[allow(clippy::type_complexity)]
pub fn random_search(
    param_distributions: &[(String, f64, f64)],
    n_iter: usize,
    score_fn: &dyn Fn(&[(String, f64)]) -> f64,
    seed: u64,
) -> GridSearchResult {
    if param_distributions.is_empty() || n_iter == 0 {
        return GridSearchResult {
            best_params: vec![],
            best_score: f64::NEG_INFINITY,
            all_scores: vec![],
        };
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_scores = Vec::with_capacity(n_iter);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = vec![];

    for _ in 0..n_iter {
        let params: Vec<(String, f64)> = param_distributions
            .iter()
            .map(|(name, min, max)| {
                let val = rng.gen_range(*min..=*max);
                (name.clone(), val)
            })
            .collect();

        let score = score_fn(&params);
        all_scores.push((params.clone(), score));

        if score > best_score {
            best_score = score;
            best_params = params;
        }
    }

    GridSearchResult {
        best_params,
        best_score,
        all_scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_search_basic() {
        let param_distributions = vec![
            ("alpha".to_string(), 0.0, 1.0),
            ("beta".to_string(), 0.0, 0.1),
        ];

        // Score = -(alpha - 0.5)^2 - (beta - 0.05)^2
        let result = random_search(
            &param_distributions,
            100,
            &|params| {
                let alpha = params[0].1;
                let beta = params[1].1;
                -(alpha - 0.5).powi(2) - (beta - 0.05).powi(2)
            },
            42,
        );

        assert_eq!(result.all_scores.len(), 100);
        // Best alpha should be near 0.5
        let best_alpha = result
            .best_params
            .iter()
            .find(|(n, _)| n == "alpha")
            .unwrap()
            .1;
        assert!(
            (best_alpha - 0.5).abs() < 0.3,
            "Expected alpha near 0.5, got {}",
            best_alpha
        );
    }

    #[test]
    fn test_random_search_reproducible() {
        let params = vec![("x".to_string(), 0.0, 10.0)];
        let score_fn = |p: &[(String, f64)]| -> f64 { -p[0].1.powi(2) };

        let r1 = random_search(&params, 50, &score_fn, 42);
        let r2 = random_search(&params, 50, &score_fn, 42);

        assert!((r1.best_score - r2.best_score).abs() < 1e-10);
        assert!((r1.best_params[0].1 - r2.best_params[0].1).abs() < 1e-10);
    }

    #[test]
    fn test_random_search_empty() {
        let result = random_search(&[], 10, &|_| 0.0, 42);
        assert!(result.best_params.is_empty());
        assert!(result.all_scores.is_empty());
    }

    #[test]
    fn test_random_search_zero_iterations() {
        let params = vec![("x".to_string(), 0.0, 1.0)];
        let result = random_search(&params, 0, &|_| 0.0, 42);
        assert!(result.all_scores.is_empty());
    }

    #[test]
    fn test_random_search_values_in_range() {
        let params = vec![("a".to_string(), 1.0, 5.0), ("b".to_string(), -10.0, -5.0)];
        let result = random_search(&params, 50, &|p| p[0].1 + p[1].1, 42);

        for (combo, _) in &result.all_scores {
            let a = combo[0].1;
            let b = combo[1].1;
            assert!(a >= 1.0 && a <= 5.0, "a out of range: {}", a);
            assert!(b >= -10.0 && b <= -5.0, "b out of range: {}", b);
        }
    }
}
