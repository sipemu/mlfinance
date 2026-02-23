//! Grid search with purged cross-validation (Snippet 9.1).
//!
//! Exhaustively searches over all combinations of parameters to find
//! the combination that maximizes the score.

use itertools::Itertools;

/// A single parameter with its possible values.
#[derive(Debug, Clone)]
pub struct ParamGrid {
    /// Parameter name.
    pub name: String,
    /// All values to try for this parameter.
    pub values: Vec<f64>,
}

/// Result of a grid search.
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    /// Best parameter combination found.
    pub best_params: Vec<(String, f64)>,
    /// Score of the best parameter combination.
    pub best_score: f64,
    /// All parameter combinations and their scores.
    pub all_scores: Vec<(Vec<(String, f64)>, f64)>,
}

/// Perform grid search over all parameter combinations.
///
/// For each combination of parameter values, calls `score_fn` to evaluate
/// the combination and returns the one with the highest score.
///
/// # Arguments
/// * `param_grids` - Slice of `ParamGrid` structs defining the search space.
/// * `score_fn` - Function that takes parameter values and returns a score.
///   Higher scores are better.
///
/// # Returns
/// A `GridSearchResult` with the best parameters, best score, and all evaluated scores.
#[allow(clippy::type_complexity)]
pub fn grid_search(
    param_grids: &[ParamGrid],
    score_fn: &dyn Fn(&[(String, f64)]) -> f64,
) -> GridSearchResult {
    if param_grids.is_empty() {
        return GridSearchResult {
            best_params: vec![],
            best_score: f64::NEG_INFINITY,
            all_scores: vec![],
        };
    }

    // Build iterators over all value combinations using Cartesian product
    let value_lists: Vec<Vec<f64>> = param_grids.iter().map(|pg| pg.values.clone()).collect();
    let names: Vec<String> = param_grids.iter().map(|pg| pg.name.clone()).collect();

    // Generate all combinations using itertools multi_cartesian_product
    let combinations: Vec<Vec<&f64>> = value_lists
        .iter()
        .map(|vals| vals.iter())
        .multi_cartesian_product()
        .collect();

    let mut all_scores = Vec::with_capacity(combinations.len());
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = vec![];

    for combo in &combinations {
        let params: Vec<(String, f64)> = names
            .iter()
            .zip(combo.iter())
            .map(|(name, &&val)| (name.clone(), val))
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
    fn test_grid_search_basic() {
        let param_grids = vec![
            ParamGrid {
                name: "alpha".to_string(),
                values: vec![0.1, 0.5, 1.0],
            },
            ParamGrid {
                name: "beta".to_string(),
                values: vec![0.01, 0.1],
            },
        ];

        // Score = -|alpha - 0.5| - |beta - 0.01|
        // Best: alpha=0.5, beta=0.01, score=0.0
        let result = grid_search(&param_grids, &|params| {
            let alpha = params.iter().find(|(n, _)| n == "alpha").unwrap().1;
            let beta = params.iter().find(|(n, _)| n == "beta").unwrap().1;
            -(alpha - 0.5).abs() - (beta - 0.01).abs()
        });

        assert_eq!(result.all_scores.len(), 6); // 3 * 2
        assert!((result.best_params[0].1 - 0.5).abs() < 1e-10);
        assert!((result.best_params[1].1 - 0.01).abs() < 1e-10);
        assert!((result.best_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_search_single_param() {
        let param_grids = vec![ParamGrid {
            name: "x".to_string(),
            values: vec![1.0, 2.0, 3.0],
        }];

        let result = grid_search(&param_grids, &|params| {
            let x = params[0].1;
            -(x - 2.5).powi(2) // max at x=2.0 or x=3.0 (x=3.0 is closer)
        });

        assert_eq!(result.all_scores.len(), 3);
        // x=3.0: -(0.5)^2 = -0.25; x=2.0: -(0.5)^2 = -0.25
        // Both have same score, first found wins depending on order
        assert!(
            (result.best_params[0].1 - 2.0).abs() < 1e-10
                || (result.best_params[0].1 - 3.0).abs() < 1e-10
        );
    }

    #[test]
    fn test_grid_search_empty() {
        let result = grid_search(&[], &|_| 0.0);
        assert!(result.best_params.is_empty());
        assert!(result.all_scores.is_empty());
    }

    #[test]
    fn test_grid_search_single_value_per_param() {
        let param_grids = vec![
            ParamGrid {
                name: "a".to_string(),
                values: vec![1.0],
            },
            ParamGrid {
                name: "b".to_string(),
                values: vec![2.0],
            },
        ];

        let result = grid_search(&param_grids, &|params| params[0].1 + params[1].1);
        assert_eq!(result.all_scores.len(), 1);
        assert!((result.best_score - 3.0).abs() < 1e-10);
    }
}
