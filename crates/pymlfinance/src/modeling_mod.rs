use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::classifier::PyClassifier;
use crate::convert::*;
use crate::types::*;

// ── PurgedKFold class ───────────────────────────────────────────────────────

/// Purged K-Fold cross-validation with embargo (AFML Ch. 7).
///
/// Prevents information leakage in time-series data by purging training
/// observations that overlap with test events, and optionally applying
/// an embargo period after each test set.
///
/// Parameters
/// ----------
/// n_splits : int, default 5
///     Number of folds.
/// embargo_pct : float, default 0.0
///     Fraction of total observations to embargo after each test fold.
#[pyclass(name = "PurgedKFold")]
pub struct PyPurgedKFold {
    inner: mlfinance::modeling::cross_validation::purged_kfold::PurgedKFold,
}

#[pymethods]
impl PyPurgedKFold {
    #[new]
    #[pyo3(signature = (n_splits=5, embargo_pct=0.0))]
    fn new(n_splits: usize, embargo_pct: f64) -> Self {
        Self {
            inner: mlfinance::modeling::cross_validation::purged_kfold::PurgedKFold::new(
                n_splits,
                embargo_pct,
            ),
        }
    }

    /// Generate train/test splits with purging and embargo.
    ///
    /// Parameters
    /// ----------
    /// events : list[tuple[int, int]]
    ///     List of (entry_idx, exit_idx) pairs for each observation.
    /// n_samples : int
    ///     Total number of samples (must equal len(events)).
    ///
    /// Returns
    /// -------
    /// list[FoldIndices]
    ///     Train/test index pairs for each fold.
    fn split(&self, events: Vec<(usize, usize)>, n_samples: usize) -> Vec<PyFoldIndices> {
        let folds = self.inner.split(&events, n_samples);
        folds
            .into_iter()
            .map(|f| PyFoldIndices {
                train: f.train,
                test: f.test,
            })
            .collect()
    }
}

// ── Cross-validation scoring ────────────────────────────────────────────────

/// Cross-validated scoring with purged K-fold (AFML Ch. 7).
///
/// Trains and evaluates a classifier on each fold, returning per-fold scores.
/// Uses purged K-fold to prevent leakage from overlapping labels.
///
/// Parameters
/// ----------
/// classifier : object
///     An sklearn-compatible classifier with ``.fit(X, y)`` and ``.predict(X)``.
/// x : numpy.ndarray
///     Feature matrix (n_samples, n_features).
/// y : numpy.ndarray
///     Label vector (n_samples,).
/// events : list[tuple[int, int]]
///     Event spans for purging.
/// n_splits : int, default 5
///     Number of CV folds.
/// embargo_pct : float, default 0.0
///     Embargo fraction.
/// sample_weight : numpy.ndarray, optional
///     Per-sample weights for training.
/// scoring : callable, optional
///     Custom scoring function ``f(y_true, y_pred) -> float``.
///     Defaults to accuracy.
///
/// Returns
/// -------
/// numpy.ndarray
///     Array of per-fold scores.
#[pyfunction]
#[pyo3(signature = (classifier, x, y, events, n_splits=5, embargo_pct=0.0, sample_weight=None, scoring=None))]
fn cv_score(
    py: Python<'_>,
    classifier: Py<PyAny>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    events: Vec<(usize, usize)>,
    n_splits: usize,
    embargo_pct: f64,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    scoring: Option<Py<PyAny>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_arr = py_to_array2(x);
    let y_arr = py_to_array1(y);
    let sw = sample_weight.map(|w| py_to_array1(w));
    let mut clf = PyClassifier::new(classifier.clone_ref(py));

    let scoring_fn: Box<dyn Fn(&ndarray::Array1<f64>, &ndarray::Array1<f64>) -> f64> =
        if let Some(scorer) = scoring {
            let scorer = scorer.clone_ref(py);
            Box::new(
                move |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                    Python::attach(|py2| {
                        let yt = numpy::PyArray1::from_array(py2, y_true);
                        let yp = numpy::PyArray1::from_array(py2, y_pred);
                        scorer
                            .call1(py2, (yt, yp))
                            .and_then(|r| r.extract::<f64>(py2))
                            .unwrap_or(0.0)
                    })
                },
            )
        } else {
            Box::new(
                |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                    // Default: accuracy
                    let correct = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .filter(|(&a, &b)| (a - b).abs() < 0.5)
                        .count();
                    correct as f64 / y_true.len().max(1) as f64
                },
            )
        };

    let scores = mlfinance::modeling::cross_validation::cv_score::cv_score(
        &mut clf,
        &x_arr,
        &y_arr,
        &events,
        n_splits,
        embargo_pct,
        sw.as_ref(),
        &*scoring_fn,
    );
    Ok(vec_to_py_array(py, scores))
}

// ── Feature importance ──────────────────────────────────────────────────────

/// Mean Decrease Impurity (MDI) feature importance.
///
/// Averages per-tree Gini importances from a random forest.
///
/// Parameters
/// ----------
/// importances_per_tree : list[list[float]]
///     Feature importances from each tree (n_trees x n_features).
///
/// Returns
/// -------
/// numpy.ndarray
///     Mean feature importance across trees.
#[pyfunction]
fn mean_decrease_impurity(
    py: Python<'_>,
    importances_per_tree: Vec<Vec<f64>>,
) -> Py<PyArray1<f64>> {
    let result =
        mlfinance::modeling::feature_importance::mdi::mean_decrease_impurity(&importances_per_tree);
    array1_to_py(py, result)
}

/// Mean Decrease Accuracy (MDA) feature importance (AFML Ch. 8).
///
/// Measures each feature's importance by the drop in accuracy when
/// the feature is permuted.
///
/// Parameters
/// ----------
/// classifier : object
///     A fitted sklearn-compatible classifier.
/// x : numpy.ndarray
///     Feature matrix (n_samples, n_features).
/// y : numpy.ndarray
///     True labels.
/// scoring : callable, optional
///     Scoring function ``f(y_true, y_pred) -> float``. Defaults to accuracy.
/// seed : int, default 42
///     Random seed for permutation.
///
/// Returns
/// -------
/// numpy.ndarray
///     Importance score per feature (higher = more important).
#[pyfunction]
#[pyo3(signature = (classifier, x, y, scoring=None, seed=42))]
fn mean_decrease_accuracy(
    py: Python<'_>,
    classifier: Py<PyAny>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    scoring: Option<Py<PyAny>>,
    seed: u64,
) -> Py<PyArray1<f64>> {
    let x_arr = py_to_array2(x);
    let y_arr = py_to_array1(y);
    let clf = PyClassifier::new(classifier.clone_ref(py));

    let scoring_fn = make_scoring_fn(py, scoring);

    let result = mlfinance::modeling::feature_importance::mda::mean_decrease_accuracy(
        &clf,
        &x_arr,
        &y_arr,
        &*scoring_fn,
        seed,
    );
    array1_to_py(py, result)
}

/// Single Feature Importance (SFI) — evaluate each feature independently (AFML Ch. 8).
///
/// Trains a separate model on each individual feature and reports
/// cross-validated performance.
///
/// Parameters
/// ----------
/// classifier : object
///     An sklearn-compatible classifier.
/// x : numpy.ndarray
///     Feature matrix.
/// y : numpy.ndarray
///     Labels.
/// events : list[tuple[int, int]]
///     Event spans for purged CV.
/// n_splits : int, default 5
///     Number of CV folds.
/// scoring : callable, optional
///     Custom scorer. Defaults to accuracy.
///
/// Returns
/// -------
/// numpy.ndarray
///     Mean CV score per feature.
#[pyfunction]
#[pyo3(signature = (classifier, x, y, events, n_splits=5, scoring=None))]
fn single_feature_importance(
    py: Python<'_>,
    classifier: Py<PyAny>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    events: Vec<(usize, usize)>,
    n_splits: usize,
    scoring: Option<Py<PyAny>>,
) -> Py<PyArray1<f64>> {
    let x_arr = py_to_array2(x);
    let y_arr = py_to_array1(y);
    let mut clf = PyClassifier::new(classifier.clone_ref(py));

    let scoring_fn = make_scoring_fn(py, scoring);

    let result = mlfinance::modeling::feature_importance::sfi::single_feature_importance(
        &mut clf,
        &x_arr,
        &y_arr,
        &events,
        n_splits,
        &*scoring_fn,
    );
    vec_to_py_array(py, result)
}

/// Extract orthogonal features via PCA.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Feature matrix (n_samples, n_features).
/// n_components : int
///     Number of principal components to retain.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     (transformed, explained_variance_ratio) — the projected data of
///     shape (n_samples, n_components) and the variance explained by
///     each component.
#[pyfunction]
fn orthogonal_features(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let x_arr = py_to_array2(x);
    let (transformed, explained) = to_pyresult(
        mlfinance::modeling::feature_importance::orthogonal::orthogonal_features(
            &x_arr,
            n_components,
        ),
    )?;
    Ok((array2_to_py(py, transformed), array1_to_py(py, explained)))
}

/// Weighted Kendall tau rank correlation.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
/// weights : numpy.ndarray, optional
///     Per-observation weights.
///
/// Returns
/// -------
/// float
///     Weighted Kendall tau coefficient in [-1, 1].
#[pyfunction]
#[pyo3(signature = (x, y, weights=None))]
fn weighted_kendall_tau(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> f64 {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    let wv = weights.map(|w| py_to_vec(w));
    mlfinance::modeling::feature_importance::kendall_tau::weighted_kendall_tau(
        &xv,
        &yv,
        wv.as_deref(),
    )
}

/// Generate a synthetic classification dataset for testing.
///
/// Creates a dataset with informative, redundant, and noise features.
///
/// Parameters
/// ----------
/// n_samples : int
///     Number of observations.
/// n_informative : int
///     Number of truly informative features.
/// n_redundant : int
///     Number of redundant (linear combinations of informative) features.
/// n_noise : int
///     Number of pure noise features.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     (X, y) — feature matrix and binary labels.
#[pyfunction]
fn make_classification(
    py: Python<'_>,
    n_samples: usize,
    n_informative: usize,
    n_redundant: usize,
    n_noise: usize,
    seed: u64,
) -> (Py<PyArray2<f64>>, Py<PyArray1<f64>>) {
    let result = mlfinance::modeling::feature_importance::synthetic_data::make_classification(
        n_samples,
        n_informative,
        n_redundant,
        n_noise,
        seed,
    );
    (array2_to_py(py, result.x), array1_to_py(py, result.y))
}

// ── Hyperparameter search ───────────────────────────────────────────────────

/// Exhaustive grid search over parameter combinations.
///
/// Parameters
/// ----------
/// param_grids : list[tuple[str, list[float]]]
///     Each entry is (parameter_name, values_to_try).
/// score_fn : callable
///     Function ``f(params_dict) -> float`` that evaluates a parameter set.
///
/// Returns
/// -------
/// dict
///     ``{"best_params": {name: value, ...}, "best_score": float}``
#[pyfunction]
fn grid_search(
    py: Python<'_>,
    param_grids: Vec<(String, Vec<f64>)>,
    score_fn: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    use mlfinance::modeling::hyperparams::grid_search::{self as gs, ParamGrid};

    let grids: Vec<ParamGrid> = param_grids
        .into_iter()
        .map(|(name, values)| ParamGrid { name, values })
        .collect();

    let score_fn_ref = score_fn.clone_ref(py);
    let scoring = move |params: &[(String, f64)]| -> f64 {
        Python::attach(|py2| {
            let dict = pyo3::types::PyDict::new(py2);
            for (k, v) in params {
                let _ = dict.set_item(k, v);
            }
            score_fn_ref
                .call1(py2, (dict,))
                .and_then(|r| r.extract::<f64>(py2))
                .unwrap_or(f64::NEG_INFINITY)
        })
    };

    let result = gs::grid_search(&grids, &scoring);
    let dict = pyo3::types::PyDict::new(py);
    let best_dict = pyo3::types::PyDict::new(py);
    for (k, v) in &result.best_params {
        best_dict.set_item(k, v)?;
    }
    dict.set_item("best_params", best_dict)?;
    dict.set_item("best_score", result.best_score)?;
    Ok(dict.into_any().unbind())
}

/// Random search over parameter distributions.
///
/// Parameters
/// ----------
/// param_distributions : list[tuple[str, float, float]]
///     Each entry is (parameter_name, low, high) defining a uniform range.
/// n_iter : int
///     Number of random combinations to evaluate.
/// score_fn : callable
///     Function ``f(params_dict) -> float``.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// dict
///     ``{"best_params": {name: value, ...}, "best_score": float}``
#[pyfunction]
fn random_search(
    py: Python<'_>,
    param_distributions: Vec<(String, f64, f64)>,
    n_iter: usize,
    score_fn: Py<PyAny>,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let score_fn_ref = score_fn.clone_ref(py);
    let scoring = move |params: &[(String, f64)]| -> f64 {
        Python::attach(|py2| {
            let dict = pyo3::types::PyDict::new(py2);
            for (k, v) in params {
                let _ = dict.set_item(k, v);
            }
            score_fn_ref
                .call1(py2, (dict,))
                .and_then(|r| r.extract::<f64>(py2))
                .unwrap_or(f64::NEG_INFINITY)
        })
    };

    let result = mlfinance::modeling::hyperparams::random_search::random_search(
        &param_distributions,
        n_iter,
        &scoring,
        seed,
    );
    let dict = pyo3::types::PyDict::new(py);
    let best_dict = pyo3::types::PyDict::new(py);
    for (k, v) in &result.best_params {
        best_dict.set_item(k, v)?;
    }
    dict.set_item("best_params", best_dict)?;
    dict.set_item("best_score", result.best_score)?;
    Ok(dict.into_any().unbind())
}

/// Sample from a log-uniform distribution.
///
/// Parameters
/// ----------
/// low : float
///     Lower bound (> 0).
/// high : float
///     Upper bound.
/// n : int
///     Number of samples.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// numpy.ndarray
///     Log-uniformly distributed samples.
#[pyfunction]
fn log_uniform_sample(
    py: Python<'_>,
    low: f64,
    high: f64,
    n: usize,
    seed: u64,
) -> Py<PyArray1<f64>> {
    let result =
        mlfinance::modeling::hyperparams::log_uniform::log_uniform_sample(low, high, n, seed);
    vec_to_py_array(py, result)
}

// ── Scoring functions ───────────────────────────────────────────────────────

/// Binary F1 score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels.
/// y_pred : numpy.ndarray
///     Predicted binary labels.
///
/// Returns
/// -------
/// float
///     F1 score (harmonic mean of precision and recall).
#[pyfunction]
fn f1_score(y_true: PyReadonlyArray1<'_, f64>, y_pred: PyReadonlyArray1<'_, f64>) -> f64 {
    let yt = py_to_vec(y_true);
    let yp = py_to_vec(y_pred);
    mlfinance::modeling::hyperparams::scoring::f1_score(&yt, &yp)
}

/// Negative log-loss (cross-entropy).
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels (0 or 1).
/// y_proba : numpy.ndarray
///     Predicted probabilities for the positive class.
///
/// Returns
/// -------
/// float
///     Negative log-loss (higher is better).
#[pyfunction]
fn neg_log_loss(y_true: PyReadonlyArray1<'_, f64>, y_proba: PyReadonlyArray1<'_, f64>) -> f64 {
    let yt = py_to_vec(y_true);
    let yp = py_to_vec(y_proba);
    mlfinance::modeling::hyperparams::scoring::neg_log_loss(&yt, &yp)
}

/// Classification accuracy score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// float
///     Fraction of correct predictions.
#[pyfunction]
fn accuracy_score(y_true: PyReadonlyArray1<'_, f64>, y_pred: PyReadonlyArray1<'_, f64>) -> f64 {
    let yt = py_to_vec(y_true);
    let yp = py_to_vec(y_pred);
    mlfinance::modeling::hyperparams::scoring::accuracy_score(&yt, &yp)
}

// ── Ensemble ────────────────────────────────────────────────────────────────

/// Theoretical accuracy of a bagging ensemble (AFML Ch. 6).
///
/// Computes the probability that a majority of ``n`` classifiers,
/// each with individual accuracy ``p``, vote correctly.
///
/// Parameters
/// ----------
/// n : int
///     Number of classifiers in the ensemble (should be odd).
/// p : float
///     Individual classifier accuracy (0 to 1).
///
/// Returns
/// -------
/// float
///     Ensemble accuracy.
#[pyfunction]
fn bagging_accuracy(n: usize, p: f64) -> f64 {
    mlfinance::modeling::ensemble::bagging::bagging_accuracy(n, p)
}

// ── Helper ──────────────────────────────────────────────────────────────────

fn make_scoring_fn(
    py: Python<'_>,
    scoring: Option<Py<PyAny>>,
) -> Box<dyn Fn(&ndarray::Array1<f64>, &ndarray::Array1<f64>) -> f64> {
    if let Some(scorer) = scoring {
        let scorer = scorer.clone_ref(py);
        Box::new(
            move |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                Python::attach(|py2| {
                    let yt = numpy::PyArray1::from_array(py2, y_true);
                    let yp = numpy::PyArray1::from_array(py2, y_pred);
                    scorer
                        .call1(py2, (yt, yp))
                        .and_then(|r| r.extract::<f64>(py2))
                        .unwrap_or(0.0)
                })
            },
        )
    } else {
        Box::new(
            |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&a, &b)| (a - b).abs() < 0.5)
                    .count();
                correct as f64 / y_true.len().max(1) as f64
            },
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_classes!(m, PyPurgedKFold);
    register_functions!(
        m,
        cv_score,
        mean_decrease_impurity,
        mean_decrease_accuracy,
        single_feature_importance,
        orthogonal_features,
        weighted_kendall_tau,
        make_classification,
        grid_search,
        random_search,
        log_uniform_sample,
        f1_score,
        neg_log_loss,
        accuracy_score,
        bagging_accuracy,
    );
    Ok(())
}
