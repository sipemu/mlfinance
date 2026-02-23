from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class PurgedKFold:
    """
    Purged K-Fold cross-validation with embargo (AFML Ch. 7).

    Prevents information leakage in time-series data by purging training
    observations that overlap with test events, and optionally applying
    an embargo period after each test set.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    embargo_pct : float, default 0.0
        Fraction of total observations to embargo after each test fold.
    """
    def __init__(self, n_splits=5, embargo_pct=0.0): ...
    def __repr__(self) -> str: ...
    def split(self, /, events, n_samples):
        """
        Generate train/test splits with purging and embargo.

        Parameters
        ----------
        events : list[tuple[int, int]]
            List of (entry_idx, exit_idx) pairs for each observation.
        n_samples : int
            Total number of samples (must equal len(events)).

        Returns
        -------
        list[FoldIndices]
            Train/test index pairs for each fold.
        """
        ...

def accuracy_score(y_true, y_pred):
    """
    Classification accuracy score.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    float
        Fraction of correct predictions.
    """
    ...

def bagging_accuracy(n, p):
    """
    Theoretical accuracy of a bagging ensemble (AFML Ch. 6).

    Computes the probability that a majority of ``n`` classifiers,
    each with individual accuracy ``p``, vote correctly.

    Parameters
    ----------
    n : int
        Number of classifiers in the ensemble (should be odd).
    p : float
        Individual classifier accuracy (0 to 1).

    Returns
    -------
    float
        Ensemble accuracy.
    """
    ...

def cv_score(classifier, x, y, events, n_splits=5, embargo_pct=0.0, sample_weight=None, scoring=None):
    """
    Cross-validated scoring with purged K-fold (AFML Ch. 7).

    Trains and evaluates a classifier on each fold, returning per-fold scores.
    Uses purged K-fold to prevent leakage from overlapping labels.

    Parameters
    ----------
    classifier : object
        An sklearn-compatible classifier with ``.fit(X, y)`` and ``.predict(X)``.
    x : numpy.ndarray
        Feature matrix (n_samples, n_features).
    y : numpy.ndarray
        Label vector (n_samples,).
    events : list[tuple[int, int]]
        Event spans for purging.
    n_splits : int, default 5
        Number of CV folds.
    embargo_pct : float, default 0.0
        Embargo fraction.
    sample_weight : numpy.ndarray, optional
        Per-sample weights for training.
    scoring : callable, optional
        Custom scoring function ``f(y_true, y_pred) -> float``.
        Defaults to accuracy.

    Returns
    -------
    numpy.ndarray
        Array of per-fold scores.
    """
    ...

def f1_score(y_true, y_pred):
    """
    Binary F1 score.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels.
    y_pred : numpy.ndarray
        Predicted binary labels.

    Returns
    -------
    float
        F1 score (harmonic mean of precision and recall).
    """
    ...

def grid_search(param_grids, score_fn):
    """
    Exhaustive grid search over parameter combinations.

    Parameters
    ----------
    param_grids : list[tuple[str, list[float]]]
        Each entry is (parameter_name, values_to_try).
    score_fn : callable
        Function ``f(params_dict) -> float`` that evaluates a parameter set.

    Returns
    -------
    dict
        ``{"best_params": {name: value, ...}, "best_score": float}``
    """
    ...

def log_uniform_sample(low, high, n, seed):
    """
    Sample from a log-uniform distribution.

    Parameters
    ----------
    low : float
        Lower bound (> 0).
    high : float
        Upper bound.
    n : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Log-uniformly distributed samples.
    """
    ...

def make_classification(n_samples, n_informative, n_redundant, n_noise, seed):
    """
    Generate a synthetic classification dataset for testing.

    Creates a dataset with informative, redundant, and noise features.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_informative : int
        Number of truly informative features.
    n_redundant : int
        Number of redundant (linear combinations of informative) features.
    n_noise : int
        Number of pure noise features.
    seed : int
        Random seed.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (X, y) — feature matrix and binary labels.
    """
    ...

def mean_decrease_accuracy(classifier, x, y, scoring=None, seed=42):
    """
    Mean Decrease Accuracy (MDA) feature importance (AFML Ch. 8).

    Measures each feature's importance by the drop in accuracy when
    the feature is permuted.

    Parameters
    ----------
    classifier : object
        A fitted sklearn-compatible classifier.
    x : numpy.ndarray
        Feature matrix (n_samples, n_features).
    y : numpy.ndarray
        True labels.
    scoring : callable, optional
        Scoring function ``f(y_true, y_pred) -> float``. Defaults to accuracy.
    seed : int, default 42
        Random seed for permutation.

    Returns
    -------
    numpy.ndarray
        Importance score per feature (higher = more important).
    """
    ...

def mean_decrease_impurity(importances_per_tree):
    """
    Mean Decrease Impurity (MDI) feature importance.

    Averages per-tree Gini importances from a random forest.

    Parameters
    ----------
    importances_per_tree : list[list[float]]
        Feature importances from each tree (n_trees x n_features).

    Returns
    -------
    numpy.ndarray
        Mean feature importance across trees.
    """
    ...

def neg_log_loss(y_true, y_proba):
    """
    Negative log-loss (cross-entropy).

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels (0 or 1).
    y_proba : numpy.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Negative log-loss (higher is better).
    """
    ...

def orthogonal_features(x, n_components):
    """
    Extract orthogonal features via PCA.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix (n_samples, n_features).
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (transformed, explained_variance_ratio) — the projected data of
        shape (n_samples, n_components) and the variance explained by
        each component.
    """
    ...

def random_search(param_distributions, n_iter, score_fn, seed):
    """
    Random search over parameter distributions.

    Parameters
    ----------
    param_distributions : list[tuple[str, float, float]]
        Each entry is (parameter_name, low, high) defining a uniform range.
    n_iter : int
        Number of random combinations to evaluate.
    score_fn : callable
        Function ``f(params_dict) -> float``.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{"best_params": {name: value, ...}, "best_score": float}``
    """
    ...

def single_feature_importance(classifier, x, y, events, n_splits=5, scoring=None):
    """
    Single Feature Importance (SFI) — evaluate each feature independently (AFML Ch. 8).

    Trains a separate model on each individual feature and reports
    cross-validated performance.

    Parameters
    ----------
    classifier : object
        An sklearn-compatible classifier.
    x : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Labels.
    events : list[tuple[int, int]]
        Event spans for purged CV.
    n_splits : int, default 5
        Number of CV folds.
    scoring : callable, optional
        Custom scorer. Defaults to accuracy.

    Returns
    -------
    numpy.ndarray
        Mean CV score per feature.
    """
    ...

def weighted_kendall_tau(x, y, weights=None):
    """
    Weighted Kendall tau rank correlation.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.
    weights : numpy.ndarray, optional
        Per-observation weights.

    Returns
    -------
    float
        Weighted Kendall tau coefficient in [-1, 1].
    """
    ...
