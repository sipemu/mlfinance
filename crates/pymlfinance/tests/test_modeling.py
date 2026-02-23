"""Tests for pymlfinance.modeling module."""

import numpy as np
import pytest
from pymlfinance import modeling


class TestPurgedKFold:
    def test_split(self):
        n_samples = 50
        kfold = modeling.PurgedKFold(n_splits=3, embargo_pct=0.01)
        # events length must equal n_samples
        events = [(i, min(i + 5, n_samples - 1)) for i in range(n_samples)]
        folds = kfold.split(events, n_samples)
        assert len(folds) == 3
        for fold in folds:
            assert hasattr(fold, "train")
            assert hasattr(fold, "test")
            assert len(fold.train) > 0
            assert len(fold.test) > 0
            assert len(set(fold.train) & set(fold.test)) == 0


class TestFeatureImportance:
    def test_mean_decrease_impurity(self):
        importances = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.12, 0.22, 0.32]]
        result = modeling.mean_decrease_impurity(importances)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_orthogonal_features(self):
        x = np.random.default_rng(42).normal(0, 1, (20, 5))
        transformed, explained = modeling.orthogonal_features(x, 3)
        assert transformed.shape == (20, 3)
        assert len(explained) == 3

    def test_weighted_kendall_tau(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = modeling.weighted_kendall_tau(x, y)
        assert isinstance(result, float)
        assert result == pytest.approx(1.0)

    def test_make_classification(self):
        x, y = modeling.make_classification(100, 3, 2, 1, 42)
        assert x.shape == (100, 6)
        assert len(y) == 100
        # Labels can be -1 or 1
        unique = set(np.unique(y))
        assert unique <= {-1.0, 0.0, 1.0}


class TestHyperparams:
    def test_grid_search(self):
        grids = [("alpha", [0.1, 0.5, 1.0]), ("beta", [0.01, 0.1])]

        def score_fn(params):
            return -(params["alpha"] ** 2 + params["beta"] ** 2)

        result = modeling.grid_search(grids, score_fn)
        assert "best_params" in result
        assert "best_score" in result
        assert result["best_params"]["alpha"] == pytest.approx(0.1)

    def test_random_search(self):
        distributions = [("alpha", 0.01, 1.0), ("beta", 0.001, 0.1)]

        def score_fn(params):
            return -(params["alpha"] ** 2 + params["beta"] ** 2)

        result = modeling.random_search(distributions, 20, score_fn, 42)
        assert "best_params" in result
        assert "best_score" in result

    def test_log_uniform_sample(self):
        result = modeling.log_uniform_sample(0.001, 1.0, 10, 42)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert all(0.001 <= v <= 1.0 for v in result)


class TestScoring:
    def test_accuracy_score(self):
        y_true = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        result = modeling.accuracy_score(y_true, y_pred)
        assert result == pytest.approx(0.8)

    def test_f1_score(self):
        y_true = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        result = modeling.f1_score(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_neg_log_loss(self):
        y_true = np.array([1.0, 0.0, 1.0])
        y_proba = np.array([0.9, 0.1, 0.8])
        result = modeling.neg_log_loss(y_true, y_proba)
        assert isinstance(result, float)
        assert result <= 0


class TestEnsemble:
    def test_bagging_accuracy(self):
        result = modeling.bagging_accuracy(100, 0.6)
        assert isinstance(result, float)
        assert result > 0.6
