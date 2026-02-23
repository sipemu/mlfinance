"""Tests for pymlfinance.sampling module."""

import numpy as np
import pytest
from pymlfinance import sampling


class TestFracDiff:
    def test_get_weights(self):
        weights = sampling.get_weights(0.5, 10)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 10
        assert weights[0] == pytest.approx(1.0)

    def test_get_weights_ffd(self):
        weights = sampling.get_weights_ffd(0.5, 1e-4)
        assert isinstance(weights, np.ndarray)
        assert len(weights) > 0
        assert weights[0] == pytest.approx(1.0)

    def test_frac_diff_ffd(self, price_series):
        result = sampling.frac_diff_ffd(price_series, 0.5, 1e-4)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(price_series)

    def test_frac_diff_expanding(self, price_series):
        result = sampling.frac_diff_expanding(price_series, 0.5, 1e-4)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(price_series)

    def test_find_min_d(self, price_series):
        d = sampling.find_min_d(price_series, 1.0, 0.1, 1e-4)
        assert 0.0 <= d <= 1.0


class TestBootstrap:
    def test_standard_bootstrap(self):
        result = sampling.standard_bootstrap(100, 50, 42)
        assert isinstance(result, list)
        assert len(result) == 50
        assert all(0 <= i < 100 for i in result)

    def test_seq_bootstrap(self):
        ind_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = sampling.seq_bootstrap(ind_matrix, 3, 42)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_compare_bootstraps(self):
        ind_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = sampling.compare_bootstraps(ind_matrix, 3, 10, 42)
        assert hasattr(result, "seq_uniqueness")
        assert hasattr(result, "std_uniqueness")
        assert result.seq_uniqueness >= 0.0


class TestConcurrency:
    def test_num_co_events(self):
        events = [(0, 2), (1, 3), (2, 4)]
        result = sampling.num_co_events(events, 5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_get_indicator_matrix(self):
        events = [(0, 2), (1, 3)]
        result = sampling.get_indicator_matrix(events, 4)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_average_uniqueness(self):
        events = [(0, 2), (1, 3), (2, 4)]
        result = sampling.average_uniqueness(events, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert all(0.0 <= u <= 1.0 for u in result)


class TestWeights:
    def test_balanced_class_weights(self):
        labels = [1, 1, 1, -1, -1, 0]
        result = sampling.balanced_class_weights(labels)
        assert isinstance(result, dict)

    def test_return_attribution_weights(self):
        # return_attribution_weights(events, returns, num_bars)
        # returns length must equal events length
        events = [(0, 2), (1, 3), (2, 4)]
        returns = np.array([0.01, -0.02, 0.03])
        result = sampling.return_attribution_weights(events, returns, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_time_decay(self):
        weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        result = sampling.time_decay(weights, 0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
