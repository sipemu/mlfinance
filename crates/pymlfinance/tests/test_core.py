"""Tests for pymlfinance.core module."""

import numpy as np
import pytest
from pymlfinance import core


class TestMath:
    def test_ewma(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = core.ewma(data, 3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert result[0] == pytest.approx(1.0)
        assert result[-1] > result[0]

    def test_ewma_std(self):
        data = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        result = core.ewma_std(data, 3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_cumsum(self):
        data = np.array([1.0, 2.0, 3.0])
        result = core.cumsum(data)
        np.testing.assert_array_almost_equal(result, [1.0, 3.0, 6.0])

    def test_log_returns(self):
        prices = np.array([100.0, 110.0, 105.0])
        result = core.log_returns(prices)
        assert len(result) == 2
        assert result[0] == pytest.approx(np.log(110.0 / 100.0))

    def test_simple_returns(self):
        prices = np.array([100.0, 110.0, 105.0])
        result = core.simple_returns(prices)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.10)


class TestStats:
    def test_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert core.mean(data) == pytest.approx(3.0)

    def test_variance(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = core.variance(data)
        assert result > 0

    def test_std_dev(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = core.std_dev(data)
        assert result > 0

    def test_skewness(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = core.skewness(data)
        assert isinstance(result, float)

    def test_kurtosis(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = core.kurtosis(data)
        assert isinstance(result, float)

    def test_weighted_mean(self):
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.5, 0.3, 0.2])
        result = core.weighted_mean(data, weights)
        assert result == pytest.approx(0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0)


class TestMatrix:
    def test_correlation_matrix(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = core.correlation_matrix(data)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_covariance_matrix(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = core.covariance_matrix(data)
        assert result.shape == (2, 2)
        assert result[0, 1] == pytest.approx(result[1, 0])

    def test_power_iteration_eig(self):
        mat = np.array([[4.0, 2.0], [2.0, 3.0]])
        # Returns (eigenvalues_array, eigenvectors_2d)
        eigenvalues, eigenvectors = core.power_iteration_eig(mat, 2)
        assert isinstance(eigenvalues, np.ndarray)
        assert len(eigenvalues) == 2
        assert eigenvectors.shape == (2, 2)

    def test_matrix_inverse(self):
        mat = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = core.matrix_inverse(mat)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(0.5)
