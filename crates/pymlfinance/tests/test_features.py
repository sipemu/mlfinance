"""Tests for pymlfinance.features module."""

import numpy as np
import pytest
from pymlfinance import features


class TestStructuralBreaks:
    def test_adf_test(self, price_series):
        # adf_test returns (stat, residuals)
        stat, residuals = features.adf_test(price_series, 5)
        assert isinstance(stat, float)
        assert isinstance(residuals, np.ndarray)

    def test_sadf(self, price_series):
        # sadf(series, min_window, max_lags)
        result = features.sadf(price_series, 20, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_sadf_stat(self, price_series):
        stat = features.sadf_stat(price_series, 20, 5)
        assert isinstance(stat, float)

    def test_gsadf_stat(self, price_series):
        stat = features.gsadf_stat(price_series, 20, 5)
        assert isinstance(stat, float)

    def test_brown_durbin_evans(self, price_series):
        # Takes residuals, returns (cusum, critical_value)
        residuals = np.diff(price_series)
        cusum, crit = features.brown_durbin_evans(residuals)
        assert isinstance(cusum, np.ndarray)
        assert isinstance(crit, float)

    def test_chu_stinchcombe_white(self, price_series):
        result = features.chu_stinchcombe_white(price_series, 5.0)
        assert isinstance(result, np.ndarray)


class TestEntropy:
    def test_shannon_entropy(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = features.shannon_entropy(probs)
        assert result == pytest.approx(2.0)

    def test_plugin_entropy(self):
        # plugin_entropy(sequence: list[int], num_symbols: int)
        data = [0, 1, 0, 1, 0, 2]
        result = features.plugin_entropy(data, 3)
        assert isinstance(result, float)
        assert result >= 0

    def test_lempel_ziv_complexity(self):
        bits = [True, False, True, True, False, False, True, False]
        result = features.lempel_ziv_complexity(bits)
        assert isinstance(result, int)
        assert result > 0

    def test_gaussian_entropy(self):
        result = features.gaussian_entropy(1.0)
        assert isinstance(result, float)


class TestMicrostructure:
    def test_amihud_lambda(self):
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.01])
        volume = np.array([1000.0, 1500.0, 1200.0, 800.0, 1100.0])
        result = features.amihud_lambda(returns, volume)
        assert isinstance(result, float)

    def test_kyle_lambda(self):
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.01])
        volume = np.array([1000.0, 1500.0, 1200.0, 800.0, 1100.0])
        result = features.kyle_lambda(returns, volume)
        assert isinstance(result, float)

    def test_roll_spread(self):
        prices = np.array([100.0, 100.5, 99.8, 100.2, 100.1])
        result = features.roll_spread(prices)
        assert isinstance(result, float)

    def test_corwin_schultz_spread(self):
        high = np.array([105.0, 106.0, 107.0, 108.0, 109.0])
        low = np.array([95.0, 96.0, 97.0, 98.0, 99.0])
        result = features.corwin_schultz_spread(high, low)
        assert isinstance(result, np.ndarray)

    def test_tick_rule_classify(self):
        prices = np.array([100.0, 100.5, 100.3, 100.8, 100.2])
        result = features.tick_rule_classify(prices)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(prices)

    def test_vpin(self):
        # vpin(volumes, prices, bucket_size, n_buckets)
        volumes = np.array([100.0, 150.0, 200.0, 120.0, 180.0, 130.0, 160.0, 140.0])
        prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 100.9, 101.5, 101.0])
        result = features.vpin(volumes, prices, 200.0, 3)
        assert isinstance(result, np.ndarray)


class TestDenoising:
    def test_denoise_corr(self):
        corr = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        result = features.denoise_corr(corr, 10.0)
        assert result.shape == (3, 3)

    def test_cov_to_corr(self):
        # Returns (corr_matrix, std_vector)
        cov = np.array([[4.0, 2.0], [2.0, 9.0]])
        corr, std = features.cov_to_corr(cov)
        assert corr.shape == (2, 2)
        assert corr[0, 0] == pytest.approx(1.0)
        assert len(std) == 2
        assert std[0] == pytest.approx(2.0)

    def test_corr_to_cov(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        std = np.array([2.0, 3.0])
        result = features.corr_to_cov(corr, std)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(4.0)
        assert result[1, 1] == pytest.approx(9.0)


class TestAllocation:
    def test_hrp_weights(self):
        # hrp_weights takes 2D returns matrix, not cov
        returns = np.array([[0.01, -0.005, 0.002],
                           [-0.005, 0.02, -0.01],
                           [0.015, 0.008, 0.003],
                           [-0.002, -0.01, 0.01],
                           [0.01, 0.005, -0.005]])
        result = features.hrp_weights(returns)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert sum(result) == pytest.approx(1.0, abs=0.01)

    def test_inverse_variance_weights(self):
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        result = features.inverse_variance_weights(cov)
        assert len(result) == 2
        assert sum(result) == pytest.approx(1.0, abs=0.01)


class TestClustering:
    def test_kmeans(self):
        data = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0], [5.1, 5.1]])
        result = features.kmeans(data, 2, seed=42)
        assert hasattr(result, "labels")
        assert hasattr(result, "n_iterations")
        assert len(result.labels) == 4

    def test_silhouette_score(self):
        data = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0], [5.1, 5.1]])
        labels = [0, 0, 1, 1]
        score = features.silhouette_score(data, labels)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


class TestCodependence:
    def test_spearmans_rho(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = features.spearmans_rho(x, y)
        assert isinstance(result, float)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_distance_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = features.distance_correlation(x, y)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_dependence_matrix(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = features.dependence_matrix(data, "pearson")
        assert result.shape == (2, 2)

    def test_distance_matrix(self):
        # First get a correlation matrix, then compute distance
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        corr = features.dependence_matrix(data, "pearson")
        result = features.distance_matrix(corr, "angular")
        assert result.shape == (2, 2)
