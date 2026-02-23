"""Tests for pymlfinance.backtesting module."""

import numpy as np
import pytest
from pymlfinance import backtesting


class TestStatistics:
    def test_sharpe_ratio(self, returns_series):
        result = backtesting.sharpe_ratio(returns_series)
        assert isinstance(result, float)

    def test_probabilistic_sharpe_ratio(self, returns_series):
        # probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_observations, skewness, kurtosis)
        sr = backtesting.sharpe_ratio(returns_series)
        result = backtesting.probabilistic_sharpe_ratio(sr, 0.0, len(returns_series), 0.0, 3.0)
        assert isinstance(result, float)

    def test_deflated_sharpe_ratio(self, returns_series):
        # deflated_sharpe_ratio(observed_sr, sr_std, n_observations, n_trials, skewness, kurtosis)
        sr = backtesting.sharpe_ratio(returns_series)
        result = backtesting.deflated_sharpe_ratio(sr, 0.5, len(returns_series), 10, 0.0, 3.0)
        assert isinstance(result, float)

    def test_compute_drawdowns(self, returns_series):
        equity = np.cumsum(returns_series) + 100
        result = backtesting.compute_drawdowns(equity)
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "max_drawdown_duration")
        assert result.max_drawdown >= 0

    def test_hhi(self, returns_series):
        result = backtesting.hhi(returns_series)
        assert isinstance(result, float)

    def test_hhi_concentration(self, returns_series):
        # Returns (hhi_pos, hhi_neg) tuple
        result = backtesting.hhi_concentration(returns_series)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hit_ratio(self, returns_series):
        result = backtesting.hit_ratio(returns_series)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_avg_holding_period(self):
        # avg_holding_period takes list of (entry, exit) tuples
        pairs = [(0, 5), (10, 15), (20, 30)]
        result = backtesting.avg_holding_period(pairs)
        assert isinstance(result, float)
        assert result > 0

    def test_turnover(self):
        positions = np.array([1.0, 1.0, 0.0, -1.0, -1.0, 0.0])
        result = backtesting.turnover(positions)
        assert isinstance(result, float)


class TestOverfitting:
    def test_bonferroni_correction(self):
        # bonferroni_correction(p_values) - just takes array
        p_values = np.array([0.01, 0.04, 0.03, 0.05])
        result = backtesting.bonferroni_correction(p_values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4

    def test_holm_correction(self):
        # holm_correction(p_values)
        p_values = np.array([0.01, 0.04, 0.03, 0.05])
        result = backtesting.holm_correction(p_values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4


class TestStrategyRisk:
    def test_sr_from_precision(self):
        # sr_from_precision(precision, freq, avg_win_loss_ratio)
        result = backtesting.sr_from_precision(0.55, 252.0, 1.0)
        assert isinstance(result, float)

    def test_implied_precision(self):
        # implied_precision(target_sr, freq, avg_win_loss_ratio)
        result = backtesting.implied_precision(1.0, 252.0, 1.0)
        assert isinstance(result, float)

    def test_implied_frequency(self):
        # implied_frequency(target_sr, precision, avg_win_loss_ratio)
        result = backtesting.implied_frequency(1.0, 0.55, 1.0)
        assert isinstance(result, float)

    def test_strategy_failure_probability(self):
        # strategy_failure_probability(estimated_precision, n_observations, break_even_precision)
        result = backtesting.strategy_failure_probability(0.55, 252, 0.5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestBetSizing:
    def test_sigmoid_bet_size(self):
        # sigmoid_bet_size(prob, num_classes)
        result = backtesting.sigmoid_bet_size(0.8, 2)
        assert isinstance(result, float)

    def test_power_bet_size(self):
        # power_bet_size(prob, num_classes, exponent)
        result = backtesting.power_bet_size(0.8, 2, 2.0)
        assert isinstance(result, float)

    def test_discrete_signal(self):
        # discrete_signal(signal, step_size) -> single float
        result = backtesting.discrete_signal(0.73, 0.1)
        assert isinstance(result, float)

    def test_avg_active_signals(self):
        # avg_active_signals(signals: list[(start, end, value)], num_bars)
        signals = [(0, 3, 0.5), (1, 4, 0.3), (2, 5, -0.2)]
        result = backtesting.avg_active_signals(signals, 6)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6


class TestSynthetic:
    def test_simulate_ou(self):
        # simulate_ou(theta, mu, sigma, x0, dt, n_steps, seed)
        result = backtesting.simulate_ou(0.5, 0.0, 0.1, 0.0, 0.01, 1000, 42)
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1000

    def test_estimate_ou_params(self):
        # estimate_ou_params(series, dt) -> (theta, mu, sigma)
        data = backtesting.simulate_ou(0.5, 0.0, 0.1, 0.0, 0.01, 500, 42)
        theta, mu, sigma = backtesting.estimate_ou_params(data, 0.01)
        assert theta > 0
        assert isinstance(mu, float)
        assert sigma > 0
