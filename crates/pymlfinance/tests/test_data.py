"""Tests for pymlfinance.data module."""

import numpy as np
import pytest
from pymlfinance import TickData, data


class TestBarAggregators:
    def _make_ticks(self, n=100):
        ticks = []
        for i in range(n):
            ticks.append(TickData(float(i), 100.0 + (i % 10) * 0.1, 50.0 + i))
        return ticks

    def test_tick_bar_aggregator(self):
        agg = data.TickBarAggregator(10)
        ticks = self._make_ticks(50)
        bars = agg.process_ticks(ticks)
        assert len(bars) >= 1
        for bar in bars:
            assert bar.high >= bar.low
            assert bar.volume > 0

    def test_volume_bar_aggregator(self):
        agg = data.VolumeBarAggregator(500.0)
        ticks = self._make_ticks(50)
        bars = agg.process_ticks(ticks)
        assert len(bars) >= 1

    def test_dollar_bar_aggregator(self):
        agg = data.DollarBarAggregator(50000.0)
        ticks = self._make_ticks(50)
        bars = agg.process_ticks(ticks)
        assert len(bars) >= 1

    def test_time_bar_aggregator(self):
        agg = data.TimeBarAggregator(10)  # 10-second bars
        ticks = self._make_ticks(50)
        bars = agg.process_ticks(ticks)
        assert len(bars) >= 1

    def test_tick_imbalance_bar_aggregator(self):
        agg = data.TickImbalanceBarAggregator(10.0, 20.0)
        ticks = self._make_ticks(100)
        bars = agg.process_ticks(ticks)
        # May or may not produce bars depending on imbalance
        assert isinstance(bars, list)

    def test_volume_runs_bar_aggregator(self):
        agg = data.VolumeRunsBarAggregator(500.0, 20.0)
        ticks = self._make_ticks(100)
        bars = agg.process_ticks(ticks)
        assert isinstance(bars, list)

    def test_process_tick_single(self):
        agg = data.TickBarAggregator(5)
        tick = TickData(1.0, 100.0, 10.0)
        result = agg.process_tick(tick)
        # First tick shouldn't complete a bar
        assert result is None


class TestSampling:
    def test_cusum_filter(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0])
        indices = data.cusum_filter(values, 1.5)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_linspace_sample(self):
        indices = data.linspace_sample(0, 100, 10)
        assert len(indices) == 10
        assert indices[0] == 0

    def test_uniform_sample(self):
        indices = data.uniform_sample(10, 100, 42)
        assert len(indices) == 10
        assert all(0 <= i < 100 for i in indices)


class TestMultiProduct:
    def test_etf_trick(self):
        prices = [[100.0, 200.0], [101.0, 202.0], [102.0, 198.0]]
        weights = [[0.5, 0.5], [0.6, 0.4], [0.5, 0.5]]
        result = data.etf_trick(prices, weights)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_pca_weights(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = data.pca_weights(cov)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_roll_gaps(self):
        prices = np.array([100.0, 101.0, 102.0, 105.0, 106.0])
        roll_dates = [2]
        result = data.roll_gaps(prices, roll_dates)
        assert len(result) == len(prices)

    def test_non_negative_rolled(self):
        prices = np.array([100.0, 101.0, 102.0, 105.0, 106.0])
        roll_dates = [2]
        result = data.non_negative_rolled(prices, roll_dates)
        assert len(result) == len(prices)
        assert all(r >= 0 for r in result)
