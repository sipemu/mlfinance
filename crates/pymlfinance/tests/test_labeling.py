"""Tests for pymlfinance.labeling module."""

import numpy as np
import pytest
from pymlfinance import OhlcvBar, TripleBarrierConfig, labeling


class TestVolatility:
    def _make_bars(self, n=50):
        bars = []
        for i in range(n):
            bars.append(
                OhlcvBar(
                    float(i),
                    100.0 + i * 0.1,  # open
                    102.0 + i * 0.1,  # high
                    98.0 + i * 0.1,   # low
                    101.0 + i * 0.1,  # close
                    1000.0,            # volume
                )
            )
        return bars

    def test_daily_volatility(self, price_series):
        # daily_volatility takes (prices, timestamps, span)
        timestamps = [float(i) for i in range(len(price_series))]
        result = labeling.daily_volatility(price_series, timestamps, 20)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(price_series)

    def test_parkinson_volatility(self):
        bars = self._make_bars(20)
        result = labeling.parkinson_volatility(bars, 5)
        assert isinstance(result, np.ndarray)

    def test_garman_klass_volatility(self):
        bars = self._make_bars(20)
        result = labeling.garman_klass_volatility(bars, 5)
        assert isinstance(result, np.ndarray)

    def test_yang_zhang_volatility(self):
        bars = self._make_bars(20)
        result = labeling.yang_zhang_volatility(bars, 5)
        assert isinstance(result, np.ndarray)


class TestEvents:
    def test_get_events(self, price_series):
        config = TripleBarrierConfig(
            upper_barrier=0.02, lower_barrier=0.02, max_holding_period=10
        )
        timestamps = [float(i) for i in range(len(price_series))]
        vol = labeling.daily_volatility(price_series, timestamps, 20)
        event_indices = list(range(20, 70))
        events = labeling.get_events(price_series, event_indices, config, vol)
        assert isinstance(events, list)
        assert len(events) > 0
        for e in events:
            assert hasattr(e, "entry_idx")
            assert hasattr(e, "exit_idx")
            assert hasattr(e, "touch_type")
            assert e.touch_type in ("upper", "lower", "vertical")

    def test_get_bins(self, price_series):
        config = TripleBarrierConfig(
            upper_barrier=0.02, lower_barrier=0.02, max_holding_period=10
        )
        timestamps = [float(i) for i in range(len(price_series))]
        vol = labeling.daily_volatility(price_series, timestamps, 20)
        event_indices = list(range(20, 70))
        events = labeling.get_events(price_series, event_indices, config, vol)
        labels = labeling.get_bins(events)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(events)

    def test_drop_rare_labels(self):
        labels = [1, 1, 1, -1, -1, 0]
        result = labeling.drop_rare_labels(labels, 0.1)
        assert isinstance(result, list)


class TestTrendScanning:
    def test_trend_scanning_labels(self, price_series):
        result = labeling.trend_scanning_labels(price_series, max_window=20)
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= len(price_series)
        for r in result:
            assert hasattr(r, "t_stat")
            assert hasattr(r, "label")
            assert r.label in (-1, 0, 1)

    def test_trend_scanning_label_series(self, price_series):
        result = labeling.trend_scanning_label_series(price_series, 20)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert len(result) <= len(price_series)


class TestMetaLabeler:
    def test_meta_labeler(self):
        labeler = labeling.MetaLabeler(0.5)
        bet = labeler.bet_size(0.7)
        assert isinstance(bet, float)
