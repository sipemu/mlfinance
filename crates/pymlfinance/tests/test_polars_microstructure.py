import polars as pl
import pymlfinance.polars  # noqa: F401
from pymlfinance.polars._lib import (
    amihud_lambda,
    amihud_lambda_rolling,
    kyle_lambda,
    roll_spread_rolling,
    corwin_schultz_spread,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
)
import pytest


@pytest.fixture
def market_df():
    return pl.DataFrame(
        {
            "returns": [0.01, -0.02, 0.005, 0.015, -0.01, 0.008, -0.003, 0.012, -0.007, 0.009],
            "dollar_vol": [1e6, 2e6, 1.5e6, 1e6, 2e6, 1.2e6, 1.8e6, 1.1e6, 1.6e6, 1.3e6],
            "signed_vol": [100.0, -50.0, 200.0, -100.0, 150.0, -80.0, 120.0, -60.0, 90.0, -40.0],
        }
    )


@pytest.fixture
def ohlc_df():
    return pl.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5],
            "high": [102.0, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5, 107.0, 106.5],
            "low":  [ 99.0, 100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5],
            "close":[101.0, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5],
        }
    )


class TestAmihudLambda:
    def test_scalar(self, market_df):
        result = market_df.select(amihud_lambda("returns", "dollar_vol"))
        assert result.shape == (1, 1)
        assert result["returns"][0] > 0

    def test_rolling(self, market_df):
        result = market_df.select(
            amihud_lambda_rolling("returns", "dollar_vol", window=5)
        )
        assert result.shape == (10, 1)


class TestKyleLambda:
    def test_scalar(self, market_df):
        result = market_df.select(kyle_lambda("returns", "signed_vol"))
        assert result.shape == (1, 1)


class TestRollSpreadRolling:
    def test_basic(self, ohlc_df):
        result = ohlc_df.select(roll_spread_rolling("close", window=5))
        assert result.shape == (10, 1)


class TestCorwinSchultzSpread:
    def test_basic(self, ohlc_df):
        result = ohlc_df.select(corwin_schultz_spread("high", "low"))
        assert result.shape == (10, 1)
        # All values should be non-negative (NaN for first, >=0 for rest)
        vals = result["high"].to_list()
        for v in vals[1:]:
            if v is not None:
                assert v >= 0.0


class TestParkinsonVolatility:
    def test_basic(self, ohlc_df):
        result = ohlc_df.select(parkinson_volatility("high", "low", window=5))
        assert result.shape == (10, 1)


class TestGarmanKlassVolatility:
    def test_basic(self, ohlc_df):
        result = ohlc_df.select(
            garman_klass_volatility("open", "high", "low", "close", window=5)
        )
        assert result.shape == (10, 1)


class TestYangZhangVolatility:
    def test_basic(self, ohlc_df):
        result = ohlc_df.select(
            yang_zhang_volatility("open", "high", "low", "close", window=5)
        )
        assert result.shape == (10, 1)
