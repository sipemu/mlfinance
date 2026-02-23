import polars as pl
import pymlfinance.polars  # noqa: F401
import pytest
import math


@pytest.fixture
def price_series():
    return pl.DataFrame({"prices": [100.0 + i * 0.5 for i in range(50)]})


class TestFracDiffFfd:
    def test_basic(self, price_series):
        result = price_series.select(
            pl.col("prices").ml.frac_diff_ffd(d=0.5, threshold=1e-4)
        )
        assert result.shape == (50, 1)

    def test_d_zero_is_identity_like(self, price_series):
        result = price_series.select(
            pl.col("prices").ml.frac_diff_ffd(d=0.0, threshold=1e-4)
        )
        # d=0 should return approximately original values
        assert result.shape == (50, 1)


class TestFracDiffExpanding:
    def test_basic(self, price_series):
        result = price_series.select(
            pl.col("prices").ml.frac_diff_expanding(d=0.5, threshold=1e-4)
        )
        assert result.shape == (50, 1)


class TestFindMinD:
    def test_basic(self, price_series):
        result = price_series.select(
            pl.col("prices").ml.find_min_d(max_d=1.0, step_size=0.1, threshold=1e-4)
        )
        # Returns scalar
        assert result.shape == (1, 1)
        val = result["prices"][0]
        assert 0.0 <= val <= 1.0
