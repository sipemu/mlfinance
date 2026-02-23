import polars as pl
import pymlfinance.polars  # noqa: F401
import pytest
import math


@pytest.fixture
def price_df():
    return pl.DataFrame(
        {
            "a": [100.0, 102.0, 101.0, 105.0, 103.0],
            "b": [50.0, 51.0, 52.0, 53.0, 54.0],
        }
    )


class TestEwma:
    def test_basic(self, price_df):
        result = price_df.select(pl.col("a").ml.ewma(span=3))
        assert result.shape == (5, 1)
        assert result["a"][0] == pytest.approx(100.0)

    def test_multi_column_parallel(self, price_df):
        result = price_df.with_columns(
            pl.col("a", "b").ml.ewma(span=3).name.suffix("_ewma")
        )
        assert "a_ewma" in result.columns
        assert "b_ewma" in result.columns
        assert result.shape == (5, 4)


class TestEwmaStd:
    def test_basic(self, price_df):
        result = price_df.select(pl.col("a").ml.ewma_std(span=10))
        assert result.shape == (5, 1)
        assert result["a"][0] == pytest.approx(0.0)
        assert result["a"][1] > 0.0


class TestCumsum:
    def test_basic(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(pl.col("x").ml.cumsum())
        assert result["x"].to_list() == pytest.approx([1.0, 3.0, 6.0, 10.0])


class TestLogReturns:
    def test_basic(self, price_df):
        result = price_df.select(pl.col("a").ml.log_returns())
        assert result.shape == (5, 1)
        # First element should be NaN (padding)
        assert math.isnan(result["a"][0])
        # Second element: ln(102/100)
        assert result["a"][1] == pytest.approx(math.log(102.0 / 100.0))


class TestSimpleReturns:
    def test_basic(self, price_df):
        result = price_df.select(pl.col("a").ml.simple_returns())
        assert result.shape == (5, 1)
        assert math.isnan(result["a"][0])
        assert result["a"][1] == pytest.approx(0.02)

    def test_multi_column(self, price_df):
        result = price_df.with_columns(
            pl.col("a", "b").ml.simple_returns().name.suffix("_ret")
        )
        assert "a_ret" in result.columns
        assert "b_ret" in result.columns
