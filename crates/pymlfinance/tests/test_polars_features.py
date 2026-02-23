import polars as pl
import pymlfinance.polars  # noqa: F401
import pytest
import math


@pytest.fixture
def price_df():
    return pl.DataFrame({"prices": [100.0 + i * 0.1 + (i % 3) * 0.5 for i in range(100)]})


@pytest.fixture
def return_df():
    import random
    random.seed(42)
    return pl.DataFrame({"rets": [random.gauss(0, 0.01) for _ in range(100)]})


class TestAdfTest:
    def test_basic(self, price_df):
        result = price_df.select(pl.col("prices").ml.adf_test(max_lags=1))
        assert result.shape == (1, 1)
        assert result["prices"][0] is not None


class TestSadf:
    def test_basic(self, price_df):
        result = price_df.select(pl.col("prices").ml.sadf(min_window=20, max_lags=1))
        assert result.shape == (100, 1)


class TestBinaryEncode:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.binary_encode())
        assert result.shape == (100, 1)
        # Values should be 0 or 1
        vals = result["rets"].to_list()
        assert all(v in (0, 1) for v in vals)


class TestQuantileEncode:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.quantile_encode(n_bins=5))
        assert result.shape == (100, 1)


class TestSigmaEncode:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.sigma_encode(n_bands=3))
        assert result.shape == (100, 1)


class TestLempelZivComplexity:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.lempel_ziv_complexity())
        assert result.shape == (1, 1)
        assert result["rets"][0] > 0


class TestShannonEntropy:
    def test_basic(self):
        # Use probability-like values
        df = pl.DataFrame({"probs": [0.25, 0.25, 0.25, 0.25]})
        result = df.select(pl.col("probs").ml.shannon_entropy())
        assert result.shape == (1, 1)


class TestPluginEntropy:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.plugin_entropy())
        assert result.shape == (1, 1)


class TestKontoyiannisEntropy:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.kontoyiannis_entropy(window=10))
        assert result.shape == (1, 1)


class TestTickRuleClassify:
    def test_basic(self):
        df = pl.DataFrame({"prices": [100.0, 101.0, 100.5, 100.5, 101.5]})
        result = df.select(pl.col("prices").ml.tick_rule_classify())
        assert result.shape == (5, 1)
        vals = result["prices"].to_list()
        assert vals[1] == pytest.approx(1.0)   # up tick
        assert vals[2] == pytest.approx(-1.0)  # down tick
