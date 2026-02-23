import polars as pl
import pymlfinance.polars  # noqa: F401
import pytest
import math


@pytest.fixture
def return_df():
    return pl.DataFrame(
        {"rets": [0.01, -0.02, 0.03, -0.01, 0.015, 0.005, -0.008, 0.02, -0.005, 0.01]}
    )


@pytest.fixture
def prob_df():
    return pl.DataFrame(
        {"probs": [0.1, 0.3, 0.5, 0.7, 0.9]}
    )


class TestSharpeRatio:
    def test_basic(self, return_df):
        result = return_df.select(
            pl.col("rets").ml.sharpe_ratio(risk_free_rate=0.0, periods_per_year=252.0)
        )
        assert result.shape == (1, 1)
        assert result["rets"][0] is not None

    def test_with_risk_free(self, return_df):
        result = return_df.select(
            pl.col("rets").ml.sharpe_ratio(risk_free_rate=0.02, periods_per_year=252.0)
        )
        assert result.shape == (1, 1)


class TestHitRatio:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.hit_ratio())
        assert result.shape == (1, 1)
        hr = result["rets"][0]
        assert 0.0 <= hr <= 1.0
        # 6 positive out of 10
        assert hr == pytest.approx(0.6)


class TestHHI:
    def test_basic(self):
        df = pl.DataFrame({"weights": [0.25, 0.25, 0.25, 0.25]})
        result = df.select(pl.col("weights").ml.hhi())
        assert result.shape == (1, 1)
        assert result["weights"][0] == pytest.approx(0.25)

    def test_concentrated(self):
        df = pl.DataFrame({"weights": [1.0, 0.0, 0.0, 0.0]})
        result = df.select(pl.col("weights").ml.hhi())
        assert result["weights"][0] == pytest.approx(1.0)


class TestComputeDrawdowns:
    def test_basic(self, return_df):
        result = return_df.select(pl.col("rets").ml.compute_drawdowns())
        assert result.shape == (10, 1)
        # Drawdowns should be non-negative
        for v in result["rets"].to_list():
            assert v >= 0.0 or math.isnan(v)


class TestSigmoidBetSize:
    def test_basic(self, prob_df):
        result = prob_df.select(pl.col("probs").ml.sigmoid_bet_size(num_classes=2))
        assert result.shape == (5, 1)
        vals = result["probs"].to_list()
        # prob=0.5 -> size=0.0
        assert vals[2] == pytest.approx(0.0)
        # prob=0.9 -> size=0.8
        assert vals[4] == pytest.approx(0.8)

    def test_multi_class(self, prob_df):
        result = prob_df.select(pl.col("probs").ml.sigmoid_bet_size(num_classes=3))
        assert result.shape == (5, 1)


class TestPowerBetSize:
    def test_basic(self, prob_df):
        result = prob_df.select(
            pl.col("probs").ml.power_bet_size(num_classes=2, exponent=2.0)
        )
        assert result.shape == (5, 1)

    def test_exponent_one_equals_sigmoid(self, prob_df):
        sig = prob_df.select(pl.col("probs").ml.sigmoid_bet_size(num_classes=2))
        pow_ = prob_df.select(
            pl.col("probs").ml.power_bet_size(num_classes=2, exponent=1.0)
        )
        for s, p in zip(sig["probs"].to_list(), pow_["probs"].to_list()):
            assert s == pytest.approx(p, abs=1e-10)


class TestDiscreteSignal:
    def test_basic(self):
        df = pl.DataFrame({"signals": [0.34, 0.36, 0.5, -0.34, 0.0]})
        result = df.select(pl.col("signals").ml.discrete_signal(step_size=0.1))
        vals = result["signals"].to_list()
        assert vals[0] == pytest.approx(0.3)
        assert vals[1] == pytest.approx(0.4)
        assert vals[2] == pytest.approx(0.5)
        assert vals[3] == pytest.approx(-0.3)
        assert vals[4] == pytest.approx(0.0)
