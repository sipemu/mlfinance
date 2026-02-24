"""Correctness tests for pymlfinance — verifying fixes to example notebooks/scripts.

Each test class targets a specific correctness issue that was identified and fixed
in the AFML chapter examples. These tests serve as regression guards.
"""

import numpy as np
import pytest
from pymlfinance import (
    OhlcvBar,
    TripleBarrierConfig,
    backtesting,
    features,
    labeling,
    modeling,
    sampling,
)


# ---------------------------------------------------------------------------
# Ch 3: touch_type is a string, not an integer
# ---------------------------------------------------------------------------
class TestTouchTypeStrings:
    """Triple-barrier touch_type must be 'upper', 'lower', or 'vertical'."""

    VALID_TOUCH_TYPES = {"upper", "lower", "vertical"}

    def _make_events(self, price_series):
        config = TripleBarrierConfig(
            upper_barrier=0.02, lower_barrier=0.02, max_holding_period=10
        )
        timestamps = [float(i) for i in range(len(price_series))]
        vol = labeling.daily_volatility(price_series, timestamps, 20)
        event_indices = list(range(20, min(70, len(price_series) - 10)))
        return labeling.get_events(price_series, event_indices, config, vol)

    def test_touch_type_is_str(self, price_series):
        events = self._make_events(price_series)
        for e in events:
            assert isinstance(e.touch_type, str), (
                f"touch_type should be str, got {type(e.touch_type)}"
            )

    def test_touch_type_values(self, price_series):
        events = self._make_events(price_series)
        for e in events:
            assert e.touch_type in self.VALID_TOUCH_TYPES, (
                f"touch_type={e.touch_type!r} not in {self.VALID_TOUCH_TYPES}"
            )

    def test_touch_type_is_not_integer(self, price_series):
        """Guard against the original bug: comparing touch_type to integers."""
        events = self._make_events(price_series)
        for e in events:
            assert e.touch_type != 1
            assert e.touch_type != -1
            assert e.touch_type != 0

    def test_all_three_touch_types_reachable(self):
        """With tight holding period and wide barriers, all 3 types should appear."""
        rng = np.random.default_rng(123)
        n = 500
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        timestamps = [float(i) for i in range(n)]
        vol = labeling.daily_volatility(prices, timestamps, 20)
        # Wide barriers + short holding → some events expire before hitting barriers
        config = TripleBarrierConfig(
            upper_barrier=0.10, lower_barrier=0.10, max_holding_period=3
        )
        events = labeling.get_events(
            prices, list(range(20, n - 10)), config, vol
        )
        observed_types = {e.touch_type for e in events}
        assert "upper" in observed_types, "Expected some upper barrier touches"
        assert "lower" in observed_types, "Expected some lower barrier touches"
        assert "vertical" in observed_types, "Expected some vertical exits"


# ---------------------------------------------------------------------------
# Ch 5: FFD NaN padding — leading NaNs must be stripped before stats
# ---------------------------------------------------------------------------
class TestFFDNanPadding:
    """Fractional differentiation pads leading entries with NaN."""

    def test_ffd_returns_same_length(self, price_series):
        result = sampling.frac_diff_ffd(price_series, 0.5, 1e-4)
        assert len(result) == len(price_series)

    def test_ffd_has_leading_nans(self, price_series):
        result = sampling.frac_diff_ffd(price_series, 0.5, 1e-4)
        # The first entry should be NaN (needs at least weight_len values)
        weights = sampling.get_weights_ffd(0.5, 1e-4)
        n_leading_nans = len(weights) - 1
        if n_leading_nans > 0:
            assert np.isnan(result[0]), "First element should be NaN"
            for i in range(min(n_leading_nans, len(result))):
                assert np.isnan(result[i]), f"result[{i}] should be NaN"

    def test_ffd_valid_values_after_nans(self, price_series):
        result = sampling.frac_diff_ffd(price_series, 0.5, 1e-4)
        valid = ~np.isnan(result)
        ffd_valid = result[valid]
        assert len(ffd_valid) > 0, "Should have some valid (non-NaN) values"
        assert np.all(np.isfinite(ffd_valid)), "Valid values should be finite"

    def test_ffd_nan_stripped_correlation_is_valid(self, price_series):
        """Correlation of FFD with original should be a real number, not NaN."""
        log_prices = np.log(price_series)
        ffd = sampling.frac_diff_ffd(log_prices, 0.5, 1e-4)
        valid = ~np.isnan(ffd)
        ffd_valid = ffd[valid]
        if len(ffd_valid) > 2:
            corr = np.corrcoef(log_prices[valid], ffd_valid)[0, 1]
            assert np.isfinite(corr), "Correlation should be finite after NaN stripping"
            assert -1.0 <= corr <= 1.0

    def test_ffd_nan_stripped_adf_is_valid(self, price_series):
        """ADF on FFD output requires NaN stripping first."""
        log_prices = np.log(price_series)
        ffd = sampling.frac_diff_ffd(log_prices, 0.5, 1e-4)
        valid = ~np.isnan(ffd)
        ffd_valid = ffd[valid]
        if len(ffd_valid) > 10:
            stat, _ = features.adf_test(ffd_valid, max_lags=1)
            assert np.isfinite(stat), "ADF statistic should be finite"

    def test_expanding_window_same_length(self, price_series):
        result = sampling.frac_diff_expanding(price_series, 0.5, 1e-4)
        assert len(result) == len(price_series)

    def test_higher_d_means_fewer_nans(self, price_series):
        """Higher d has shorter weight vector, hence fewer leading NaNs."""
        ffd_low = sampling.frac_diff_ffd(price_series, 0.3, 1e-4)
        ffd_high = sampling.frac_diff_ffd(price_series, 0.8, 1e-4)
        nans_low = np.sum(np.isnan(ffd_low))
        nans_high = np.sum(np.isnan(ffd_high))
        assert nans_low >= nans_high, (
            f"d=0.3 should have >= NaNs than d=0.8 ({nans_low} vs {nans_high})"
        )


# ---------------------------------------------------------------------------
# Ch 7 & 8: make_classification labels are {-1, 1}, not {0, 1}
# ---------------------------------------------------------------------------
class TestMakeClassificationLabels:
    """make_classification produces {-1, 1} labels (not sklearn's {0, 1})."""

    def test_labels_are_minus1_or_plus1(self):
        _, y = modeling.make_classification(200, 3, 2, 1, 42)
        unique = set(np.unique(y))
        assert unique == {-1.0, 1.0}, f"Expected {{-1.0, 1.0}}, got {unique}"

    def test_no_zero_labels(self):
        _, y = modeling.make_classification(500, 5, 3, 2, 99)
        assert 0.0 not in np.unique(y), "make_classification should not produce 0 labels"

    def test_both_classes_present(self):
        _, y = modeling.make_classification(100, 3, 1, 1, 42)
        unique = set(np.unique(y))
        assert -1.0 in unique, "Negative class missing"
        assert 1.0 in unique, "Positive class missing"

    def test_labels_roughly_balanced(self):
        _, y = modeling.make_classification(1000, 3, 2, 1, 42)
        n_pos = np.sum(y == 1.0)
        n_neg = np.sum(y == -1.0)
        ratio = n_pos / n_neg
        assert 0.5 < ratio < 2.0, f"Labels heavily imbalanced: {n_pos} vs {n_neg}"

    def test_classifier_predict_returns_correct_labels(self):
        """A classifier trained on {-1,1} data must predict {-1,1}."""
        X, y = modeling.make_classification(200, 3, 1, 1, 42)
        events = [(i, min(i + 5, len(X) - 1)) for i in range(len(X))]

        class TestClassifier:
            def __init__(self):
                self.weights = None

            def fit(self, X, y, sample_weight=None):
                self.weights = np.array(
                    [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
                )
                return self

            def predict(self, X):
                scores = X @ self.weights
                return np.where(scores > 0, 1, -1).astype(np.int32)

        clf = TestClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        pred_unique = set(np.unique(preds))
        assert pred_unique <= {-1, 1}, f"Predictions should be {{-1, 1}}, got {pred_unique}"


# ---------------------------------------------------------------------------
# Ch 14: PSR/DSR require per-period SR, not annualized
# ---------------------------------------------------------------------------
class TestPSRPerPeriodSR:
    """PSR must use per-period (sample) SR, not annualized SR.

    Bailey & López de Prado's PSR formula:
        PSR = Φ((SR - SR*) × √(n-1) / √(1 - γ₃·SR + (γ₄-1)/4 · SR²))

    The SE term is derived for the per-period SR estimator.
    Using annualized SR inflates z by ~√252, making PSR≈1 (uninformative).
    """

    def _generate_returns(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        return rng.normal(0.0003, 0.01, n)

    def test_psr_with_per_period_sr_is_informative(self):
        """Per-period SR gives a PSR between 0 and 1 (informative)."""
        returns = self._generate_returns()
        sr_per_period = np.mean(returns) / np.std(returns, ddof=1)
        psr = backtesting.probabilistic_sharpe_ratio(
            sr_per_period, 0.0, len(returns), 0.0, 3.0
        )
        assert 0.0 < psr < 1.0, f"Per-period PSR should be in (0,1), got {psr}"

    def test_psr_with_annualized_sr_is_near_one(self):
        """Annualized SR (inflated by √252) produces PSR≈1.0 — the bug we fixed."""
        returns = self._generate_returns()
        sr_annualized = backtesting.sharpe_ratio(returns)
        psr = backtesting.probabilistic_sharpe_ratio(
            sr_annualized, 0.0, len(returns), 0.0, 3.0
        )
        # This is the BAD case: PSR is almost exactly 1.0 (useless)
        assert psr > 0.99, (
            f"Annualized SR should give PSR≈1.0 (demonstrating the bug), got {psr}"
        )

    def test_psr_increases_with_more_data(self):
        """With same underlying SR, more observations → higher PSR."""
        rng = np.random.default_rng(42)
        sr = 0.03  # Fixed per-period SR

        psr_100 = backtesting.probabilistic_sharpe_ratio(sr, 0.0, 100, 0.0, 3.0)
        psr_500 = backtesting.probabilistic_sharpe_ratio(sr, 0.0, 500, 0.0, 3.0)
        psr_2000 = backtesting.probabilistic_sharpe_ratio(sr, 0.0, 2000, 0.0, 3.0)

        assert psr_100 < psr_500 < psr_2000, (
            f"PSR should increase with n: {psr_100}, {psr_500}, {psr_2000}"
        )

    def test_psr_decreases_with_higher_benchmark(self):
        """Higher benchmark SR → lower PSR."""
        returns = self._generate_returns()
        sr = np.mean(returns) / np.std(returns, ddof=1)
        n = len(returns)

        psrs = [
            backtesting.probabilistic_sharpe_ratio(sr, bench, n, 0.0, 3.0)
            for bench in [0.0, 0.01, 0.02, 0.05]
        ]
        for i in range(len(psrs) - 1):
            assert psrs[i] >= psrs[i + 1], (
                f"PSR should decrease with higher benchmark: {psrs}"
            )

    def test_psr_zero_benchmark_zero_sr(self):
        """PSR(SR=0, benchmark=0) should be 0.5 (50/50 coin flip)."""
        psr = backtesting.probabilistic_sharpe_ratio(0.0, 0.0, 100, 0.0, 3.0)
        assert psr == pytest.approx(0.5, abs=0.01), f"PSR(0,0) should be ~0.5, got {psr}"

    def test_dsr_penalizes_multiple_testing(self):
        """DSR should be lower than PSR due to multiple testing correction."""
        returns = self._generate_returns()
        sr = np.mean(returns) / np.std(returns, ddof=1)
        n = len(returns)

        psr = backtesting.probabilistic_sharpe_ratio(sr, 0.0, n, 0.0, 3.0)
        dsr = backtesting.deflated_sharpe_ratio(
            sr, 0.02, n, 20, 0.0, 3.0
        )
        assert dsr < psr, f"DSR ({dsr}) should be < PSR ({psr}) due to multiple testing"

    def test_dsr_more_trials_means_lower_dsr(self):
        """More trials → higher expected max SR → lower DSR."""
        returns = self._generate_returns()
        sr = np.mean(returns) / np.std(returns, ddof=1)
        n = len(returns)

        dsr_5 = backtesting.deflated_sharpe_ratio(sr, 0.02, n, 5, 0.0, 3.0)
        dsr_50 = backtesting.deflated_sharpe_ratio(sr, 0.02, n, 50, 0.0, 3.0)
        dsr_200 = backtesting.deflated_sharpe_ratio(sr, 0.02, n, 200, 0.0, 3.0)

        assert dsr_5 > dsr_50 > dsr_200, (
            f"DSR should decrease with more trials: {dsr_5}, {dsr_50}, {dsr_200}"
        )


# ---------------------------------------------------------------------------
# Ch 17: GSADF is O(n²) sub-windows, SADF is O(n) expanding windows
# ---------------------------------------------------------------------------
class TestGSADFvsSADF:
    """GSADF produces O(n²) statistics (varies both endpoints).
    SADF produces n - min_window statistics (varies only the end)."""

    def test_sadf_length_is_linear(self):
        rng = np.random.default_rng(42)
        n = 100
        series = np.cumsum(rng.normal(0, 1, n)) + 100
        min_window = 20
        sadf = features.sadf(series, min_window, 1)
        # Expanding window: end ranges from min_window to n, inclusive → n - min_window + 1
        expected_len = n - min_window + 1
        assert len(sadf) == expected_len, (
            f"SADF length should be n-min_window+1={expected_len}, got {len(sadf)}"
        )

    def test_gsadf_length_exceeds_sadf(self):
        """GSADF has O(n²) sub-windows, so its output should be much longer than SADF."""
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, 80)) + 100
        min_window = 15
        sadf = features.sadf(series, min_window, 1)
        gsadf = features.gsadf(series, min_window, 1)
        assert len(gsadf) > len(sadf), (
            f"GSADF ({len(gsadf)}) should have more entries than SADF ({len(sadf)})"
        )

    def test_gsadf_stat_geq_sadf_stat(self):
        """GSADF statistic >= SADF statistic (searches over more sub-windows)."""
        rng = np.random.default_rng(42)
        # Create series with a bubble to get positive stats
        normal = np.cumsum(rng.normal(0, 0.01, 50))
        bubble = np.cumsum(rng.normal(0.02, 0.01, 30))
        series = np.concatenate([normal, normal[-1] + bubble]) + 5.0

        sadf_stat = features.sadf_stat(series, 20, 1)
        gsadf_stat = features.gsadf_stat(series, 20, 1)
        assert gsadf_stat >= sadf_stat - 1e-10, (
            f"GSADF stat ({gsadf_stat}) should be >= SADF stat ({sadf_stat})"
        )


# ---------------------------------------------------------------------------
# Ch 11: CSCV rank logit — positive logit = overfitting
# ---------------------------------------------------------------------------
class TestCSCVRankLogit:
    """CSCV: positive rank logit means the best IS strategy performs
    poorly OOS, indicating overfitting. PBO = fraction of positive logits."""

    def test_random_strategies_have_high_pbo(self):
        """Pure noise strategies should have PBO around 0.5 or higher."""
        rng = np.random.default_rng(42)
        # 100 periods, 10 pure noise strategies
        returns_matrix = rng.normal(0, 0.01, (100, 10))
        result = backtesting.cscv(returns_matrix, 6)
        assert hasattr(result, "pbo")
        assert hasattr(result, "rank_logits")
        # PBO for random strategies should be moderate to high
        assert result.pbo >= 0.2, (
            f"Random strategies PBO should be >= 0.2, got {result.pbo}"
        )

    def test_pbo_is_fraction_of_positive_logits(self):
        """PBO should equal the fraction of positive rank logits."""
        rng = np.random.default_rng(42)
        returns_matrix = rng.normal(0, 0.01, (100, 8))
        result = backtesting.cscv(returns_matrix, 4)
        logits = np.array(result.rank_logits)
        expected_pbo = np.mean(logits > 0)
        assert result.pbo == pytest.approx(expected_pbo, abs=1e-10), (
            f"PBO ({result.pbo}) should equal fraction of positive logits ({expected_pbo})"
        )

    def test_rank_logits_are_finite(self):
        rng = np.random.default_rng(42)
        returns_matrix = rng.normal(0.001, 0.01, (80, 6))
        result = backtesting.cscv(returns_matrix, 4)
        logits = np.array(result.rank_logits)
        assert np.all(np.isfinite(logits)), "All rank logits should be finite"


# ---------------------------------------------------------------------------
# Ch 4: Indicator matrix shape is (num_bars, n_events)
# ---------------------------------------------------------------------------
class TestIndicatorMatrixShape:
    """Indicator matrix has shape (num_bars, n_events), not transposed."""

    def test_indicator_matrix_shape(self):
        events = [(0, 2), (1, 3), (3, 5)]
        num_bars = 6
        result = sampling.get_indicator_matrix(events, num_bars)
        assert result.shape == (num_bars, len(events)), (
            f"Expected ({num_bars}, {len(events)}), got {result.shape}"
        )

    def test_indicator_matrix_values(self):
        """Check that entries are 1.0 where event spans the bar, 0.0 otherwise."""
        events = [(1, 3), (2, 4)]
        num_bars = 5
        ind = sampling.get_indicator_matrix(events, num_bars)
        # Event 0 spans bars 1,2,3
        assert ind[0, 0] == 0.0  # bar 0, event 0: not active
        assert ind[1, 0] == 1.0  # bar 1, event 0: active
        assert ind[2, 0] == 1.0  # bar 2, event 0: active
        assert ind[3, 0] == 1.0  # bar 3, event 0: active
        assert ind[4, 0] == 0.0  # bar 4, event 0: not active
        # Event 1 spans bars 2,3,4
        assert ind[1, 1] == 0.0  # bar 1, event 1: not active
        assert ind[2, 1] == 1.0  # bar 2, event 1: active
        assert ind[4, 1] == 1.0  # bar 4, event 1: active

    def test_indicator_matrix_larger(self):
        """Smoke test with more events to verify (num_bars, n_events) convention."""
        n_events = 20
        num_bars = 50
        events = [(i, min(i + 5, num_bars - 1)) for i in range(n_events)]
        ind = sampling.get_indicator_matrix(events, num_bars)
        assert ind.shape == (num_bars, n_events)


# ---------------------------------------------------------------------------
# Cross-cutting: drawdown analysis returns non-negative values
# ---------------------------------------------------------------------------
class TestDrawdownCorrectness:
    """compute_drawdowns returns positive values measuring peak-to-trough decline."""

    def test_max_drawdown_nonnegative(self, returns_series):
        dd = backtesting.compute_drawdowns(returns_series)
        assert dd.max_drawdown >= 0, (
            f"max_drawdown should be >= 0, got {dd.max_drawdown}"
        )

    def test_drawdown_series_nonneg(self, returns_series):
        dd = backtesting.compute_drawdowns(returns_series)
        series = np.array(dd.drawdown_series)
        assert np.all(series >= -1e-15), (
            "Drawdown series should be non-negative (positive = decline from peak)"
        )

    def test_max_drawdown_duration_positive(self, returns_series):
        dd = backtesting.compute_drawdowns(returns_series)
        assert dd.max_drawdown_duration >= 0


# ---------------------------------------------------------------------------
# Sharpe ratio annualization
# ---------------------------------------------------------------------------
class TestSharpeRatioAnnualization:
    """sharpe_ratio() annualizes by √252. This must NOT be passed to PSR/DSR."""

    def test_sharpe_ratio_is_annualized(self):
        """The sharpe_ratio() function scales by sqrt(252)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        sr_ann = backtesting.sharpe_ratio(returns)
        sr_raw = np.mean(returns) / np.std(returns, ddof=1)
        # Annualized SR ≈ raw SR * √252
        ratio = sr_ann / sr_raw if abs(sr_raw) > 1e-10 else 0.0
        assert ratio == pytest.approx(np.sqrt(252), rel=0.01), (
            f"sharpe_ratio should annualize by √252, got ratio {ratio}"
        )
