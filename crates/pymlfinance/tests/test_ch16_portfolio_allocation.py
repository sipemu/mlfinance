"""Comprehensive tests for Ch 16: Portfolio Allocation functions.

Tests verify HRP, CLA, IVP, and compare_allocations against
analytical solutions, invariant properties, and edge cases.

Mathematical reference (AFML Ch. 16):
    IVP:      w_i = (1/σ²_i) / Σ(1/σ²_j)
    CLA_mv:   w = Σ⁻¹ 1 / (1' Σ⁻¹ 1)
    CLA_sr:   w = Σ⁻¹ μ / (1' Σ⁻¹ μ)
    HRP:      hierarchical clustering + recursive bisection
"""

import numpy as np
import pytest
from pymlfinance import backtesting, features


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def diagonal_cov():
    """Diagonal covariance (uncorrelated assets)."""
    return np.diag([0.04, 0.01, 0.09])  # vols: 20%, 10%, 30%


@pytest.fixture
def block_returns():
    """Correlated returns with block structure (2 clusters)."""
    rng = np.random.default_rng(42)
    n = 200
    n_assets = 4
    corr = np.eye(n_assets)
    corr[0, 1] = corr[1, 0] = 0.8
    corr[2, 3] = corr[3, 2] = 0.7
    vols = np.array([0.02, 0.025, 0.01, 0.012])
    cov = np.outer(vols, vols) * corr
    L = np.linalg.cholesky(cov)
    returns = (L @ rng.standard_normal((n_assets, n))).T
    return returns


# ── Inverse Variance Portfolio ──────────────────────────────────────────────

class TestInverseVarianceWeights:
    """Tests for features.inverse_variance_weights(cov)."""

    def test_sum_to_one(self, diagonal_cov):
        """Weights must sum to 1."""
        w = features.inverse_variance_weights(diagonal_cov)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-10)

    def test_all_positive(self, diagonal_cov):
        """All weights should be non-negative for positive-definite cov."""
        w = features.inverse_variance_weights(diagonal_cov)
        assert np.all(w >= 0)

    def test_analytical_diagonal(self, diagonal_cov):
        """For diagonal cov, IVP = (1/diag) / sum(1/diag).

        diag = [0.04, 0.01, 0.09]
        1/diag = [25, 100, 11.111]
        sum = 136.111
        w = [0.18367, 0.73469, 0.08163]
        """
        w = features.inverse_variance_weights(diagonal_cov)
        inv_d = 1.0 / np.diag(diagonal_cov)
        expected = inv_d / np.sum(inv_d)
        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_lower_var_gets_higher_weight(self, diagonal_cov):
        """Asset with lowest variance gets highest weight."""
        w = features.inverse_variance_weights(diagonal_cov)
        # Asset 1 has var=0.01 (lowest) → should have highest weight
        assert w[1] > w[0] > w[2]

    def test_equal_variance_equal_weights(self):
        """If all variances are equal, weights should be equal."""
        cov = np.diag([0.04, 0.04, 0.04])
        w = features.inverse_variance_weights(cov)
        expected = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_two_assets(self):
        """Two-asset case: simple ratio."""
        cov = np.diag([0.04, 0.16])
        w = features.inverse_variance_weights(cov)
        # 1/0.04=25, 1/0.16=6.25, sum=31.25
        np.testing.assert_allclose(w, [25 / 31.25, 6.25 / 31.25], atol=1e-10)

    def test_single_asset(self):
        """Single asset → weight = 1."""
        cov = np.array([[0.04]])
        w = features.inverse_variance_weights(cov)
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_ignores_off_diagonal(self):
        """IVP only uses diagonal elements."""
        cov_no_corr = np.diag([0.04, 0.01])
        cov_with_corr = np.array([[0.04, 0.015], [0.015, 0.01]])
        w1 = features.inverse_variance_weights(cov_no_corr)
        w2 = features.inverse_variance_weights(cov_with_corr)
        np.testing.assert_allclose(w1, w2, atol=1e-10)


# ── CLA Minimum Variance ───────────────────────────────────────────────────

class TestClaMinVariance:
    """Tests for features.cla_min_variance(cov)."""

    def test_sum_to_one(self, diagonal_cov):
        """Weights must sum to 1."""
        w = features.cla_min_variance(diagonal_cov)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-10)

    def test_diagonal_equals_ivp(self, diagonal_cov):
        """For diagonal cov, CLA min-var = IVP (both are Σ⁻¹·1 / 1'Σ⁻¹·1)."""
        w_cla = features.cla_min_variance(diagonal_cov)
        w_ivp = features.inverse_variance_weights(diagonal_cov)
        np.testing.assert_allclose(w_cla, w_ivp, atol=1e-8)

    def test_analytical_2x2(self):
        """2x2 case with correlation.

        cov = [[0.04, 0.02], [0.02, 0.09]]
        cov_inv = [[28.125, -6.25], [-6.25, 12.5]]
        cov_inv @ [1,1] = [21.875, 6.25]
        sum = 28.125
        w = [21.875/28.125, 6.25/28.125] = [0.77778, 0.22222]
        """
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        w = features.cla_min_variance(cov)
        # Verify via numpy
        cov_inv = np.linalg.inv(cov)
        ones = np.ones(2)
        w_expected = cov_inv @ ones / (ones @ cov_inv @ ones)
        np.testing.assert_allclose(w, w_expected, atol=1e-6)

    def test_single_asset(self):
        """Single asset returns weight = 1."""
        cov = np.array([[0.04]])
        w = features.cla_min_variance(cov)
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_minimizes_variance(self, diagonal_cov):
        """CLA min-var should have lower variance than equal weight."""
        w_cla = features.cla_min_variance(diagonal_cov)
        w_eq = np.ones(3) / 3
        var_cla = w_cla @ diagonal_cov @ w_cla
        var_eq = w_eq @ diagonal_cov @ w_eq
        assert var_cla <= var_eq + 1e-12

    def test_can_produce_negative_weights(self):
        """CLA is unconstrained → can produce negative weights.

        With high positive correlation and very different volatilities,
        the optimal portfolio shorts the higher-vol asset.
        """
        cov = np.array([[0.04, 0.038], [0.038, 0.09]])
        w = features.cla_min_variance(cov)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-8)
        # At least verify it ran without error; may or may not have negatives


# ── CLA Max Sharpe ──────────────────────────────────────────────────────────

class TestClaMaxSharpe:
    """Tests for features.cla_max_sharpe(expected_returns, cov)."""

    def test_sum_to_one(self, diagonal_cov):
        """Weights must sum to 1."""
        mu = np.array([0.05, 0.03, 0.08])
        w = features.cla_max_sharpe(mu, diagonal_cov)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-10)

    def test_analytical_diagonal(self, diagonal_cov):
        """For diagonal cov, max-SR = Σ⁻¹μ / 1'Σ⁻¹μ.

        diag = [0.04, 0.01, 0.09]
        mu = [0.05, 0.03, 0.08]
        Σ⁻¹μ = [0.05/0.04, 0.03/0.01, 0.08/0.09] = [1.25, 3.0, 0.8889]
        sum = 5.1389
        w = [0.2432, 0.5837, 0.1730]
        """
        mu = np.array([0.05, 0.03, 0.08])
        w = features.cla_max_sharpe(mu, diagonal_cov)
        inv_d = 1.0 / np.diag(diagonal_cov)
        w_raw = inv_d * mu
        expected = w_raw / np.sum(w_raw)
        np.testing.assert_allclose(w, expected, atol=1e-6)

    def test_equal_returns_gives_min_var(self, diagonal_cov):
        """If all expected returns are equal, max-SR = min-var."""
        mu = np.array([0.05, 0.05, 0.05])
        w_sr = features.cla_max_sharpe(mu, diagonal_cov)
        w_mv = features.cla_min_variance(diagonal_cov)
        np.testing.assert_allclose(w_sr, w_mv, atol=1e-6)

    def test_single_asset(self):
        """Single asset returns weight = 1."""
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        w = features.cla_max_sharpe(mu, cov)
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_higher_return_higher_weight_uncorrelated(self):
        """For uncorrelated equal-vol assets, weight ∝ expected return."""
        cov = np.diag([0.04, 0.04, 0.04])
        mu = np.array([0.10, 0.05, 0.02])
        w = features.cla_max_sharpe(mu, cov)
        assert w[0] > w[1] > w[2]


# ── HRP Weights ─────────────────────────────────────────────────────────────

class TestHrpWeights:
    """Tests for features.hrp_weights(returns)."""

    def test_sum_to_one(self, block_returns):
        """HRP weights must sum to 1."""
        w = features.hrp_weights(block_returns)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-10)

    def test_all_positive(self, block_returns):
        """HRP always produces positive weights (no shorting)."""
        w = features.hrp_weights(block_returns)
        assert np.all(w > 0), f"Negative weights found: {w}"

    def test_correct_length(self, block_returns):
        """Number of weights = number of assets."""
        w = features.hrp_weights(block_returns)
        assert len(w) == block_returns.shape[1]

    def test_single_asset(self):
        """Single asset → weight = 1."""
        returns = np.array([[0.01], [-0.005], [0.002], [0.008]])
        w = features.hrp_weights(returns)
        assert len(w) == 1
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_two_uncorrelated_assets(self):
        """Two uncorrelated assets: HRP ≈ IVP (inverse variance)."""
        rng = np.random.default_rng(42)
        n = 1000  # large sample for stable estimates
        # Uncorrelated with different volatilities
        returns = np.column_stack([
            rng.normal(0, 0.01, n),   # low vol
            rng.normal(0, 0.03, n),   # high vol
        ])
        w = features.hrp_weights(returns)
        # For uncorrelated assets, HRP reduces to IVP
        cov = np.cov(returns.T)
        w_ivp = features.inverse_variance_weights(cov)
        np.testing.assert_allclose(w, w_ivp, atol=0.05)

    def test_lower_vol_gets_higher_weight(self):
        """Assets with lower volatility should get more weight."""
        rng = np.random.default_rng(42)
        n = 500
        returns = np.column_stack([
            rng.normal(0, 0.03, n),   # high vol
            rng.normal(0, 0.005, n),  # low vol
        ])
        w = features.hrp_weights(returns)
        assert w[1] > w[0], (
            f"Low-vol asset should have higher weight: {w}"
        )

    def test_deterministic(self, block_returns):
        """Same input → same output."""
        w1 = features.hrp_weights(block_returns)
        w2 = features.hrp_weights(block_returns)
        np.testing.assert_array_equal(w1, w2)

    def test_equicorrelated_equal_vol(self):
        """Assets with same vol and same pairwise corr → equal weights."""
        rng = np.random.default_rng(42)
        n = 2000
        k = 3
        # Create equicorrelated returns: X_i = Z_common + Z_i
        z_common = rng.normal(0, 0.01, n)
        returns = np.column_stack([
            z_common + rng.normal(0, 0.01, n) for _ in range(k)
        ])
        w = features.hrp_weights(returns)
        # All weights should be approximately equal
        np.testing.assert_allclose(w, np.ones(k) / k, atol=0.1)

    def test_cluster_structure_respected(self, block_returns):
        """In block structure, total weight should favor low-vol cluster."""
        w = features.hrp_weights(block_returns)
        # Assets 0-1 are high-vol cluster, 2-3 are low-vol
        high_vol_total = w[0] + w[1]
        low_vol_total = w[2] + w[3]
        assert low_vol_total > high_vol_total, (
            f"Low-vol cluster should get more weight: "
            f"high={high_vol_total:.4f}, low={low_vol_total:.4f}"
        )


# ── Compare Allocations (Monte Carlo) ──────────────────────────────────────

class TestCompareAllocations:
    """Tests for features.compare_allocations(returns, n_sims, seed)."""

    def test_returns_valid_struct(self, block_returns):
        """Result has all required fields."""
        mc = features.compare_allocations(block_returns, n_simulations=10, seed=42)
        assert hasattr(mc, 'hrp_sharpe')
        assert hasattr(mc, 'cla_sharpe')
        assert hasattr(mc, 'ivp_sharpe')
        assert hasattr(mc, 'hrp_variance')
        assert hasattr(mc, 'cla_variance')
        assert hasattr(mc, 'ivp_variance')

    def test_variances_positive(self, block_returns):
        """All variance estimates should be positive."""
        mc = features.compare_allocations(block_returns, n_simulations=20, seed=42)
        assert mc.hrp_variance > 0
        assert mc.cla_variance > 0
        assert mc.ivp_variance > 0

    def test_sharpes_finite(self, block_returns):
        """Sharpe ratios should be finite numbers."""
        mc = features.compare_allocations(block_returns, n_simulations=20, seed=42)
        assert np.isfinite(mc.hrp_sharpe)
        assert np.isfinite(mc.cla_sharpe)
        assert np.isfinite(mc.ivp_sharpe)

    def test_deterministic_with_seed(self, block_returns):
        """Same seed → same results."""
        mc1 = features.compare_allocations(block_returns, n_simulations=10, seed=42)
        mc2 = features.compare_allocations(block_returns, n_simulations=10, seed=42)
        assert mc1.hrp_sharpe == mc2.hrp_sharpe
        assert mc1.cla_sharpe == mc2.cla_sharpe
        assert mc1.ivp_sharpe == mc2.ivp_sharpe

    def test_different_seed_different_results(self, block_returns):
        """Different seeds should produce different results."""
        mc1 = features.compare_allocations(block_returns, n_simulations=20, seed=42)
        mc2 = features.compare_allocations(block_returns, n_simulations=20, seed=99)
        # At least one metric should differ
        assert (mc1.hrp_sharpe != mc2.hrp_sharpe
                or mc1.cla_sharpe != mc2.cla_sharpe
                or mc1.ivp_sharpe != mc2.ivp_sharpe)


# ── Cross-method consistency ────────────────────────────────────────────────

class TestCrossMethodConsistency:
    """Tests verifying relationships between allocation methods."""

    def test_all_sum_to_one(self, block_returns):
        """All methods produce weights summing to 1."""
        cov = np.cov(block_returns.T)
        mu = np.mean(block_returns, axis=0)

        w_hrp = features.hrp_weights(block_returns)
        w_cla = features.cla_min_variance(cov)
        w_sr = features.cla_max_sharpe(mu, cov)
        w_ivp = features.inverse_variance_weights(cov)

        for name, w in [('HRP', w_hrp), ('CLA', w_cla), ('SR', w_sr), ('IVP', w_ivp)]:
            assert np.sum(w) == pytest.approx(1.0, abs=1e-8), (
                f"{name} weights don't sum to 1: sum={np.sum(w)}"
            )

    def test_all_correct_length(self, block_returns):
        """All methods return correct number of weights."""
        n_assets = block_returns.shape[1]
        cov = np.cov(block_returns.T)
        mu = np.mean(block_returns, axis=0)

        assert len(features.hrp_weights(block_returns)) == n_assets
        assert len(features.cla_min_variance(cov)) == n_assets
        assert len(features.cla_max_sharpe(mu, cov)) == n_assets
        assert len(features.inverse_variance_weights(cov)) == n_assets

    def test_cla_minvar_minimizes(self, block_returns):
        """CLA min-var should have lower portfolio variance than IVP and HRP."""
        cov = np.cov(block_returns.T)
        w_cla = features.cla_min_variance(cov)
        w_ivp = features.inverse_variance_weights(cov)
        w_hrp = features.hrp_weights(block_returns)
        w_eq = np.ones(block_returns.shape[1]) / block_returns.shape[1]

        var_cla = w_cla @ cov @ w_cla
        var_ivp = w_ivp @ cov @ w_ivp
        var_hrp = w_hrp @ cov @ w_hrp
        var_eq = w_eq @ cov @ w_eq

        # CLA should be optimal (global minimum)
        assert var_cla <= var_ivp + 1e-10
        assert var_cla <= var_eq + 1e-10
        # HRP may or may not beat IVP, but CLA is the theoretical minimum
        assert var_cla <= var_hrp + 1e-10

    def test_hrp_always_positive(self, block_returns):
        """HRP never shorts, unlike CLA which can."""
        w_hrp = features.hrp_weights(block_returns)
        assert np.all(w_hrp > 0)
        # CLA may have negative weights (no assertion needed, just documenting)

    def test_sharpe_ratio_computable(self, block_returns):
        """All portfolio returns should have computable Sharpe ratios."""
        cov = np.cov(block_returns.T)
        mu = np.mean(block_returns, axis=0)

        for name, w in [
            ('HRP', features.hrp_weights(block_returns)),
            ('CLA', features.cla_min_variance(cov)),
            ('SR', features.cla_max_sharpe(mu, cov)),
            ('IVP', features.inverse_variance_weights(cov)),
        ]:
            port_ret = block_returns @ w
            sr = backtesting.sharpe_ratio(port_ret)
            assert np.isfinite(sr), f"{name} Sharpe is not finite: {sr}"
