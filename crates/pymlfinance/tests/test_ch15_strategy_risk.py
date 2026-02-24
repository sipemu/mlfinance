"""Comprehensive tests for Ch 15: Strategy Risk functions.

Tests verify numerical correctness against hand-derived analytical
values, round-trip consistency between forward/inverse functions,
monotonicity properties, and edge cases.

Mathematical reference (AFML Ch. 15):
    SR = [p*R - (1-p)] / sqrt(p*(1-p)) / (R+1) * sqrt(freq)
    break_even_precision = 1 / (1 + R)
    failure_prob = 1 - Phi(z), z = (p_hat - p*) / sqrt(p*(1-p*)/n)
"""

import math

import numpy as np
import pytest
from pymlfinance import backtesting


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sr_from_precision_reference(p, freq, r):
    """Pure-Python reference implementation of sr_from_precision."""
    if p <= 0 or p >= 1 or freq <= 0 or r <= 0:
        return 0.0
    e = p * r - (1 - p)
    var = p * (1 - p) * (r + 1) ** 2
    if var <= 0:
        return 0.0
    return (e / math.sqrt(var)) * math.sqrt(freq)


def _failure_prob_reference(p_hat, n, p_star):
    """Pure-Python reference: normal approximation to binomial."""
    if n == 0 or p_star <= 0:
        return 0.0
    if p_star >= 1:
        return 1.0
    se = math.sqrt(p_star * (1 - p_star) / n)
    if se < 1e-15:
        return 0.0 if p_hat >= p_star else 1.0
    z = (p_hat - p_star) / se
    # 1 - Phi(z) using erfc
    return 0.5 * math.erfc(z / math.sqrt(2))


# ── sr_from_precision ───────────────────────────────────────────────────────

class TestSrFromPrecision:
    """Tests for backtesting.sr_from_precision(precision, freq, win_loss)."""

    def test_break_even_is_zero(self):
        """At break-even precision p=1/(1+R), expected SR should be 0."""
        for r in [0.5, 1.0, 1.5, 2.0, 3.0]:
            p_be = 1.0 / (1.0 + r)
            sr = backtesting.sr_from_precision(p_be, 252.0, r)
            assert sr == pytest.approx(0.0, abs=1e-10), (
                f"Break-even p={p_be:.4f}, R={r} should give SR=0, got {sr}"
            )

    def test_symmetric_zero_at_half(self):
        """For R=1 (symmetric bets), p=0.5 is break-even."""
        sr = backtesting.sr_from_precision(0.5, 252.0, 1.0)
        assert sr == pytest.approx(0.0, abs=1e-12)

    def test_known_value_p55_freq252_r1(self):
        """Hand-computed: p=0.55, freq=252, R=1.0.

        E = 0.55*1 - 0.45 = 0.10
        Var = 0.55*0.45*(1+1)^2 = 0.99
        SR_bet = 0.10/sqrt(0.99) = 0.100504
        SR = 0.100504 * sqrt(252) = 1.5954
        """
        sr = backtesting.sr_from_precision(0.55, 252.0, 1.0)
        assert sr == pytest.approx(1.5954, abs=0.001)

    def test_known_value_p60_freq252_r1(self):
        """Hand-computed: p=0.60, freq=252, R=1.0.

        E = 0.60 - 0.40 = 0.20
        Var = 0.60*0.40*4 = 0.96
        SR_bet = 0.20/sqrt(0.96) = 0.204124
        SR = 0.204124 * sqrt(252) = 3.2404
        """
        sr = backtesting.sr_from_precision(0.60, 252.0, 1.0)
        assert sr == pytest.approx(3.2404, abs=0.001)

    def test_known_value_p40_negative(self):
        """p=0.40, R=1.0 is below break-even → negative SR."""
        sr = backtesting.sr_from_precision(0.40, 252.0, 1.0)
        assert sr == pytest.approx(-3.2404, abs=0.001)

    def test_antisymmetry_around_breakeven(self):
        """SR(0.5+d) = -SR(0.5-d) for R=1 (symmetric)."""
        for d in [0.01, 0.05, 0.10, 0.20]:
            sr_above = backtesting.sr_from_precision(0.5 + d, 252.0, 1.0)
            sr_below = backtesting.sr_from_precision(0.5 - d, 252.0, 1.0)
            assert sr_above == pytest.approx(-sr_below, abs=1e-10)

    def test_monotone_increasing_in_precision(self):
        """SR is strictly increasing in precision for fixed freq, R."""
        precisions = np.linspace(0.30, 0.80, 50)
        srs = [backtesting.sr_from_precision(p, 252.0, 1.0) for p in precisions]
        for i in range(1, len(srs)):
            assert srs[i] > srs[i - 1], (
                f"Non-monotone: SR({precisions[i]:.3f})={srs[i]:.6f} "
                f"<= SR({precisions[i-1]:.3f})={srs[i-1]:.6f}"
            )

    def test_monotone_increasing_in_frequency(self):
        """SR increases with sqrt(freq) for p > break-even."""
        freqs = [12, 52, 252, 1000]
        srs = [backtesting.sr_from_precision(0.55, f, 1.0) for f in freqs]
        for i in range(1, len(srs)):
            assert srs[i] > srs[i - 1]

    def test_sqrt_freq_scaling(self):
        """SR scales as sqrt(freq): SR(4*freq) = 2*SR(freq)."""
        sr1 = backtesting.sr_from_precision(0.55, 63.0, 1.0)
        sr4 = backtesting.sr_from_precision(0.55, 252.0, 1.0)
        assert sr4 == pytest.approx(2.0 * sr1, abs=1e-10)

    def test_asymmetric_win_loss(self):
        """Higher R lowers break-even and raises SR for same precision."""
        sr_r1 = backtesting.sr_from_precision(0.55, 252.0, 1.0)
        sr_r2 = backtesting.sr_from_precision(0.55, 252.0, 2.0)
        assert sr_r2 > sr_r1

    def test_matches_reference_implementation(self):
        """Cross-check against pure-Python reference for many inputs."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            p = rng.uniform(0.1, 0.9)
            freq = rng.uniform(12, 1000)
            r = rng.uniform(0.3, 5.0)
            expected = _sr_from_precision_reference(p, freq, r)
            actual = backtesting.sr_from_precision(p, freq, r)
            assert actual == pytest.approx(expected, abs=1e-10), (
                f"Mismatch: p={p}, freq={freq}, R={r}: "
                f"expected={expected}, got={actual}"
            )

    def test_edge_precision_zero(self):
        """p=0 should return 0 (edge case guard)."""
        assert backtesting.sr_from_precision(0.0, 252.0, 1.0) == 0.0

    def test_edge_precision_one(self):
        """p=1 should return 0 (edge case guard)."""
        assert backtesting.sr_from_precision(1.0, 252.0, 1.0) == 0.0

    def test_edge_freq_zero(self):
        """freq=0 should return 0."""
        assert backtesting.sr_from_precision(0.55, 0.0, 1.0) == 0.0

    def test_edge_wl_zero(self):
        """R=0 should return 0."""
        assert backtesting.sr_from_precision(0.55, 252.0, 0.0) == 0.0


# ── implied_precision ───────────────────────────────────────────────────────

class TestImpliedPrecision:
    """Tests for backtesting.implied_precision(target_sr, freq, win_loss)."""

    def test_round_trip_sr_to_precision(self):
        """implied_precision inverts sr_from_precision."""
        test_cases = [
            (0.55, 252.0, 1.0),
            (0.60, 252.0, 1.5),
            (0.52, 52.0, 0.8),
            (0.70, 12.0, 2.0),
        ]
        for p, freq, r in test_cases:
            sr = backtesting.sr_from_precision(p, freq, r)
            p_recovered = backtesting.implied_precision(sr, freq, r)
            assert p_recovered == pytest.approx(p, abs=1e-4), (
                f"Round-trip failed: p={p}, freq={freq}, R={r}, "
                f"SR={sr}, recovered_p={p_recovered}"
            )

    def test_zero_sr_gives_break_even(self):
        """Target SR=0 should give break-even precision 1/(1+R)."""
        for r in [0.5, 1.0, 1.5, 2.0]:
            p = backtesting.implied_precision(0.0, 252.0, r)
            expected = 1.0 / (1.0 + r)
            assert p == pytest.approx(expected, abs=1e-4), (
                f"SR=0, R={r}: expected p={expected:.4f}, got {p:.4f}"
            )

    def test_known_value_sr1_freq252_r1(self):
        """From notebook output: implied_precision(1.0, 252, 1.0) ≈ 0.5314."""
        p = backtesting.implied_precision(1.0, 252.0, 1.0)
        assert p == pytest.approx(0.5314, abs=0.001)

    def test_monotone_increasing_in_target(self):
        """Higher target SR requires higher precision."""
        targets = [0.5, 1.0, 1.5, 2.0, 2.5]
        precs = [backtesting.implied_precision(t, 252.0, 1.0) for t in targets]
        for i in range(1, len(precs)):
            assert precs[i] > precs[i - 1], (
                f"Non-monotone: p(SR={targets[i]})={precs[i]:.4f} "
                f"<= p(SR={targets[i-1]})={precs[i-1]:.4f}"
            )

    def test_higher_wl_needs_lower_precision(self):
        """Higher R means you need less precision for the same SR."""
        p_r1 = backtesting.implied_precision(1.0, 252.0, 1.0)
        p_r2 = backtesting.implied_precision(1.0, 252.0, 2.0)
        assert p_r2 < p_r1

    def test_higher_freq_needs_lower_precision(self):
        """Higher frequency means you need less precision for the same SR."""
        p_f252 = backtesting.implied_precision(1.0, 252.0, 1.0)
        p_f1000 = backtesting.implied_precision(1.0, 1000.0, 1.0)
        assert p_f1000 < p_f252


# ── implied_frequency ───────────────────────────────────────────────────────

class TestImpliedFrequency:
    """Tests for backtesting.implied_frequency(target_sr, precision, win_loss)."""

    def test_round_trip_sr_to_frequency(self):
        """implied_frequency inverts sr_from_precision."""
        test_cases = [
            (0.55, 252.0, 1.0),
            (0.60, 52.0, 1.5),
            (0.52, 1000.0, 0.8),
        ]
        for p, freq, r in test_cases:
            sr = backtesting.sr_from_precision(p, freq, r)
            f_recovered = backtesting.implied_frequency(sr, p, r)
            assert f_recovered == pytest.approx(freq, rel=0.01), (
                f"Round-trip failed: p={p}, freq={freq}, R={r}, "
                f"SR={sr}, recovered_freq={f_recovered}"
            )

    def test_known_value_p55_sr1(self):
        """From notebook: implied_frequency(1.0, 0.55, 1.0) ≈ 99.

        SR_bet = 0.10 / sqrt(0.99) = 0.100504
        freq = (1.0 / 0.100504)^2 = 99.0
        """
        f = backtesting.implied_frequency(1.0, 0.55, 1.0)
        assert f == pytest.approx(99.0, abs=1.0)

    def test_known_value_p51_sr1(self):
        """From notebook: implied_frequency(1.0, 0.51, 1.0) ≈ 2499.

        SR_bet = 0.02 / sqrt(0.9996) = 0.020004
        freq = (1.0/0.020004)^2 ≈ 2499
        """
        f = backtesting.implied_frequency(1.0, 0.51, 1.0)
        assert f == pytest.approx(2499.0, abs=5.0)

    def test_quadratic_scaling(self):
        """freq scales as SR^2: doubling SR requires 4x the frequency."""
        f1 = backtesting.implied_frequency(1.0, 0.55, 1.0)
        f2 = backtesting.implied_frequency(2.0, 0.55, 1.0)
        assert f2 == pytest.approx(4.0 * f1, rel=0.01)

    def test_break_even_precision_gives_infinity_or_zero(self):
        """At break-even precision, SR_bet=0, so freq should be inf or 0."""
        f = backtesting.implied_frequency(1.0, 0.5, 1.0)
        # Implementation returns 0.0 for non-finite results
        assert f == 0.0 or f == float('inf')

    def test_zero_target_gives_zero(self):
        """SR=0 requires no trading."""
        f = backtesting.implied_frequency(0.0, 0.55, 1.0)
        assert f == pytest.approx(0.0, abs=1e-10)

    def test_monotone_decreasing_in_precision(self):
        """Higher precision requires fewer trades for the same SR."""
        precs = [0.52, 0.55, 0.60, 0.65]
        freqs = [backtesting.implied_frequency(1.0, p, 1.0) for p in precs]
        for i in range(1, len(freqs)):
            assert freqs[i] < freqs[i - 1], (
                f"Non-monotone: freq(p={precs[i]})={freqs[i]:.1f} "
                f">= freq(p={precs[i-1]})={freqs[i-1]:.1f}"
            )


# ── strategy_failure_probability ────────────────────────────────────────────

class TestStrategyFailureProbability:
    """Tests for backtesting.strategy_failure_probability(p_hat, n, p_star)."""

    def test_at_break_even_gives_half(self):
        """If observed p = break-even p*, failure prob should be ~0.5."""
        fp = backtesting.strategy_failure_probability(0.5, 1000, 0.5)
        assert fp == pytest.approx(0.5, abs=0.01)

    def test_known_value_n100(self):
        """Hand-computed: p_hat=0.55, n=100, p*=0.50.

        SE = sqrt(0.5*0.5/100) = 0.05
        z = (0.55-0.50)/0.05 = 1.0
        P[fail] = 1 - Phi(1.0) = 0.15866
        """
        fp = backtesting.strategy_failure_probability(0.55, 100, 0.5)
        assert fp == pytest.approx(0.15866, abs=0.001)

    def test_known_value_n500(self):
        """Hand-computed: p_hat=0.55, n=500, p*=0.50.

        SE = sqrt(0.25/500) = 0.02236
        z = 0.05/0.02236 = 2.236
        P[fail] = 1 - Phi(2.236) = 0.01267
        """
        fp = backtesting.strategy_failure_probability(0.55, 500, 0.5)
        assert fp == pytest.approx(0.01267, abs=0.001)

    def test_known_value_n50(self):
        """Hand-computed: p_hat=0.55, n=50, p*=0.50.

        SE = sqrt(0.25/50) = 0.07071
        z = 0.05/0.07071 = 0.7071
        P[fail] = 1 - Phi(0.7071) = 0.23975
        """
        fp = backtesting.strategy_failure_probability(0.55, 50, 0.5)
        assert fp == pytest.approx(0.23975, abs=0.001)

    def test_monotone_decreasing_in_n(self):
        """More observations → lower failure prob (for p_hat > p*)."""
        ns = [50, 100, 200, 500, 1000, 2000]
        fps = [backtesting.strategy_failure_probability(0.55, n, 0.5) for n in ns]
        for i in range(1, len(fps)):
            assert fps[i] < fps[i - 1], (
                f"Non-monotone: fp(n={ns[i]})={fps[i]:.6f} "
                f">= fp(n={ns[i-1]})={fps[i-1]:.6f}"
            )

    def test_monotone_decreasing_in_precision(self):
        """Higher observed precision → lower failure prob."""
        precs = [0.51, 0.52, 0.55, 0.60, 0.70]
        fps = [backtesting.strategy_failure_probability(p, 500, 0.5) for p in precs]
        for i in range(1, len(fps)):
            assert fps[i] < fps[i - 1]

    def test_monotone_increasing_in_break_even(self):
        """Higher break-even threshold → higher failure prob."""
        bes = [0.40, 0.45, 0.50, 0.52]
        fps = [backtesting.strategy_failure_probability(0.55, 500, be) for be in bes]
        for i in range(1, len(fps)):
            assert fps[i] > fps[i - 1]

    def test_below_break_even_high_failure(self):
        """If observed p < break-even, failure prob > 0.5."""
        fp = backtesting.strategy_failure_probability(0.45, 100, 0.50)
        assert fp > 0.5

    def test_far_above_break_even_near_zero(self):
        """If observed p >> break-even with large n, failure prob ≈ 0."""
        fp = backtesting.strategy_failure_probability(0.70, 1000, 0.50)
        assert fp < 1e-6

    def test_range_always_0_to_1(self):
        """Failure probability is always in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.uniform(0.3, 0.8)
            n = rng.integers(10, 2000)
            be = rng.uniform(0.3, 0.7)
            fp = backtesting.strategy_failure_probability(p, int(n), be)
            assert 0.0 <= fp <= 1.0, f"Out of range: p={p}, n={n}, be={be}, fp={fp}"

    def test_matches_reference(self):
        """Cross-check against pure-Python normal approx."""
        cases = [
            (0.55, 100, 0.50),
            (0.52, 500, 0.48),
            (0.60, 200, 0.55),
            (0.54, 300, 0.4545),
        ]
        for p_hat, n, p_star in cases:
            expected = _failure_prob_reference(p_hat, n, p_star)
            actual = backtesting.strategy_failure_probability(p_hat, n, p_star)
            assert actual == pytest.approx(expected, abs=0.002), (
                f"Mismatch: p_hat={p_hat}, n={n}, p*={p_star}: "
                f"expected={expected:.6f}, got={actual:.6f}"
            )

    def test_edge_n_zero(self):
        """n=0 observations → no information → return 0 (by convention)."""
        fp = backtesting.strategy_failure_probability(0.55, 0, 0.5)
        assert fp == 0.0

    def test_edge_break_even_zero(self):
        """If break-even is 0, strategy always succeeds."""
        fp = backtesting.strategy_failure_probability(0.55, 100, 0.0)
        assert fp == 0.0

    def test_edge_break_even_one(self):
        """If break-even is 1.0, strategy always fails."""
        fp = backtesting.strategy_failure_probability(0.55, 100, 1.0)
        assert fp == 1.0


# ── Cross-function consistency ──────────────────────────────────────────────

class TestCrossFunctionConsistency:
    """Tests verifying relationships between the four Ch 15 functions."""

    def test_sr_and_implied_precision_inverse(self):
        """Full round-trip for many random inputs."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            p = rng.uniform(0.3, 0.8)
            freq = rng.uniform(50, 500)
            r = rng.uniform(0.5, 3.0)
            sr = backtesting.sr_from_precision(p, freq, r)
            p_back = backtesting.implied_precision(sr, freq, r)
            assert p_back == pytest.approx(p, abs=0.005)

    def test_sr_and_implied_frequency_inverse(self):
        """Full round-trip for many random inputs."""
        rng = np.random.default_rng(77)
        for _ in range(50):
            p = rng.uniform(0.52, 0.8)  # above break-even
            freq = rng.uniform(20, 500)
            r = rng.uniform(0.5, 3.0)
            sr = backtesting.sr_from_precision(p, freq, r)
            if sr > 0.01:  # meaningful positive SR
                f_back = backtesting.implied_frequency(sr, p, r)
                assert f_back == pytest.approx(freq, rel=0.02)

    def test_failure_prob_consistent_with_sr(self):
        """Strategy with SR=0 at break-even should have ~50% failure."""
        r = 1.5
        p_be = 1.0 / (1.0 + r)
        sr = backtesting.sr_from_precision(p_be, 252.0, r)
        assert abs(sr) < 1e-8
        fp = backtesting.strategy_failure_probability(p_be, 1000, p_be)
        assert fp == pytest.approx(0.5, abs=0.01)
