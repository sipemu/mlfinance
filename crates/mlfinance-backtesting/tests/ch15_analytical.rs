//! Analytical correctness tests for Chapter 15: Strategy Risk.
//!
//! These tests verify hand-computed values, round-trip consistency between
//! forward/inverse functions, monotonicity properties, and edge cases.
//!
//! Mathematical reference (AFML Ch. 15):
//!   SR = \[p*R - (1-p)\] / sqrt(p*(1-p)) / (R+1) * sqrt(freq)
//!   break_even_precision = 1 / (1 + R)
//!   failure_prob = 1 - Phi(z),  z = (p_hat - p*) / sqrt(p*(1-p*)/n)

use mlfinance_backtesting::strategy_risk::failure_probability::strategy_failure_probability;
use mlfinance_backtesting::strategy_risk::implied_frequency::implied_frequency;
use mlfinance_backtesting::strategy_risk::implied_precision::implied_precision;
use mlfinance_backtesting::strategy_risk::sr_from_precision::sr_from_precision;

// ── sr_from_precision: known analytical values ─────────────────────────────

#[test]
fn sr_break_even_is_zero() {
    // At break-even precision p = 1/(1+R), SR should be 0.
    for r in [0.5, 1.0, 1.5, 2.0, 3.0] {
        let p_be = 1.0 / (1.0 + r);
        let sr = sr_from_precision(p_be, 252.0, r);
        assert!(
            sr.abs() < 1e-10,
            "Break-even p={:.4}, R={} should give SR=0, got {:.10e}",
            p_be,
            r,
            sr
        );
    }
}

#[test]
fn sr_known_value_p55_freq252_r1() {
    // E = 0.55*1 - 0.45 = 0.10
    // Var = 0.55*0.45*(2)^2 = 0.99
    // SR_bet = 0.10/sqrt(0.99) = 0.100504
    // SR = 0.100504 * sqrt(252) = 1.5954
    let sr = sr_from_precision(0.55, 252.0, 1.0);
    assert!((sr - 1.5954).abs() < 0.001, "Expected 1.5954, got {}", sr);
}

#[test]
fn sr_known_value_p60_freq252_r1() {
    // E = 0.20, Var = 0.96, SR_bet = 0.204124, SR = 3.2404
    let sr = sr_from_precision(0.60, 252.0, 1.0);
    assert!((sr - 3.2404).abs() < 0.001, "Expected 3.2404, got {}", sr);
}

#[test]
fn sr_known_value_p40_negative() {
    // Mirror of p=0.60: should give -3.2404
    let sr = sr_from_precision(0.40, 252.0, 1.0);
    assert!(
        (sr - (-3.2404)).abs() < 0.001,
        "Expected -3.2404, got {}",
        sr
    );
}

#[test]
fn sr_antisymmetry_around_breakeven() {
    // For R=1: SR(0.5+d) = -SR(0.5-d)
    for d in [0.01, 0.05, 0.10, 0.20] {
        let sr_above = sr_from_precision(0.5 + d, 252.0, 1.0);
        let sr_below = sr_from_precision(0.5 - d, 252.0, 1.0);
        assert!(
            (sr_above + sr_below).abs() < 1e-10,
            "d={}: SR(0.5+d)={}, SR(0.5-d)={}, sum={}",
            d,
            sr_above,
            sr_below,
            sr_above + sr_below
        );
    }
}

#[test]
fn sr_sqrt_freq_scaling() {
    // SR(4*freq) = 2*SR(freq)
    let sr1 = sr_from_precision(0.55, 63.0, 1.0);
    let sr4 = sr_from_precision(0.55, 252.0, 1.0);
    assert!(
        (sr4 - 2.0 * sr1).abs() < 1e-10,
        "SR should scale as sqrt(freq): sr4={}, 2*sr1={}",
        sr4,
        2.0 * sr1
    );
}

#[test]
fn sr_monotone_increasing_in_precision() {
    let mut prev_sr = f64::NEG_INFINITY;
    for i in 0..50 {
        let p = 0.30 + (i as f64) * 0.01;
        let sr = sr_from_precision(p, 252.0, 1.0);
        assert!(
            sr > prev_sr,
            "Non-monotone at p={}: {} <= {}",
            p,
            sr,
            prev_sr
        );
        prev_sr = sr;
    }
}

#[test]
fn sr_matches_manual_reference() {
    // Cross-check against a manually computed reference implementation
    let cases: [(f64, f64, f64); 4] = [
        (0.55, 252.0, 1.0),
        (0.60, 52.0, 1.5),
        (0.45, 252.0, 2.0),
        (0.70, 12.0, 0.8),
    ];
    for (p, freq, r) in cases {
        let e = p * r - (1.0 - p);
        let var = p * (1.0 - p) * (r + 1.0_f64).powi(2);
        let expected = (e / var.sqrt()) * freq.sqrt();
        let actual = sr_from_precision(p, freq, r);
        assert!(
            (actual - expected).abs() < 1e-10,
            "p={}, freq={}, R={}: expected={}, got={}",
            p,
            freq,
            r,
            expected,
            actual
        );
    }
}

// ── implied_precision: round-trip and known values ─────────────────────────

#[test]
fn implied_precision_round_trip() {
    let cases = [
        (0.55, 252.0, 1.0),
        (0.60, 252.0, 1.5),
        (0.52, 52.0, 0.8),
        (0.70, 12.0, 2.0),
    ];
    for (p, freq, r) in cases {
        let sr = sr_from_precision(p, freq, r);
        let recovered = implied_precision(sr, freq, r);
        assert!(
            (recovered - p).abs() < 1e-4,
            "Round-trip failed: p={}, freq={}, R={}, SR={}, recovered={}",
            p,
            freq,
            r,
            sr,
            recovered
        );
    }
}

#[test]
fn implied_precision_zero_sr_gives_break_even() {
    for r in [0.5, 1.0, 1.5, 2.0] {
        let p = implied_precision(0.0, 252.0, r);
        let expected = 1.0 / (1.0 + r);
        assert!(
            (p - expected).abs() < 1e-4,
            "SR=0, R={}: expected p={:.4}, got {:.4}",
            r,
            expected,
            p
        );
    }
}

#[test]
fn implied_precision_known_value_sr1() {
    // From notebook: implied_precision(1.0, 252, 1.0) ≈ 0.5314
    let p = implied_precision(1.0, 252.0, 1.0);
    assert!((p - 0.5314).abs() < 0.001, "Expected ~0.5314, got {}", p);
}

#[test]
fn implied_precision_monotone_in_target() {
    let targets = [0.5, 1.0, 1.5, 2.0, 2.5];
    let mut prev_p = 0.0;
    for &tsr in &targets {
        let p = implied_precision(tsr, 252.0, 1.0);
        assert!(
            p > prev_p,
            "Non-monotone at SR={}: {} <= {}",
            tsr,
            p,
            prev_p
        );
        prev_p = p;
    }
}

// ── implied_frequency: round-trip and known values ─────────────────────────

#[test]
fn implied_frequency_round_trip() {
    let cases = [(0.55, 252.0, 1.0), (0.60, 52.0, 1.5), (0.52, 1000.0, 0.8)];
    for (p, freq, r) in cases {
        let sr = sr_from_precision(p, freq, r);
        let recovered = implied_frequency(sr, p, r);
        let rel_err = (recovered - freq).abs() / freq;
        assert!(
            rel_err < 0.01,
            "Round-trip failed: p={}, freq={}, R={}, SR={}, recovered={}",
            p,
            freq,
            r,
            sr,
            recovered
        );
    }
}

#[test]
fn implied_frequency_known_value_p55_sr1() {
    // SR_bet = 0.10/sqrt(0.99) ≈ 0.100504
    // freq = (1.0/0.100504)^2 ≈ 99.0
    let f = implied_frequency(1.0, 0.55, 1.0);
    assert!((f - 99.0).abs() < 1.0, "Expected ~99, got {}", f);
}

#[test]
fn implied_frequency_known_value_p51_sr1() {
    // SR_bet = 0.02/sqrt(0.9996) ≈ 0.020004
    // freq = (1.0/0.020004)^2 ≈ 2499
    let f = implied_frequency(1.0, 0.51, 1.0);
    assert!((f - 2499.0).abs() < 5.0, "Expected ~2499, got {}", f);
}

#[test]
fn implied_frequency_quadratic_scaling() {
    // freq ∝ SR^2: doubling SR requires 4x frequency
    let f1 = implied_frequency(1.0, 0.55, 1.0);
    let f2 = implied_frequency(2.0, 0.55, 1.0);
    let ratio = f2 / f1;
    assert!(
        (ratio - 4.0).abs() < 0.01,
        "Should scale as SR^2: f2/f1={}, expected 4.0",
        ratio
    );
}

#[test]
fn implied_frequency_monotone_decreasing_in_precision() {
    let precs = [0.52, 0.55, 0.60, 0.65];
    let freqs: Vec<f64> = precs
        .iter()
        .map(|&p| implied_frequency(1.0, p, 1.0))
        .collect();
    for i in 1..freqs.len() {
        assert!(
            freqs[i] < freqs[i - 1],
            "Non-monotone at p={}: {} >= {}",
            precs[i],
            freqs[i],
            freqs[i - 1]
        );
    }
}

// ── strategy_failure_probability: known analytical values ──────────────────

#[test]
fn failure_prob_at_break_even_gives_half() {
    let fp = strategy_failure_probability(0.5, 1000, 0.5);
    assert!(
        (fp - 0.5).abs() < 0.01,
        "At break-even should be ~0.5, got {}",
        fp
    );
}

#[test]
fn failure_prob_known_value_n100() {
    // SE = sqrt(0.25/100) = 0.05,  z = 0.05/0.05 = 1.0
    // P[fail] = 1 - Phi(1.0) = 0.15866
    let fp = strategy_failure_probability(0.55, 100, 0.5);
    assert!((fp - 0.15866).abs() < 0.001, "Expected 0.15866, got {}", fp);
}

#[test]
fn failure_prob_known_value_n500() {
    // SE = sqrt(0.25/500) = 0.02236,  z = 0.05/0.02236 = 2.236
    // P[fail] = 1 - Phi(2.236) ≈ 0.01267
    let fp = strategy_failure_probability(0.55, 500, 0.5);
    assert!((fp - 0.01267).abs() < 0.001, "Expected 0.01267, got {}", fp);
}

#[test]
fn failure_prob_known_value_n50() {
    // SE = sqrt(0.25/50) = 0.07071,  z = 0.05/0.07071 = 0.7071
    // P[fail] = 1 - Phi(0.7071) ≈ 0.23975
    let fp = strategy_failure_probability(0.55, 50, 0.5);
    assert!((fp - 0.23975).abs() < 0.001, "Expected 0.23975, got {}", fp);
}

#[test]
fn failure_prob_monotone_decreasing_in_n() {
    let ns = [50_usize, 100, 200, 500, 1000, 2000];
    let fps: Vec<f64> = ns
        .iter()
        .map(|&n| strategy_failure_probability(0.55, n, 0.5))
        .collect();
    for i in 1..fps.len() {
        assert!(
            fps[i] < fps[i - 1],
            "Non-monotone at n={}: {} >= {}",
            ns[i],
            fps[i],
            fps[i - 1]
        );
    }
}

#[test]
fn failure_prob_monotone_decreasing_in_precision() {
    let precs = [0.51, 0.52, 0.55, 0.60, 0.70];
    let fps: Vec<f64> = precs
        .iter()
        .map(|&p| strategy_failure_probability(p, 500, 0.5))
        .collect();
    for i in 1..fps.len() {
        assert!(
            fps[i] < fps[i - 1],
            "Non-monotone at p={}: {} >= {}",
            precs[i],
            fps[i],
            fps[i - 1]
        );
    }
}

#[test]
fn failure_prob_below_break_even_high() {
    let fp = strategy_failure_probability(0.45, 100, 0.50);
    assert!(fp > 0.5, "Below break-even should be >0.5, got {}", fp);
}

#[test]
fn failure_prob_far_above_near_zero() {
    let fp = strategy_failure_probability(0.70, 1000, 0.50);
    assert!(fp < 1e-6, "Far above break-even should be ~0, got {}", fp);
}

// ── Cross-function consistency ─────────────────────────────────────────────

#[test]
fn sr_and_implied_precision_inverse_many() {
    let cases = [
        (0.35, 252.0, 1.0),
        (0.55, 252.0, 1.0),
        (0.60, 52.0, 1.5),
        (0.52, 500.0, 0.8),
        (0.70, 12.0, 2.5),
        (0.45, 100.0, 3.0),
    ];
    for (p, freq, r) in cases {
        let sr = sr_from_precision(p, freq, r);
        let p_back = implied_precision(sr, freq, r);
        assert!(
            (p_back - p).abs() < 0.005,
            "Round-trip: p={}, SR={}, recovered={}",
            p,
            sr,
            p_back
        );
    }
}

#[test]
fn sr_and_implied_frequency_inverse_many() {
    let cases = [
        (0.55, 252.0, 1.0),
        (0.60, 52.0, 1.5),
        (0.52, 1000.0, 0.8),
        (0.70, 100.0, 2.0),
    ];
    for (p, freq, r) in cases {
        let sr = sr_from_precision(p, freq, r);
        if sr.abs() > 0.01 {
            let f_back = implied_frequency(sr, p, r);
            let rel_err = (f_back - freq).abs() / freq;
            assert!(
                rel_err < 0.02,
                "Round-trip: p={}, freq={}, SR={}, recovered={}",
                p,
                freq,
                sr,
                f_back
            );
        }
    }
}

#[test]
fn failure_prob_consistent_with_sr_at_breakeven() {
    // Strategy with SR=0 at break-even should have ~50% failure
    let r = 1.5;
    let p_be = 1.0 / (1.0 + r);
    let sr = sr_from_precision(p_be, 252.0, r);
    assert!(sr.abs() < 1e-8);
    let fp = strategy_failure_probability(p_be, 1000, p_be);
    assert!((fp - 0.5).abs() < 0.01);
}
