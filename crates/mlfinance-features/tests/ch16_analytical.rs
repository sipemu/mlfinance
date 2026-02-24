//! Analytical correctness tests for Chapter 16: Portfolio Allocation.
//!
//! Tests verify HRP, CLA, IVP, and compare_allocations against analytical
//! solutions, invariant properties, and cross-method consistency.
//!
//! Mathematical reference (AFML Ch. 16):
//!   IVP:      w_i = (1/σ²_i) / Σ(1/σ²_j)
//!   CLA_mv:   w = Σ⁻¹ 1 / (1' Σ⁻¹ 1)
//!   CLA_sr:   w = Σ⁻¹ μ / (1' Σ⁻¹ μ)
//!   HRP:      hierarchical clustering + recursive bisection

use mlfinance_features::allocation::cla::{cla_max_sharpe, cla_min_variance};
use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::allocation::ivp::inverse_variance_weights;
use mlfinance_features::allocation::monte_carlo::compare_allocations;
use ndarray::{array, Array1, Array2};

// ── IVP: Inverse Variance Portfolio ────────────────────────────────────────

#[test]
fn ivp_analytical_3x3_diagonal() {
    // diag = [0.04, 0.01, 0.09]
    // 1/diag = [25, 100, 11.111]
    // sum = 136.111
    // w = [0.18367, 0.73469, 0.08163]
    let cov = array![[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.09]];
    let w = inverse_variance_weights(&cov);
    let inv_d: Vec<f64> = [0.04, 0.01, 0.09].iter().map(|&d| 1.0 / d).collect();
    let total: f64 = inv_d.iter().sum();
    for (i, (actual, expected)) in w.iter().zip(inv_d.iter().map(|v| v / total)).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "IVP[{}]: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn ivp_lower_var_higher_weight() {
    let cov = array![[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.09]];
    let w = inverse_variance_weights(&cov);
    // Asset 1 (var=0.01) > Asset 0 (var=0.04) > Asset 2 (var=0.09)
    assert!(w[1] > w[0] && w[0] > w[2]);
}

#[test]
fn ivp_equal_variance_equal_weights() {
    let cov = array![[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]];
    let w = inverse_variance_weights(&cov);
    for &wi in w.iter() {
        assert!((wi - 1.0 / 3.0).abs() < 1e-10);
    }
}

#[test]
fn ivp_ignores_off_diagonal() {
    let cov_diag = array![[0.04, 0.0], [0.0, 0.01]];
    let cov_corr = array![[0.04, 0.015], [0.015, 0.01]];
    let w1 = inverse_variance_weights(&cov_diag);
    let w2 = inverse_variance_weights(&cov_corr);
    for (a, b) in w1.iter().zip(w2.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn ivp_two_assets_analytical() {
    // 1/0.04=25, 1/0.16=6.25, sum=31.25
    // w = [0.8, 0.2]
    let cov = array![[0.04, 0.0], [0.0, 0.16]];
    let w = inverse_variance_weights(&cov);
    assert!((w[0] - 25.0 / 31.25).abs() < 1e-10);
    assert!((w[1] - 6.25 / 31.25).abs() < 1e-10);
}

// ── CLA Min Variance ───────────────────────────────────────────────────────

#[test]
fn cla_minvar_diagonal_equals_ivp() {
    // For diagonal cov, CLA min-var = IVP
    let cov = array![[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.09]];
    let w_cla = cla_min_variance(&cov).unwrap();
    let w_ivp = inverse_variance_weights(&cov);
    for (a, b) in w_cla.iter().zip(w_ivp.iter()) {
        assert!(
            (a - b).abs() < 1e-8,
            "CLA={}, IVP={} should match for diagonal cov",
            a,
            b
        );
    }
}

#[test]
fn cla_minvar_analytical_2x2() {
    // cov = [[0.04, 0.02], [0.02, 0.09]]
    // Verify against numpy: cov_inv @ ones / (ones @ cov_inv @ ones)
    let cov = array![[0.04, 0.02], [0.02, 0.09]];
    let w = cla_min_variance(&cov).unwrap();
    assert!((w.sum() - 1.0).abs() < 1e-10);
    // Check via manual computation:
    // det = 0.04*0.09 - 0.02*0.02 = 0.0036 - 0.0004 = 0.0032
    // inv = [[0.09, -0.02], [-0.02, 0.04]] / 0.0032
    //     = [[28.125, -6.25], [-6.25, 12.5]]
    // inv @ [1,1] = [21.875, 6.25]
    // sum = 28.125
    // w = [21.875/28.125, 6.25/28.125] = [0.77778, 0.22222]
    assert!((w[0] - 21.875 / 28.125).abs() < 1e-4);
    assert!((w[1] - 6.25 / 28.125).abs() < 1e-4);
}

#[test]
fn cla_minvar_minimizes_variance() {
    let cov = array![
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.003],
        [0.005, 0.003, 0.01]
    ];
    let w_cla = cla_min_variance(&cov).unwrap();
    let w_eq = array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let w_ivp = inverse_variance_weights(&cov);

    let var_cla = w_cla.dot(&cov.dot(&w_cla));
    let var_eq = w_eq.dot(&cov.dot(&w_eq));
    let var_ivp = w_ivp.dot(&cov.dot(&w_ivp));

    assert!(
        var_cla <= var_eq + 1e-12,
        "CLA var={} should be <= equal-weight var={}",
        var_cla,
        var_eq
    );
    assert!(
        var_cla <= var_ivp + 1e-12,
        "CLA var={} should be <= IVP var={}",
        var_cla,
        var_ivp
    );
}

#[test]
fn cla_minvar_single_asset() {
    let cov = array![[0.04]];
    let w = cla_min_variance(&cov).unwrap();
    assert!((w[0] - 1.0).abs() < 1e-10);
}

// ── CLA Max Sharpe ─────────────────────────────────────────────────────────

#[test]
fn cla_maxsharpe_diagonal_analytical() {
    // For diagonal cov, max-SR = Σ⁻¹μ / 1'Σ⁻¹μ
    let cov = array![[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.09]];
    let mu = array![0.05, 0.03, 0.08];
    let w = cla_max_sharpe(&mu, &cov).unwrap();

    // Σ⁻¹μ = [0.05/0.04, 0.03/0.01, 0.08/0.09] = [1.25, 3.0, 0.8889]
    let w_raw: Vec<f64> = [0.05 / 0.04, 0.03 / 0.01, 0.08 / 0.09].to_vec();
    let total: f64 = w_raw.iter().sum();
    for (i, (&actual, expected)) in w.iter().zip(w_raw.iter().map(|v| v / total)).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "MaxSR[{}]: got {}, expected {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn cla_maxsharpe_equal_returns_equals_minvar() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let mu = array![0.05, 0.05]; // equal returns
    let w_sr = cla_max_sharpe(&mu, &cov).unwrap();
    let w_mv = cla_min_variance(&cov).unwrap();
    for (a, b) in w_sr.iter().zip(w_mv.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Equal returns: max-SR={} should equal min-var={}",
            a,
            b
        );
    }
}

#[test]
fn cla_maxsharpe_higher_return_higher_weight() {
    // Uncorrelated, equal vol → weight ∝ expected return
    let cov = array![[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]];
    let mu = array![0.10, 0.05, 0.02];
    let w = cla_max_sharpe(&mu, &cov).unwrap();
    assert!(w[0] > w[1] && w[1] > w[2]);
}

#[test]
fn cla_maxsharpe_single_asset() {
    let mu = array![0.10];
    let cov = array![[0.04]];
    let w = cla_max_sharpe(&mu, &cov).unwrap();
    assert!((w[0] - 1.0).abs() < 1e-10);
}

// ── HRP ────────────────────────────────────────────────────────────────────

fn make_block_returns(seed: u64) -> Array2<f64> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = 200;
    let n_assets = 4;

    // Block correlation: assets 0-1 correlated, 2-3 correlated
    let mut corr = Array2::<f64>::eye(n_assets);
    corr[[0, 1]] = 0.8;
    corr[[1, 0]] = 0.8;
    corr[[2, 3]] = 0.7;
    corr[[3, 2]] = 0.7;

    let vols = array![0.02, 0.025, 0.01, 0.012];
    let mut cov = Array2::zeros((n_assets, n_assets));
    for i in 0..n_assets {
        for j in 0..n_assets {
            cov[[i, j]] = vols[i] * vols[j] * corr[[i, j]];
        }
    }

    // Manual Cholesky for 4x4 positive-definite matrix
    let l = {
        let mut l = Array2::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    l[[i, j]] = (cov[[i, j]] - sum).sqrt();
                } else {
                    l[[i, j]] = (cov[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        l
    };

    let mut returns = Array2::zeros((n, n_assets));
    for t in 0..n {
        let z: Vec<f64> = (0..n_assets).map(|_| normal.sample(&mut rng)).collect();
        for i in 0..n_assets {
            let mut val = 0.0;
            for j in 0..=i {
                val += l[[i, j]] * z[j];
            }
            returns[[t, i]] = val;
        }
    }
    returns
}

#[test]
fn hrp_sum_to_one() {
    let returns = make_block_returns(42);
    let w = hrp_weights(&returns).unwrap();
    assert!((w.sum() - 1.0).abs() < 1e-10);
}

#[test]
fn hrp_all_positive() {
    let returns = make_block_returns(42);
    let w = hrp_weights(&returns).unwrap();
    for (i, &wi) in w.iter().enumerate() {
        assert!(wi > 0.0, "HRP weight[{}] = {} should be positive", i, wi);
    }
}

#[test]
fn hrp_correct_length() {
    let returns = make_block_returns(42);
    let w = hrp_weights(&returns).unwrap();
    assert_eq!(w.len(), 4);
}

#[test]
fn hrp_single_asset() {
    let returns = array![[0.01], [-0.005], [0.002], [0.008]];
    let w = hrp_weights(&returns).unwrap();
    assert_eq!(w.len(), 1);
    assert!((w[0] - 1.0).abs() < 1e-10);
}

#[test]
fn hrp_deterministic() {
    let returns = make_block_returns(42);
    let w1 = hrp_weights(&returns).unwrap();
    let w2 = hrp_weights(&returns).unwrap();
    for (a, b) in w1.iter().zip(w2.iter()) {
        assert_eq!(*a, *b);
    }
}

#[test]
fn hrp_cluster_structure_respected() {
    // Assets 0-1 are high-vol, 2-3 are low-vol.
    // Low-vol cluster should get more total weight.
    let returns = make_block_returns(42);
    let w = hrp_weights(&returns).unwrap();
    let high_vol_total = w[0] + w[1];
    let low_vol_total = w[2] + w[3];
    assert!(
        low_vol_total > high_vol_total,
        "Low-vol cluster ({:.4}) should get more weight than high-vol ({:.4})",
        low_vol_total,
        high_vol_total
    );
}

#[test]
fn hrp_lower_vol_higher_weight() {
    // Two uncorrelated assets with different vols
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = 1000;
    let mut returns = Array2::zeros((n, 2));
    for t in 0..n {
        returns[[t, 0]] = normal.sample(&mut rng) * 0.01; // low vol
        returns[[t, 1]] = normal.sample(&mut rng) * 0.03; // high vol
    }
    let w = hrp_weights(&returns).unwrap();
    assert!(
        w[0] > w[1],
        "Low-vol asset should have higher weight: w[0]={}, w[1]={}",
        w[0],
        w[1]
    );
}

// ── Compare Allocations (Monte Carlo) ──────────────────────────────────────

#[test]
fn compare_allocations_variances_positive() {
    let returns = make_block_returns(42);
    let mc = compare_allocations(&returns, 10, 42).unwrap();
    assert!(mc.hrp_variance > 0.0);
    assert!(mc.cla_variance > 0.0);
    assert!(mc.ivp_variance > 0.0);
}

#[test]
fn compare_allocations_sharpes_finite() {
    let returns = make_block_returns(42);
    let mc = compare_allocations(&returns, 10, 42).unwrap();
    assert!(mc.hrp_sharpe.is_finite());
    assert!(mc.cla_sharpe.is_finite());
    assert!(mc.ivp_sharpe.is_finite());
}

#[test]
fn compare_allocations_deterministic() {
    let returns = make_block_returns(42);
    let mc1 = compare_allocations(&returns, 10, 42).unwrap();
    let mc2 = compare_allocations(&returns, 10, 42).unwrap();
    assert_eq!(mc1.hrp_sharpe, mc2.hrp_sharpe);
    assert_eq!(mc1.cla_sharpe, mc2.cla_sharpe);
    assert_eq!(mc1.ivp_sharpe, mc2.ivp_sharpe);
}

// ── Cross-method consistency ───────────────────────────────────────────────

#[test]
fn all_methods_sum_to_one() {
    let returns = make_block_returns(42);
    let cov = {
        let n = returns.nrows();
        let means: Array1<f64> = returns.mean_axis(ndarray::Axis(0)).unwrap();
        let centered = &returns - &means;
        centered.t().dot(&centered) / ((n - 1) as f64)
    };
    let mu = returns.mean_axis(ndarray::Axis(0)).unwrap();

    let w_hrp = hrp_weights(&returns).unwrap();
    let w_cla = cla_min_variance(&cov).unwrap();
    let w_sr = cla_max_sharpe(&mu, &cov).unwrap();
    let w_ivp = inverse_variance_weights(&cov);

    for (name, w) in [
        ("HRP", &w_hrp),
        ("CLA", &w_cla),
        ("MaxSR", &w_sr),
        ("IVP", &w_ivp),
    ] {
        assert!(
            (w.sum() - 1.0).abs() < 1e-8,
            "{} weights sum to {} (expected 1.0)",
            name,
            w.sum()
        );
    }
}

#[test]
fn cla_minvar_is_global_minimum() {
    // CLA min-var should have lower portfolio variance than IVP, HRP, and equal weight.
    let returns = make_block_returns(42);
    let n = returns.nrows();
    let n_assets = returns.ncols();
    let means: Array1<f64> = returns.mean_axis(ndarray::Axis(0)).unwrap();
    let centered = &returns - &means;
    let cov = centered.t().dot(&centered) / ((n - 1) as f64);

    let w_cla = cla_min_variance(&cov).unwrap();
    let w_ivp = inverse_variance_weights(&cov);
    let w_hrp = hrp_weights(&returns).unwrap();
    let w_eq = Array1::from_elem(n_assets, 1.0 / n_assets as f64);

    let var_cla = w_cla.dot(&cov.dot(&w_cla));
    let var_ivp = w_ivp.dot(&cov.dot(&w_ivp));
    let var_hrp = w_hrp.dot(&cov.dot(&w_hrp));
    let var_eq = w_eq.dot(&cov.dot(&w_eq));

    assert!(
        var_cla <= var_ivp + 1e-10,
        "CLA ({}) should <= IVP ({})",
        var_cla,
        var_ivp
    );
    assert!(
        var_cla <= var_eq + 1e-10,
        "CLA ({}) should <= EqW ({})",
        var_cla,
        var_eq
    );
    assert!(
        var_cla <= var_hrp + 1e-10,
        "CLA ({}) should <= HRP ({})",
        var_cla,
        var_hrp
    );
}

#[test]
fn hrp_always_positive_cla_may_not_be() {
    let returns = make_block_returns(42);
    let w_hrp = hrp_weights(&returns).unwrap();
    for &w in w_hrp.iter() {
        assert!(w > 0.0);
    }
    // CLA has no such guarantee (documenting, not asserting)
}
