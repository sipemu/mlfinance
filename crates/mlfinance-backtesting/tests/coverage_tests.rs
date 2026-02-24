use mlfinance_backtesting::bet_sizing::active_bets::avg_active_signals;
use mlfinance_backtesting::bet_sizing::discretization::discrete_signal;
use mlfinance_backtesting::bet_sizing::dynamic_sizing::{dynamic_bet_size, DynamicBetSize};
use mlfinance_backtesting::cpcv::combinatorial::CpcvSplit;
use mlfinance_backtesting::cpcv::path_builder::build_paths;
use mlfinance_backtesting::overfitting::pbo::probability_of_backtest_overfitting;
use mlfinance_backtesting::synthetic::optimal_trading_rule::otr_mesh;
use mlfinance_backtesting::synthetic::ornstein_uhlenbeck::{estimate_ou_params, simulate_ou};
use mlfinance_backtesting::synthetic::sharpe_mesh::sharpe_mesh;
use ndarray::Array2;

// ── avg_active_signals ──

#[test]
fn test_avg_active_signals_basic() {
    // signals: (start, end, value)
    let signals = vec![(0, 2, 1.0), (1, 3, 2.0), (3, 4, -1.0)];
    let avg = avg_active_signals(&signals, 5);
    assert_eq!(avg.len(), 5);
    // bar 0: only signal 0 active => 1.0
    assert!((avg[0] - 1.0).abs() < 1e-10);
    // bar 1: signals 0 and 1 active => (1.0 + 2.0) / 2 = 1.5
    assert!((avg[1] - 1.5).abs() < 1e-10);
}

#[test]
fn test_avg_active_signals_no_signals() {
    let avg = avg_active_signals(&[], 5);
    assert_eq!(avg, vec![0.0; 5]);
}

// ── dynamic_bet_size ──

#[test]
fn test_dynamic_bet_size_no_position() {
    let config = DynamicBetSize {
        target_position: 10.0,
        current_position: 0.0,
        max_position: 10.0,
        sigma: 1.0,
    };
    let (limit_delta, bet) = dynamic_bet_size(&config, 0.5);
    assert!(limit_delta.is_finite());
    assert!(bet.is_finite());
}

#[test]
fn test_dynamic_bet_size_at_target() {
    let config = DynamicBetSize {
        target_position: 5.0,
        current_position: 5.0,
        max_position: 10.0,
        sigma: 1.0,
    };
    let (_, bet) = dynamic_bet_size(&config, 0.5);
    // Already at target, bet should be small or zero
    assert!(bet.abs() < 5.1);
}

// ── discrete_signal ──

#[test]
fn test_discrete_signal() {
    assert!((discrete_signal(0.73, 0.25) - 0.75).abs() < 1e-10);
    assert!((discrete_signal(0.1, 0.5) - 0.0).abs() < 1e-10);
    assert!((discrete_signal(-0.73, 0.25) - (-0.75)).abs() < 1e-10);
}

#[test]
fn test_discrete_signal_zero_step() {
    assert_eq!(discrete_signal(0.5, 0.0), 0.0);
}

#[test]
fn test_discrete_signal_nan() {
    let result = discrete_signal(f64::NAN, 0.25);
    assert_eq!(result, 0.0);
}

// ── probability_of_backtest_overfitting ──

#[test]
fn test_pbo_basic() {
    // 20 time periods, 5 strategies
    let mut data = Vec::new();
    for i in 0..20 {
        for j in 0..5 {
            data.push(((i * 7 + j * 3) % 11) as f64 - 5.0);
        }
    }
    let returns = Array2::from_shape_vec((20, 5), data).unwrap();
    let pbo = probability_of_backtest_overfitting(&returns, 4, 42);
    assert!((0.0..=1.0).contains(&pbo));
}

// ── build_paths ──

#[test]
fn test_build_paths() {
    // 3 groups, 2 test groups per split => C(3,2)=3 splits
    let splits = vec![
        CpcvSplit {
            train_groups: vec![0],
            test_groups: vec![1, 2],
        },
        CpcvSplit {
            train_groups: vec![1],
            test_groups: vec![0, 2],
        },
        CpcvSplit {
            train_groups: vec![2],
            test_groups: vec![0, 1],
        },
    ];
    let paths = build_paths(&splits, 3);
    // Each path should cover all 3 groups
    for path in &paths {
        let mut covered: Vec<usize> = path
            .iter()
            .flat_map(|&s| splits[s].test_groups.clone())
            .collect();
        covered.sort();
        covered.dedup();
        assert_eq!(covered.len(), 3);
    }
}

// ── simulate_ou ──

#[test]
fn test_simulate_ou() {
    let path = simulate_ou(1.0, 0.0, 0.5, 0.0, 0.01, 1000, 42);
    assert_eq!(path.len(), 1001); // n_steps + 1
    assert_eq!(path[0], 0.0);
    // Should stay finite
    for &v in &path {
        assert!(v.is_finite());
    }
}

#[test]
fn test_simulate_ou_mean_reversion() {
    let path = simulate_ou(5.0, 10.0, 0.1, 0.0, 0.01, 5000, 42);
    // With strong mean reversion (theta=5), the path should approach mu=10
    let tail_mean: f64 = path[4000..].iter().sum::<f64>() / 1000.0;
    assert!(
        (tail_mean - 10.0).abs() < 3.0,
        "OU should revert to mu=10, tail mean was {tail_mean}"
    );
}

// ── estimate_ou_params ──

#[test]
fn test_estimate_ou_params_roundtrip() {
    let path = simulate_ou(2.0, 5.0, 0.3, 5.0, 0.01, 10000, 42);
    let (theta, mu, sigma) = estimate_ou_params(&path, 0.01);
    assert!(theta > 0.0, "theta should be positive, got {theta}");
    assert!((mu - 5.0).abs() < 2.0, "mu should be near 5.0, got {mu}");
    assert!(sigma > 0.0, "sigma should be positive, got {sigma}");
}

#[test]
fn test_estimate_ou_params_short_series() {
    let (theta, mu, sigma) = estimate_ou_params(&[1.0, 2.0], 0.01);
    assert!(theta.is_finite());
    assert!(mu.is_finite());
    assert!(sigma.is_finite());
}

// ── sharpe_mesh ──

#[test]
fn test_sharpe_mesh() {
    let returns = Array2::from_shape_vec(
        (4, 3),
        vec![
            0.01, 0.02, 0.03, 0.02, 0.01, 0.04, 0.015, 0.025, 0.035, 0.01, 0.03, 0.02,
        ],
    )
    .unwrap();
    let mesh = sharpe_mesh(&returns);
    assert_eq!(mesh.shape(), returns.shape());
    for &v in mesh.iter() {
        assert!(v.is_finite());
    }
}

// ── otr_mesh ──

#[test]
fn test_otr_mesh() {
    let pt_range = vec![0.5, 1.0, 1.5];
    let sl_range = vec![-0.5, -1.0];
    let mesh = otr_mesh(1.0, 0.0, 0.5, &pt_range, &sl_range, 50, 42);
    assert_eq!(mesh.shape(), &[3, 2]);
    for &v in mesh.iter() {
        assert!(v.is_finite());
    }
}
