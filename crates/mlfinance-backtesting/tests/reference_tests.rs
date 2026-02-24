use mlfinance_backtesting::bet_sizing::probability_to_size::{power_bet_size, sigmoid_bet_size};
use mlfinance_backtesting::overfitting::cscv::cscv;
use mlfinance_backtesting::overfitting::multiple_testing::{
    bonferroni_correction, holm_correction,
};
use mlfinance_backtesting::statistics::drawdown::compute_drawdowns;
use mlfinance_backtesting::statistics::dsr::deflated_sharpe_ratio;
use mlfinance_backtesting::statistics::general::{avg_holding_period, hit_ratio, turnover};
use mlfinance_backtesting::statistics::hhi::{hhi, hhi_concentration};
use mlfinance_backtesting::statistics::psr::probabilistic_sharpe_ratio;
use mlfinance_backtesting::statistics::sharpe::sharpe_ratio;
use mlfinance_backtesting::strategy_risk::failure_probability::strategy_failure_probability;
use mlfinance_backtesting::strategy_risk::implied_precision::implied_precision;
use mlfinance_backtesting::strategy_risk::sr_from_precision::sr_from_precision;
use ndarray::Array2;
use serde_json::Value;
use std::fs;

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
}

fn load_fixture(name: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(fixture_path(name)).unwrap()).unwrap()
}

fn parse_f64_array(val: &Value) -> Vec<f64> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[test]
fn test_bet_sizing_match_python() {
    let data = load_fixture("bet_sizing.json");

    for case in data["cases"].as_array().unwrap() {
        let case_type = case["type"].as_str().unwrap();
        let prob = case["prob"].as_f64().unwrap();
        let num_classes = case["num_classes"].as_u64().unwrap() as usize;
        let expected = case["result"].as_f64().unwrap();

        match case_type {
            "sigmoid" => {
                let actual = sigmoid_bet_size(prob, num_classes);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Sigmoid mismatch: prob={}, classes={}, got={}, expected={}",
                    prob,
                    num_classes,
                    actual,
                    expected
                );
            }
            "power" => {
                let exponent = case["exponent"].as_f64().unwrap();
                let actual = power_bet_size(prob, num_classes, exponent);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Power mismatch: prob={}, classes={}, exp={}, got={}, expected={}",
                    prob,
                    num_classes,
                    exponent,
                    actual,
                    expected
                );
            }
            _ => panic!("Unknown case type: {}", case_type),
        }
    }
}

#[test]
fn test_sharpe_ratio_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "sharpe_ratio" {
            continue;
        }
        let returns = parse_f64_array(&case["returns"]);
        let rf = case["risk_free_rate"].as_f64().unwrap();
        let periods = case["periods_per_year"].as_f64().unwrap();
        let expected = case["result"].as_f64().unwrap();

        let actual = sharpe_ratio(&returns, rf, periods);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Sharpe mismatch: rf={}, periods={}, got={}, expected={}",
            rf,
            periods,
            actual,
            expected
        );
    }
}

#[test]
fn test_hit_ratio_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "hit_ratio" {
            continue;
        }
        let returns = parse_f64_array(&case["returns"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = hit_ratio(&returns);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Hit ratio mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_avg_holding_period_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "avg_holding_period" {
            continue;
        }
        let pairs: Vec<(usize, usize)> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let arr = p.as_array().unwrap();
                (
                    arr[0].as_u64().unwrap() as usize,
                    arr[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let expected = case["result"].as_f64().unwrap();

        let actual = avg_holding_period(&pairs);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Avg holding period mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_turnover_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "turnover" {
            continue;
        }
        let positions = parse_f64_array(&case["positions"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = turnover(&positions);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Turnover mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_hhi_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "hhi" {
            continue;
        }
        let weights = parse_f64_array(&case["weights"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = hhi(&weights);
        assert!(
            (actual - expected).abs() < 1e-10,
            "HHI mismatch: got={}, expected={}",
            actual,
            expected
        );
    }
}

#[test]
fn test_hhi_concentration_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "hhi_concentration" {
            continue;
        }
        let returns = parse_f64_array(&case["returns"]);
        let expected_pos = case["hhi_positive"].as_f64().unwrap();
        let expected_neg = case["hhi_negative"].as_f64().unwrap();

        let (actual_pos, actual_neg) = hhi_concentration(&returns);
        assert!(
            (actual_pos - expected_pos).abs() < 1e-10,
            "HHI pos mismatch: got={}, expected={}",
            actual_pos,
            expected_pos
        );
        assert!(
            (actual_neg - expected_neg).abs() < 1e-10,
            "HHI neg mismatch: got={}, expected={}",
            actual_neg,
            expected_neg
        );
    }
}

#[test]
fn test_drawdowns_match_python() {
    let data = load_fixture("drawdowns.json");

    for case in data["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let returns = parse_f64_array(&case["returns"]);
        let expected_max_dd = case["max_drawdown"].as_f64().unwrap();
        let expected_max_dur = case["max_drawdown_duration"].as_u64().unwrap() as usize;
        let expected_dd_series = parse_f64_array(&case["drawdown_series"]);
        let expected_tuw: Vec<usize> = case["time_under_water"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let dd = compute_drawdowns(&returns);

        assert!(
            (dd.max_drawdown - expected_max_dd).abs() < 1e-10,
            "{}: max_drawdown mismatch: got={}, expected={}",
            label,
            dd.max_drawdown,
            expected_max_dd
        );
        assert_eq!(
            dd.max_drawdown_duration, expected_max_dur,
            "{}: max_drawdown_duration mismatch",
            label
        );
        assert_eq!(dd.drawdown_series.len(), expected_dd_series.len());
        for (i, (actual, expected)) in dd
            .drawdown_series
            .iter()
            .zip(expected_dd_series.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-10,
                "{}: drawdown_series[{}] mismatch: got={}, expected={}",
                label,
                i,
                actual,
                expected
            );
        }
        assert_eq!(
            dd.time_under_water, expected_tuw,
            "{}: time_under_water mismatch",
            label
        );
    }
}

#[test]
fn test_psr_match_python() {
    let data = load_fixture("psr_dsr.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "psr" {
            continue;
        }
        let observed_sr = case["observed_sr"].as_f64().unwrap();
        let benchmark_sr = case["benchmark_sr"].as_f64().unwrap();
        let n_obs = case["n_observations"].as_u64().unwrap() as usize;
        let skewness = case["skewness"].as_f64().unwrap();
        let kurtosis = case["kurtosis"].as_f64().unwrap();
        let expected = case["result"].as_f64().unwrap();

        let actual =
            probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skewness, kurtosis);
        assert!(
            (actual - expected).abs() < 1e-6,
            "PSR mismatch: sr={}, bench={}, n={}, got={}, expected={}",
            observed_sr,
            benchmark_sr,
            n_obs,
            actual,
            expected
        );
    }
}

#[test]
fn test_dsr_match_python() {
    let data = load_fixture("psr_dsr.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "dsr" {
            continue;
        }
        let observed_sr = case["observed_sr"].as_f64().unwrap();
        let sr_std = case["sr_std"].as_f64().unwrap();
        let n_obs = case["n_observations"].as_u64().unwrap() as usize;
        let n_trials = case["n_trials"].as_u64().unwrap() as usize;
        let skewness = case["skewness"].as_f64().unwrap();
        let kurtosis = case["kurtosis"].as_f64().unwrap();
        let expected = case["result"].as_f64().unwrap();

        let actual =
            deflated_sharpe_ratio(observed_sr, sr_std, n_obs, n_trials, skewness, kurtosis);
        assert!(
            (actual - expected).abs() < 1e-6,
            "DSR mismatch: sr={}, trials={}, got={}, expected={}",
            observed_sr,
            n_trials,
            actual,
            expected
        );
    }
}

#[test]
fn test_strategy_failure_probability_match_python() {
    let data = load_fixture("strategy_risk.json");

    for case in data["cases"].as_array().unwrap() {
        let precision = case["estimated_precision"].as_f64().unwrap();
        let n_obs = case["n_observations"].as_u64().unwrap() as usize;
        let break_even = case["break_even_precision"].as_f64().unwrap();
        let expected = case["result"].as_f64().unwrap();

        let actual = strategy_failure_probability(precision, n_obs, break_even);
        assert!(
            (actual - expected).abs() < 1e-6,
            "Failure prob mismatch: p={}, n={}, be={}, got={}, expected={}",
            precision,
            n_obs,
            break_even,
            actual,
            expected
        );
    }
}

#[test]
fn test_bonferroni_match_python() {
    let data = load_fixture("multiple_testing.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "bonferroni" {
            continue;
        }
        let p_values = parse_f64_array(&case["p_values"]);
        let expected = parse_f64_array(&case["adjusted"]);

        let actual = bonferroni_correction(&p_values);
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "Bonferroni[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}

#[test]
fn test_holm_match_python() {
    let data = load_fixture("multiple_testing.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "holm" {
            continue;
        }
        let p_values = parse_f64_array(&case["p_values"]);
        let expected = parse_f64_array(&case["adjusted"]);

        let actual = holm_correction(&p_values);
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "Holm[{}] mismatch: got={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}

#[test]
fn test_cscv_match_python() {
    let data = load_fixture("backtesting_statistics.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "cscv" {
            continue;
        }
        let returns_raw: Vec<Vec<f64>> = case["returns_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(parse_f64_array)
            .collect();
        let n_rows = returns_raw.len();
        let n_cols = returns_raw[0].len();
        let flat: Vec<f64> = returns_raw.into_iter().flatten().collect();
        let returns_matrix = Array2::from_shape_vec((n_rows, n_cols), flat).unwrap();
        let num_groups = case["num_groups"].as_u64().unwrap() as usize;
        let expected_pbo = case["pbo"].as_f64().unwrap();
        let expected_logits = parse_f64_array(&case["rank_logits"]);

        let result = cscv(&returns_matrix, num_groups);
        assert!(
            (result.pbo - expected_pbo).abs() < 1e-4,
            "CSCV PBO mismatch: got={}, expected={}",
            result.pbo,
            expected_pbo
        );
        assert_eq!(
            result.rank_logits.len(),
            expected_logits.len(),
            "CSCV rank_logits length mismatch"
        );
        for (i, (a, e)) in result
            .rank_logits
            .iter()
            .zip(expected_logits.iter())
            .enumerate()
        {
            if e.is_infinite() {
                assert!(a.is_infinite(), "CSCV logit[{}] should be inf", i);
            } else {
                assert!(
                    (a - e).abs() < 1e-4,
                    "CSCV logit[{}] mismatch: got={}, expected={}",
                    i,
                    a,
                    e
                );
            }
        }
    }
}

// =====================================================================
// P1 — Strategy risk extended reference tests
// =====================================================================

#[test]
fn test_sr_from_precision_match_python() {
    let data = load_fixture("strategy_risk_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "sr_from_precision" {
            continue;
        }
        let precision = case["precision"].as_f64().unwrap();
        let freq = case["freq"].as_f64().unwrap();
        let ratio = case["avg_win_loss_ratio"].as_f64().unwrap();
        let expected = case["result"].as_f64().unwrap();

        let actual = sr_from_precision(precision, freq, ratio);
        assert!(
            (actual - expected).abs() < 1e-10,
            "SR from precision mismatch: p={}, freq={}, ratio={}, got={}, expected={}",
            precision,
            freq,
            ratio,
            actual,
            expected
        );
    }
}

#[test]
fn test_implied_precision_match_python() {
    let data = load_fixture("strategy_risk_extended.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "implied_precision" {
            continue;
        }
        let target_sr = case["target_sr"].as_f64().unwrap();
        let freq = case["freq"].as_f64().unwrap();
        let ratio = case["avg_win_loss_ratio"].as_f64().unwrap();
        let expected = case["result"].as_f64().unwrap();

        let actual = implied_precision(target_sr, freq, ratio);
        assert!(
            (actual - expected).abs() < 1e-4,
            "Implied precision mismatch: sr={}, freq={}, ratio={}, got={}, expected={}",
            target_sr,
            freq,
            ratio,
            actual,
            expected
        );
    }
}
