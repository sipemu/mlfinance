use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;
use crate::types::*;

// ── Statistics ──────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (returns, risk_free_rate=0.0, periods_per_year=252.0))]
fn sharpe_ratio(
    returns: PyReadonlyArray1<'_, f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> f64 {
    let r = py_to_vec(returns);
    mlfinance::backtesting::statistics::sharpe::sharpe_ratio(&r, risk_free_rate, periods_per_year)
}

#[pyfunction]
fn probabilistic_sharpe_ratio(
    observed_sr: f64,
    benchmark_sr: f64,
    n_observations: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    mlfinance::backtesting::statistics::psr::probabilistic_sharpe_ratio(
        observed_sr,
        benchmark_sr,
        n_observations,
        skewness,
        kurtosis,
    )
}

#[pyfunction]
fn deflated_sharpe_ratio(
    observed_sr: f64,
    sr_std: f64,
    n_observations: usize,
    n_trials: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    mlfinance::backtesting::statistics::dsr::deflated_sharpe_ratio(
        observed_sr,
        sr_std,
        n_observations,
        n_trials,
        skewness,
        kurtosis,
    )
}

#[pyfunction]
fn compute_drawdowns(returns: PyReadonlyArray1<'_, f64>) -> PyDrawdownResult {
    let r = py_to_vec(returns);
    let result = mlfinance::backtesting::statistics::drawdown::compute_drawdowns(&r);
    PyDrawdownResult {
        max_drawdown: result.max_drawdown,
        max_drawdown_duration: result.max_drawdown_duration,
        drawdown_series: result.drawdown_series,
        time_under_water: result.time_under_water,
    }
}

#[pyfunction]
fn hhi(weights: PyReadonlyArray1<'_, f64>) -> f64 {
    let w = py_to_vec(weights);
    mlfinance::backtesting::statistics::hhi::hhi(&w)
}

#[pyfunction]
fn hhi_concentration(returns: PyReadonlyArray1<'_, f64>) -> (f64, f64) {
    let r = py_to_vec(returns);
    mlfinance::backtesting::statistics::hhi::hhi_concentration(&r)
}

#[pyfunction]
fn hit_ratio(returns: PyReadonlyArray1<'_, f64>) -> f64 {
    let r = py_to_vec(returns);
    mlfinance::backtesting::statistics::general::hit_ratio(&r)
}

#[pyfunction]
fn avg_holding_period(entry_exit_pairs: Vec<(usize, usize)>) -> f64 {
    mlfinance::backtesting::statistics::general::avg_holding_period(&entry_exit_pairs)
}

#[pyfunction]
fn turnover(positions: PyReadonlyArray1<'_, f64>) -> f64 {
    let p = py_to_vec(positions);
    mlfinance::backtesting::statistics::general::turnover(&p)
}

// ── Overfitting ─────────────────────────────────────────────────────────────

#[pyfunction]
fn probability_of_backtest_overfitting(
    returns_matrix: PyReadonlyArray2<'_, f64>,
    num_partitions: usize,
    seed: u64,
) -> f64 {
    let m = py_to_array2(returns_matrix);
    mlfinance::backtesting::overfitting::pbo::probability_of_backtest_overfitting(
        &m,
        num_partitions,
        seed,
    )
}

#[pyfunction]
fn cscv(returns_matrix: PyReadonlyArray2<'_, f64>, num_groups: usize) -> PyCscvResult {
    let m = py_to_array2(returns_matrix);
    let result = mlfinance::backtesting::overfitting::cscv::cscv(&m, num_groups);
    PyCscvResult {
        pbo: result.pbo,
        rank_logits: result.rank_logits,
    }
}

#[pyfunction]
fn bonferroni_correction(py: Python<'_>, p_values: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let pv = py_to_vec(p_values);
    let result = mlfinance::backtesting::overfitting::multiple_testing::bonferroni_correction(&pv);
    vec_to_py_array(py, result)
}

#[pyfunction]
fn holm_correction(py: Python<'_>, p_values: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let pv = py_to_vec(p_values);
    let result = mlfinance::backtesting::overfitting::multiple_testing::holm_correction(&pv);
    vec_to_py_array(py, result)
}

// ── Strategy risk ───────────────────────────────────────────────────────────

#[pyfunction]
fn sr_from_precision(precision: f64, freq: f64, avg_win_loss_ratio: f64) -> f64 {
    mlfinance::backtesting::strategy_risk::sr_from_precision::sr_from_precision(
        precision,
        freq,
        avg_win_loss_ratio,
    )
}

#[pyfunction]
fn implied_precision(target_sr: f64, freq: f64, avg_win_loss_ratio: f64) -> f64 {
    mlfinance::backtesting::strategy_risk::implied_precision::implied_precision(
        target_sr,
        freq,
        avg_win_loss_ratio,
    )
}

#[pyfunction]
fn implied_frequency(target_sr: f64, precision: f64, avg_win_loss_ratio: f64) -> f64 {
    mlfinance::backtesting::strategy_risk::implied_frequency::implied_frequency(
        target_sr,
        precision,
        avg_win_loss_ratio,
    )
}

#[pyfunction]
fn strategy_failure_probability(
    estimated_precision: f64,
    n_observations: usize,
    break_even_precision: f64,
) -> f64 {
    mlfinance::backtesting::strategy_risk::failure_probability::strategy_failure_probability(
        estimated_precision,
        n_observations,
        break_even_precision,
    )
}

// ── Bet sizing ──────────────────────────────────────────────────────────────

#[pyfunction]
fn sigmoid_bet_size(prob: f64, num_classes: usize) -> f64 {
    mlfinance::backtesting::bet_sizing::probability_to_size::sigmoid_bet_size(prob, num_classes)
}

#[pyfunction]
fn power_bet_size(prob: f64, num_classes: usize, exponent: f64) -> f64 {
    mlfinance::backtesting::bet_sizing::probability_to_size::power_bet_size(
        prob,
        num_classes,
        exponent,
    )
}

#[pyfunction]
fn discrete_signal(signal: f64, step_size: f64) -> f64 {
    mlfinance::backtesting::bet_sizing::discretization::discrete_signal(signal, step_size)
}

#[pyfunction]
fn avg_active_signals(
    py: Python<'_>,
    signals: Vec<(usize, usize, f64)>,
    num_bars: usize,
) -> Py<PyArray1<f64>> {
    let result =
        mlfinance::backtesting::bet_sizing::active_bets::avg_active_signals(&signals, num_bars);
    vec_to_py_array(py, result)
}

// ── Synthetic ───────────────────────────────────────────────────────────────

#[pyfunction]
fn simulate_ou(
    py: Python<'_>,
    theta: f64,
    mu: f64,
    sigma: f64,
    x0: f64,
    dt: f64,
    n_steps: usize,
    seed: u64,
) -> Py<PyArray1<f64>> {
    let result = mlfinance::backtesting::synthetic::ornstein_uhlenbeck::simulate_ou(
        theta, mu, sigma, x0, dt, n_steps, seed,
    );
    vec_to_py_array(py, result)
}

#[pyfunction]
fn estimate_ou_params(series: PyReadonlyArray1<'_, f64>, dt: f64) -> (f64, f64, f64) {
    let s = py_to_vec(series);
    mlfinance::backtesting::synthetic::ornstein_uhlenbeck::estimate_ou_params(&s, dt)
}

#[pyfunction]
fn otr_mesh(
    py: Python<'_>,
    theta: f64,
    mu: f64,
    sigma: f64,
    pt_range: PyReadonlyArray1<'_, f64>,
    sl_range: PyReadonlyArray1<'_, f64>,
    n_simulations: usize,
    seed: u64,
) -> Py<PyArray2<f64>> {
    let pt = py_to_vec(pt_range);
    let sl = py_to_vec(sl_range);
    let result = mlfinance::backtesting::synthetic::optimal_trading_rule::otr_mesh(
        theta,
        mu,
        sigma,
        &pt,
        &sl,
        n_simulations,
        seed,
    );
    array2_to_py(py, result)
}

#[pyfunction]
fn sharpe_mesh(py: Python<'_>, returns_grid: PyReadonlyArray2<'_, f64>) -> Py<PyArray2<f64>> {
    let m = py_to_array2(returns_grid);
    let result = mlfinance::backtesting::synthetic::sharpe_mesh::sharpe_mesh(&m);
    array2_to_py(py, result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Statistics
    m.add_function(wrap_pyfunction!(sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(probabilistic_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(deflated_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(compute_drawdowns, m)?)?;
    m.add_function(wrap_pyfunction!(hhi, m)?)?;
    m.add_function(wrap_pyfunction!(hhi_concentration, m)?)?;
    m.add_function(wrap_pyfunction!(hit_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(avg_holding_period, m)?)?;
    m.add_function(wrap_pyfunction!(turnover, m)?)?;
    // Overfitting
    m.add_function(wrap_pyfunction!(probability_of_backtest_overfitting, m)?)?;
    m.add_function(wrap_pyfunction!(cscv, m)?)?;
    m.add_function(wrap_pyfunction!(bonferroni_correction, m)?)?;
    m.add_function(wrap_pyfunction!(holm_correction, m)?)?;
    // Strategy risk
    m.add_function(wrap_pyfunction!(sr_from_precision, m)?)?;
    m.add_function(wrap_pyfunction!(implied_precision, m)?)?;
    m.add_function(wrap_pyfunction!(implied_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(strategy_failure_probability, m)?)?;
    // Bet sizing
    m.add_function(wrap_pyfunction!(sigmoid_bet_size, m)?)?;
    m.add_function(wrap_pyfunction!(power_bet_size, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_signal, m)?)?;
    m.add_function(wrap_pyfunction!(avg_active_signals, m)?)?;
    // Synthetic
    m.add_function(wrap_pyfunction!(simulate_ou, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_ou_params, m)?)?;
    m.add_function(wrap_pyfunction!(otr_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_mesh, m)?)?;
    Ok(())
}
