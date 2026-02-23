use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;
use crate::types::*;

// ── Statistics ──────────────────────────────────────────────────────────────

/// Annualized Sharpe ratio.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Periodic return series.
/// risk_free_rate : float, default 0.0
///     Risk-free rate per period.
/// periods_per_year : float, default 252.0
///     Annualization factor (252 for daily, 12 for monthly).
///
/// Returns
/// -------
/// float
///     Annualized Sharpe ratio.
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

/// Probabilistic Sharpe Ratio (PSR) — probability that the true SR exceeds a benchmark.
///
/// Accounts for skewness and kurtosis of the return distribution (AFML Ch. 14).
///
/// Parameters
/// ----------
/// observed_sr : float
///     Observed (sample) Sharpe ratio.
/// benchmark_sr : float
///     Benchmark Sharpe ratio to compare against.
/// n_observations : int
///     Number of return observations.
/// skewness : float
///     Skewness of returns.
/// kurtosis : float
///     Excess kurtosis of returns.
///
/// Returns
/// -------
/// float
///     Probability (0 to 1) that the true SR exceeds the benchmark.
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

/// Deflated Sharpe Ratio — adjusts for multiple testing (AFML Ch. 14).
///
/// Corrects the observed SR for the number of strategies tested,
/// accounting for higher moments.
///
/// Parameters
/// ----------
/// observed_sr : float
///     Best observed Sharpe ratio.
/// sr_std : float
///     Standard deviation of Sharpe ratios across all trials.
/// n_observations : int
///     Number of return observations.
/// n_trials : int
///     Number of strategies/configurations tested.
/// skewness : float
///     Skewness of returns.
/// kurtosis : float
///     Excess kurtosis of returns.
///
/// Returns
/// -------
/// float
///     Deflated Sharpe ratio (probability-adjusted).
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

/// Compute drawdown statistics from a return series.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Periodic return series.
///
/// Returns
/// -------
/// DrawdownResult
///     Contains max drawdown, max duration, drawdown series, and
///     time-under-water series.
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

/// Herfindahl-Hirschman Index measuring concentration.
///
/// Parameters
/// ----------
/// weights : numpy.ndarray
///     Weight or share vector (should sum to 1).
///
/// Returns
/// -------
/// float
///     HHI value. Ranges from 1/n (perfectly diversified) to 1 (fully concentrated).
#[pyfunction]
fn hhi(weights: PyReadonlyArray1<'_, f64>) -> f64 {
    let w = py_to_vec(weights);
    mlfinance::backtesting::statistics::hhi::hhi(&w)
}

/// Concentration of positive and negative returns using HHI.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return series.
///
/// Returns
/// -------
/// tuple[float, float]
///     (positive_hhi, negative_hhi) — concentration in gains and losses.
#[pyfunction]
fn hhi_concentration(returns: PyReadonlyArray1<'_, f64>) -> (f64, f64) {
    let r = py_to_vec(returns);
    mlfinance::backtesting::statistics::hhi::hhi_concentration(&r)
}

/// Fraction of positive returns (hit rate / win rate).
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return series.
///
/// Returns
/// -------
/// float
///     Proportion of returns > 0.
#[pyfunction]
fn hit_ratio(returns: PyReadonlyArray1<'_, f64>) -> f64 {
    let r = py_to_vec(returns);
    mlfinance::backtesting::statistics::general::hit_ratio(&r)
}

/// Average holding period from entry-exit pairs.
///
/// Parameters
/// ----------
/// entry_exit_pairs : list[tuple[int, int]]
///     List of (entry_idx, exit_idx) pairs.
///
/// Returns
/// -------
/// float
///     Mean holding period in bars.
#[pyfunction]
fn avg_holding_period(entry_exit_pairs: Vec<(usize, usize)>) -> f64 {
    mlfinance::backtesting::statistics::general::avg_holding_period(&entry_exit_pairs)
}

/// Portfolio turnover from a position series.
///
/// Parameters
/// ----------
/// positions : numpy.ndarray
///     Position sizes over time.
///
/// Returns
/// -------
/// float
///     Average absolute change in position per period.
#[pyfunction]
fn turnover(positions: PyReadonlyArray1<'_, f64>) -> f64 {
    let p = py_to_vec(positions);
    mlfinance::backtesting::statistics::general::turnover(&p)
}

// ── Overfitting ─────────────────────────────────────────────────────────────

/// Probability of Backtest Overfitting (PBO) via CSCV (AFML Ch. 14).
///
/// Estimates the probability that the best in-sample strategy will
/// underperform out-of-sample using combinatorial data splitting.
///
/// Parameters
/// ----------
/// returns_matrix : numpy.ndarray
///     Matrix of shape (n_periods, n_strategies) with period returns.
/// num_partitions : int
///     Number of partitions for CSCV (must be even).
/// seed : int
///     Random seed for partition shuffling.
///
/// Returns
/// -------
/// float
///     PBO estimate in [0, 1]. Higher means more overfitting risk.
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

/// Combinatorially Symmetric Cross-Validation (CSCV).
///
/// Parameters
/// ----------
/// returns_matrix : numpy.ndarray
///     Matrix of shape (n_periods, n_strategies).
/// num_groups : int
///     Number of groups for combinatorial splitting.
///
/// Returns
/// -------
/// CscvResult
///     PBO and rank-logit vector.
#[pyfunction]
fn cscv(returns_matrix: PyReadonlyArray2<'_, f64>, num_groups: usize) -> PyCscvResult {
    let m = py_to_array2(returns_matrix);
    let result = mlfinance::backtesting::overfitting::cscv::cscv(&m, num_groups);
    PyCscvResult {
        pbo: result.pbo,
        rank_logits: result.rank_logits,
    }
}

/// Bonferroni correction for multiple hypothesis testing.
///
/// Parameters
/// ----------
/// p_values : numpy.ndarray
///     Uncorrected p-values.
///
/// Returns
/// -------
/// numpy.ndarray
///     Corrected p-values (clamped to [0, 1]).
#[pyfunction]
fn bonferroni_correction(py: Python<'_>, p_values: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let pv = py_to_vec(p_values);
    let result = mlfinance::backtesting::overfitting::multiple_testing::bonferroni_correction(&pv);
    vec_to_py_array(py, result)
}

/// Holm-Bonferroni step-down correction for multiple testing.
///
/// Less conservative than Bonferroni while still controlling the
/// family-wise error rate.
///
/// Parameters
/// ----------
/// p_values : numpy.ndarray
///     Uncorrected p-values.
///
/// Returns
/// -------
/// numpy.ndarray
///     Corrected p-values.
#[pyfunction]
fn holm_correction(py: Python<'_>, p_values: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let pv = py_to_vec(p_values);
    let result = mlfinance::backtesting::overfitting::multiple_testing::holm_correction(&pv);
    vec_to_py_array(py, result)
}

// ── Strategy risk ───────────────────────────────────────────────────────────

/// Compute Sharpe ratio from precision, frequency, and win/loss ratio.
///
/// Parameters
/// ----------
/// precision : float
///     Hit rate (probability of a winning trade).
/// freq : float
///     Average number of trades per year.
/// avg_win_loss_ratio : float
///     Average win size divided by average loss size.
///
/// Returns
/// -------
/// float
///     Expected annualized Sharpe ratio.
#[pyfunction]
fn sr_from_precision(precision: f64, freq: f64, avg_win_loss_ratio: f64) -> f64 {
    mlfinance::backtesting::strategy_risk::sr_from_precision::sr_from_precision(
        precision,
        freq,
        avg_win_loss_ratio,
    )
}

/// Implied precision needed to achieve a target Sharpe ratio.
///
/// Parameters
/// ----------
/// target_sr : float
///     Desired annualized Sharpe ratio.
/// freq : float
///     Average trades per year.
/// avg_win_loss_ratio : float
///     Average win/loss ratio.
///
/// Returns
/// -------
/// float
///     Required hit rate (precision).
#[pyfunction]
fn implied_precision(target_sr: f64, freq: f64, avg_win_loss_ratio: f64) -> f64 {
    mlfinance::backtesting::strategy_risk::implied_precision::implied_precision(
        target_sr,
        freq,
        avg_win_loss_ratio,
    )
}

/// Implied trading frequency needed to achieve a target Sharpe ratio.
///
/// Parameters
/// ----------
/// target_sr : float
///     Desired annualized Sharpe ratio.
/// precision : float
///     Hit rate.
/// avg_win_loss_ratio : float
///     Average win/loss ratio.
///
/// Returns
/// -------
/// float
///     Required trades per year.
#[pyfunction]
fn implied_frequency(target_sr: f64, precision: f64, avg_win_loss_ratio: f64) -> f64 {
    mlfinance::backtesting::strategy_risk::implied_frequency::implied_frequency(
        target_sr,
        precision,
        avg_win_loss_ratio,
    )
}

/// Probability that a strategy will fail (precision below break-even).
///
/// Uses a binomial test to estimate the probability that the true
/// precision is below the break-even threshold.
///
/// Parameters
/// ----------
/// estimated_precision : float
///     Observed hit rate.
/// n_observations : int
///     Number of trades observed.
/// break_even_precision : float
///     Minimum precision needed to break even.
///
/// Returns
/// -------
/// float
///     Failure probability.
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

/// Sigmoid bet sizing from prediction probability (AFML Ch. 10).
///
/// Maps a probability to a position size using a sigmoid function.
///
/// Parameters
/// ----------
/// prob : float
///     Predicted probability of the positive class.
/// num_classes : int
///     Number of discrete classes (usually 2).
///
/// Returns
/// -------
/// float
///     Bet size in [-1, 1].
#[pyfunction]
fn sigmoid_bet_size(prob: f64, num_classes: usize) -> f64 {
    mlfinance::backtesting::bet_sizing::probability_to_size::sigmoid_bet_size(prob, num_classes)
}

/// Power-law bet sizing from prediction probability.
///
/// Parameters
/// ----------
/// prob : float
///     Predicted probability.
/// num_classes : int
///     Number of classes.
/// exponent : float
///     Power exponent (higher = more aggressive sizing).
///
/// Returns
/// -------
/// float
///     Bet size in [-1, 1].
#[pyfunction]
fn power_bet_size(prob: f64, num_classes: usize, exponent: f64) -> f64 {
    mlfinance::backtesting::bet_sizing::probability_to_size::power_bet_size(
        prob,
        num_classes,
        exponent,
    )
}

/// Discretize a continuous signal to the nearest step.
///
/// Parameters
/// ----------
/// signal : float
///     Continuous signal value.
/// step_size : float
///     Discretization step (e.g. 0.1 for 10% increments).
///
/// Returns
/// -------
/// float
///     Rounded signal.
#[pyfunction]
fn discrete_signal(signal: f64, step_size: f64) -> f64 {
    mlfinance::backtesting::bet_sizing::discretization::discrete_signal(signal, step_size)
}

/// Average active signals at each bar.
///
/// Computes the mean signal strength from all active positions at each point.
///
/// Parameters
/// ----------
/// signals : list[tuple[int, int, float]]
///     List of (start_idx, end_idx, signal_value) triples.
/// num_bars : int
///     Total number of bars.
///
/// Returns
/// -------
/// numpy.ndarray
///     Average signal at each bar.
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

/// Simulate an Ornstein-Uhlenbeck mean-reverting process.
///
/// ``dX = theta * (mu - X) * dt + sigma * dW``
///
/// Parameters
/// ----------
/// theta : float
///     Mean-reversion speed.
/// mu : float
///     Long-term mean.
/// sigma : float
///     Volatility.
/// x0 : float
///     Initial value.
/// dt : float
///     Time step.
/// n_steps : int
///     Number of simulation steps.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// numpy.ndarray
///     Simulated path of length n_steps + 1 (including x0).
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

/// Estimate O-U process parameters from a time series via OLS.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Observed time series.
/// dt : float
///     Time step between observations.
///
/// Returns
/// -------
/// tuple[float, float, float]
///     (theta, mu, sigma) — mean-reversion speed, long-term mean, volatility.
#[pyfunction]
fn estimate_ou_params(series: PyReadonlyArray1<'_, f64>, dt: f64) -> (f64, f64, f64) {
    let s = py_to_vec(series);
    mlfinance::backtesting::synthetic::ornstein_uhlenbeck::estimate_ou_params(&s, dt)
}

/// Compute an optimal trading rule (OTR) mesh over profit-take and stop-loss ranges.
///
/// Simulates O-U processes and evaluates average returns for each
/// (profit_take, stop_loss) combination.
///
/// Parameters
/// ----------
/// theta : float
///     Mean-reversion speed.
/// mu : float
///     Long-term mean.
/// sigma : float
///     Volatility.
/// pt_range : numpy.ndarray
///     Profit-take thresholds to evaluate.
/// sl_range : numpy.ndarray
///     Stop-loss thresholds to evaluate.
/// n_simulations : int
///     Monte Carlo simulations per grid point.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// numpy.ndarray
///     2-D array of shape (len(pt_range), len(sl_range)) with average returns.
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

/// Compute Sharpe ratios over a grid of return series.
///
/// Parameters
/// ----------
/// returns_grid : numpy.ndarray
///     2-D array where each column is a return series.
///
/// Returns
/// -------
/// numpy.ndarray
///     Sharpe ratio for each column, reshaped to match the input grid.
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
