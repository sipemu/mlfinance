from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def avg_active_signals(signals, num_bars):
    """
    Average active signals at each bar.

    Computes the mean signal strength from all active positions at each point.

    Parameters
    ----------
    signals : list[tuple[int, int, float]]
        List of (start_idx, end_idx, signal_value) triples.
    num_bars : int
        Total number of bars.

    Returns
    -------
    numpy.ndarray
        Average signal at each bar.
    """
    ...

def avg_holding_period(entry_exit_pairs):
    """
    Average holding period from entry-exit pairs.

    Parameters
    ----------
    entry_exit_pairs : list[tuple[int, int]]
        List of (entry_idx, exit_idx) pairs.

    Returns
    -------
    float
        Mean holding period in bars.
    """
    ...

def bonferroni_correction(p_values):
    """
    Bonferroni correction for multiple hypothesis testing.

    Parameters
    ----------
    p_values : numpy.ndarray
        Uncorrected p-values.

    Returns
    -------
    numpy.ndarray
        Corrected p-values (clamped to [0, 1]).
    """
    ...

def compute_drawdowns(returns):
    """
    Compute drawdown statistics from a return series.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic return series.

    Returns
    -------
    DrawdownResult
        Contains max drawdown, max duration, drawdown series, and
        time-under-water series.
    """
    ...

def cscv(returns_matrix, num_groups):
    """
    Combinatorially Symmetric Cross-Validation (CSCV).

    Parameters
    ----------
    returns_matrix : numpy.ndarray
        Matrix of shape (n_periods, n_strategies).
    num_groups : int
        Number of groups for combinatorial splitting.

    Returns
    -------
    CscvResult
        PBO and rank-logit vector.
    """
    ...

def deflated_sharpe_ratio(observed_sr, sr_std, n_observations, n_trials, skewness, kurtosis):
    """
    Deflated Sharpe Ratio — adjusts for multiple testing (AFML Ch. 14).

    Corrects the observed SR for the number of strategies tested,
    accounting for higher moments.

    Parameters
    ----------
    observed_sr : float
        Best observed Sharpe ratio.
    sr_std : float
        Standard deviation of Sharpe ratios across all trials.
    n_observations : int
        Number of return observations.
    n_trials : int
        Number of strategies/configurations tested.
    skewness : float
        Skewness of returns.
    kurtosis : float
        Excess kurtosis of returns.

    Returns
    -------
    float
        Deflated Sharpe ratio (probability-adjusted).
    """
    ...

def discrete_signal(signal, step_size):
    """
    Discretize a continuous signal to the nearest step.

    Parameters
    ----------
    signal : float
        Continuous signal value.
    step_size : float
        Discretization step (e.g. 0.1 for 10% increments).

    Returns
    -------
    float
        Rounded signal.
    """
    ...

def estimate_ou_params(series, dt):
    """
    Estimate O-U process parameters from a time series via OLS.

    Parameters
    ----------
    series : numpy.ndarray
        Observed time series.
    dt : float
        Time step between observations.

    Returns
    -------
    tuple[float, float, float]
        (theta, mu, sigma) — mean-reversion speed, long-term mean, volatility.
    """
    ...

def hhi(weights):
    """
    Herfindahl-Hirschman Index measuring concentration.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight or share vector (should sum to 1).

    Returns
    -------
    float
        HHI value. Ranges from 1/n (perfectly diversified) to 1 (fully concentrated).
    """
    ...

def hhi_concentration(returns):
    """
    Concentration of positive and negative returns using HHI.

    Parameters
    ----------
    returns : numpy.ndarray
        Return series.

    Returns
    -------
    tuple[float, float]
        (positive_hhi, negative_hhi) — concentration in gains and losses.
    """
    ...

def hit_ratio(returns):
    """
    Fraction of positive returns (hit rate / win rate).

    Parameters
    ----------
    returns : numpy.ndarray
        Return series.

    Returns
    -------
    float
        Proportion of returns > 0.
    """
    ...

def holm_correction(p_values):
    """
    Holm-Bonferroni step-down correction for multiple testing.

    Less conservative than Bonferroni while still controlling the
    family-wise error rate.

    Parameters
    ----------
    p_values : numpy.ndarray
        Uncorrected p-values.

    Returns
    -------
    numpy.ndarray
        Corrected p-values.
    """
    ...

def implied_frequency(target_sr, precision, avg_win_loss_ratio):
    """
    Implied trading frequency needed to achieve a target Sharpe ratio.

    Parameters
    ----------
    target_sr : float
        Desired annualized Sharpe ratio.
    precision : float
        Hit rate.
    avg_win_loss_ratio : float
        Average win/loss ratio.

    Returns
    -------
    float
        Required trades per year.
    """
    ...

def implied_precision(target_sr, freq, avg_win_loss_ratio):
    """
    Implied precision needed to achieve a target Sharpe ratio.

    Parameters
    ----------
    target_sr : float
        Desired annualized Sharpe ratio.
    freq : float
        Average trades per year.
    avg_win_loss_ratio : float
        Average win/loss ratio.

    Returns
    -------
    float
        Required hit rate (precision).
    """
    ...

def otr_mesh(theta, mu, sigma, pt_range, sl_range, n_simulations, seed):
    """
    Compute an optimal trading rule (OTR) mesh over profit-take and stop-loss ranges.

    Simulates O-U processes and evaluates average returns for each
    (profit_take, stop_loss) combination.

    Parameters
    ----------
    theta : float
        Mean-reversion speed.
    mu : float
        Long-term mean.
    sigma : float
        Volatility.
    pt_range : numpy.ndarray
        Profit-take thresholds to evaluate.
    sl_range : numpy.ndarray
        Stop-loss thresholds to evaluate.
    n_simulations : int
        Monte Carlo simulations per grid point.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        2-D array of shape (len(pt_range), len(sl_range)) with average returns.
    """
    ...

def power_bet_size(prob, num_classes, exponent):
    """
    Power-law bet sizing from prediction probability.

    Parameters
    ----------
    prob : float
        Predicted probability.
    num_classes : int
        Number of classes.
    exponent : float
        Power exponent (higher = more aggressive sizing).

    Returns
    -------
    float
        Bet size in [-1, 1].
    """
    ...

def probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_observations, skewness, kurtosis):
    """
    Probabilistic Sharpe Ratio (PSR) — probability that the true SR exceeds a benchmark.

    Accounts for skewness and kurtosis of the return distribution (AFML Ch. 14).

    Parameters
    ----------
    observed_sr : float
        Observed (sample) Sharpe ratio.
    benchmark_sr : float
        Benchmark Sharpe ratio to compare against.
    n_observations : int
        Number of return observations.
    skewness : float
        Skewness of returns.
    kurtosis : float
        Excess kurtosis of returns.

    Returns
    -------
    float
        Probability (0 to 1) that the true SR exceeds the benchmark.
    """
    ...

def probability_of_backtest_overfitting(returns_matrix, num_partitions, seed):
    """
    Probability of Backtest Overfitting (PBO) via CSCV (AFML Ch. 14).

    Estimates the probability that the best in-sample strategy will
    underperform out-of-sample using combinatorial data splitting.

    Parameters
    ----------
    returns_matrix : numpy.ndarray
        Matrix of shape (n_periods, n_strategies) with period returns.
    num_partitions : int
        Number of partitions for CSCV (must be even).
    seed : int
        Random seed for partition shuffling.

    Returns
    -------
    float
        PBO estimate in [0, 1]. Higher means more overfitting risk.
    """
    ...

def sharpe_mesh(returns_grid):
    """
    Compute Sharpe ratios over a grid of return series.

    Parameters
    ----------
    returns_grid : numpy.ndarray
        2-D array where each column is a return series.

    Returns
    -------
    numpy.ndarray
        Sharpe ratio for each column, reshaped to match the input grid.
    """
    ...

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252.0):
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic return series.
    risk_free_rate : float, default 0.0
        Risk-free rate per period.
    periods_per_year : float, default 252.0
        Annualization factor (252 for daily, 12 for monthly).

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    ...

def sigmoid_bet_size(prob, num_classes):
    """
    Sigmoid bet sizing from prediction probability (AFML Ch. 10).

    Maps a probability to a position size using a sigmoid function.

    Parameters
    ----------
    prob : float
        Predicted probability of the positive class.
    num_classes : int
        Number of discrete classes (usually 2).

    Returns
    -------
    float
        Bet size in [-1, 1].
    """
    ...

def simulate_ou(theta, mu, sigma, x0, dt, n_steps, seed):
    """
    Simulate an Ornstein-Uhlenbeck mean-reverting process.

    ``dX = theta * (mu - X) * dt + sigma * dW``

    Parameters
    ----------
    theta : float
        Mean-reversion speed.
    mu : float
        Long-term mean.
    sigma : float
        Volatility.
    x0 : float
        Initial value.
    dt : float
        Time step.
    n_steps : int
        Number of simulation steps.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Simulated path of length n_steps + 1 (including x0).
    """
    ...

def sr_from_precision(precision, freq, avg_win_loss_ratio):
    """
    Compute Sharpe ratio from precision, frequency, and win/loss ratio.

    Parameters
    ----------
    precision : float
        Hit rate (probability of a winning trade).
    freq : float
        Average number of trades per year.
    avg_win_loss_ratio : float
        Average win size divided by average loss size.

    Returns
    -------
    float
        Expected annualized Sharpe ratio.
    """
    ...

def strategy_failure_probability(estimated_precision, n_observations, break_even_precision):
    """
    Probability that a strategy will fail (precision below break-even).

    Uses a binomial test to estimate the probability that the true
    precision is below the break-even threshold.

    Parameters
    ----------
    estimated_precision : float
        Observed hit rate.
    n_observations : int
        Number of trades observed.
    break_even_precision : float
        Minimum precision needed to break even.

    Returns
    -------
    float
        Failure probability.
    """
    ...

def turnover(positions):
    """
    Portfolio turnover from a position series.

    Parameters
    ----------
    positions : numpy.ndarray
        Position sizes over time.

    Returns
    -------
    float
        Average absolute change in position per period.
    """
    ...
