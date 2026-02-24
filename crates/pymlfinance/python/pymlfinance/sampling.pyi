from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def average_uniqueness(events, num_bars):
    """
    Compute average uniqueness of each event (AFML Ch. 4).

    Uniqueness at bar j for event i is 1 / (number of concurrent events at j).
    Average uniqueness is the mean across all bars spanned by the event.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (entry_idx, exit_idx) pairs.
    num_bars : int
        Total number of bars.

    Returns
    -------
    numpy.ndarray
        Average uniqueness per event.
    """
    ...

def balanced_class_weights(labels):
    """
    Compute balanced class weights inversely proportional to class frequency.

    Parameters
    ----------
    labels : list[int]
        Label vector (e.g. [-1, 0, 1]).

    Returns
    -------
    dict[int, float]
        Mapping from label value to weight.
    """
    ...

def compare_bootstraps(ind_matrix, num_samples, num_trials, seed):
    """
    Monte Carlo comparison of sequential vs. standard bootstrap uniqueness.

    Parameters
    ----------
    ind_matrix : numpy.ndarray
        Indicator matrix (num_bars x n_events).
    num_samples : int
        Samples per trial.
    num_trials : int
        Number of Monte Carlo repetitions.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapComparison
        Average uniqueness for sequential and standard methods.
    """
    ...

def find_min_d(series, max_d, step_size, threshold):
    """
    Find the minimum fractional differentiation order that makes a series stationary.

    Performs a grid search over d values, applying FFD and testing stationarity
    with ADF at each step (AFML Ch. 5).

    Parameters
    ----------
    series : numpy.ndarray
        Input time series.
    max_d : float
        Maximum d to test.
    step_size : float
        Increment between d values.
    threshold : float
        Weight truncation threshold for FFD.

    Returns
    -------
    float
        Minimum d for stationarity (returns max_d if none found).
    """
    ...

def frac_diff_expanding(series, d, threshold):
    """
    Expanding-window fractional differentiation.

    Uses all available history at each point (no truncation), producing
    a more accurate but slower computation.

    Parameters
    ----------
    series : numpy.ndarray
        Input time series.
    d : float
        Differentiation order.
    threshold : float
        Minimum weight for inclusion.

    Returns
    -------
    numpy.ndarray
        Fractionally differenced series.
    """
    ...

def frac_diff_ffd(series, d, threshold):
    """
    Apply FFD (fixed-width window fractional differentiation) to a series.

    Produces a stationary series that retains memory, using a truncated
    weight kernel (AFML Ch. 5).

    Parameters
    ----------
    series : numpy.ndarray
        Input time series (e.g. log prices).
    d : float
        Differentiation order (typically 0.3-0.7).
    threshold : float
        Weight truncation threshold (e.g. 1e-4).

    Returns
    -------
    numpy.ndarray
        Fractionally differenced series.
    """
    ...

def get_indicator_matrix(events, num_bars):
    """
    Build an indicator matrix mapping events to bars.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (entry_idx, exit_idx) pairs.
    num_bars : int
        Total number of bars.

    Returns
    -------
    numpy.ndarray
        Binary matrix of shape (num_bars, n_events) where entry (t, i) = 1
        if event i is active at bar t.
    """
    ...

def get_weights(d, size):
    """
    Compute fractional differentiation weights (AFML Ch. 5).

    Returns the weight vector for a given differentiation order ``d``.
    Weights decay with lag; the series ``size`` controls truncation.

    Parameters
    ----------
    d : float
        Fractional differentiation order (0 < d < 1 for stationarity).
    size : int
        Number of weights to compute.

    Returns
    -------
    numpy.ndarray
        Weight vector of length ``size``.
    """
    ...

def get_weights_ffd(d, threshold):
    """
    Compute FFD (Fixed-width window Fractional Differentiation) weights.

    Truncates weights below a threshold to create a fixed-width kernel.

    Parameters
    ----------
    d : float
        Fractional differentiation order.
    threshold : float
        Minimum absolute weight to keep (e.g. 1e-4).

    Returns
    -------
    numpy.ndarray
        Truncated weight vector.
    """
    ...

def num_co_events(events, num_bars):
    """
    Count concurrent events at each bar (AFML Ch. 4).

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (entry_idx, exit_idx) pairs.
    num_bars : int
        Total number of bars in the series.

    Returns
    -------
    list[int]
        Number of active events at each bar index.
    """
    ...

def return_attribution_weights(events, returns, num_bars):
    """
    Compute return-attribution sample weights (AFML Ch. 4).

    Weights each event proportionally to its return contribution,
    adjusted for concurrency.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (entry_idx, exit_idx) pairs.
    returns : numpy.ndarray
        Per-bar return series.
    num_bars : int
        Total number of bars.

    Returns
    -------
    numpy.ndarray
        Sample weights (one per event).
    """
    ...

def seq_bootstrap(ind_matrix, num_samples, seed):
    """
    Sequential bootstrap with uniqueness-aware sampling (AFML Ch. 4).

    Draws samples with probability proportional to their average uniqueness,
    reducing redundancy from overlapping labels.

    Parameters
    ----------
    ind_matrix : numpy.ndarray
        Indicator matrix (num_bars x n_events) from ``get_indicator_matrix``.
    num_samples : int
        Number of samples to draw.
    seed : int
        Random seed.

    Returns
    -------
    list[int]
        Sampled event indices.
    """
    ...

def standard_bootstrap(num_observations, num_samples, seed):
    """
    Standard IID bootstrap sampling.

    Parameters
    ----------
    num_observations : int
        Total number of observations to sample from.
    num_samples : int
        Number of bootstrap samples to draw.
    seed : int
        Random seed.

    Returns
    -------
    list[int]
        Sampled indices (with replacement).
    """
    ...

def time_decay(weights, oldest_weight):
    """
    Apply time-decay to sample weights (AFML Ch. 4).

    Linearly decays weights from 1.0 (most recent) to ``oldest_weight``
    (least recent).

    Parameters
    ----------
    weights : numpy.ndarray
        Input weights (typically from return attribution).
    oldest_weight : float
        Weight for the oldest observation. Use 0 for full linear decay,
        1 for no decay.

    Returns
    -------
    numpy.ndarray
        Time-decayed weights.
    """
    ...
