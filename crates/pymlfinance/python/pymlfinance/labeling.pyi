from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class MetaLabeler:
    """
    Meta-labeling model for bet sizing (AFML Ch. 3).

    Wraps a primary model's predictions: generates meta-labels that
    indicate whether to act on the primary signal, and sizes bets
    based on prediction probability.

    Parameters
    ----------
    min_probability : float
        Minimum probability threshold for a positive meta-label.
    """
    def __init__(self, min_probability): ...
    def __repr__(self) -> str: ...
    def bet_size(self, /, probability):
        """
        Compute bet size from a predicted probability.

        Parameters
        ----------
        probability : float
            Predicted probability of the positive class.

        Returns
        -------
        float
            Bet size in [0, 1]. Returns 0 if below ``min_probability``.
        """
        ...
    def generate_labels(self, /, events, primary_predictions):
        """
        Generate binary meta-labels from events and primary predictions.

        Parameters
        ----------
        events : list[Event]
            Labeled events.
        primary_predictions : list[int]
            Primary model's directional predictions (+1 or -1).

        Returns
        -------
        numpy.ndarray[int32]
            Meta-labels (0 = do not trade, 1 = trade).
        """
        ...

def add_vertical_barrier(entry_indices, max_holding, series_len):
    """
    Add vertical barriers (maximum holding period) to entry indices.

    Parameters
    ----------
    entry_indices : list[int]
        Entry bar indices.
    max_holding : int
        Maximum holding period in bars.
    series_len : int
        Total length of the price series (for clamping).

    Returns
    -------
    list[int]
        Exit indices corresponding to each entry (clamped to series_len - 1).
    """
    ...

def daily_volatility(prices, timestamps, span):
    """
    Compute daily volatility as the EWMA standard deviation of returns.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series.
    timestamps : list[float]
        Unix timestamps corresponding to prices.
    span : int
        EWMA span for volatility estimation.

    Returns
    -------
    numpy.ndarray
        Daily volatility estimates.
    """
    ...

def drop_rare_labels(labels, min_pct):
    """
    Generate a boolean mask to drop labels below a minimum frequency.

    Parameters
    ----------
    labels : list[int]
        Label vector.
    min_pct : float
        Minimum fraction (0 to 1) a label must represent to be kept.

    Returns
    -------
    list[bool]
        Mask where True means the sample's label is frequent enough.
    """
    ...

def garman_klass_volatility(bars, window):
    """
    Garman-Klass volatility estimator using OHLC prices.

    Uses open, high, low, and close prices for a more efficient estimate
    than Parkinson (accounts for opening jumps).

    Parameters
    ----------
    bars : list[OhlcvBar]
        OHLCV bars.
    window : int
        Rolling window size.

    Returns
    -------
    numpy.ndarray
        Rolling Garman-Klass volatility estimates.
    """
    ...

def get_bins(events):
    """
    Convert events to directional labels (-1, 0, +1).

    Parameters
    ----------
    events : list[Event]
        Labeled events from ``get_events``.

    Returns
    -------
    numpy.ndarray[int32]
        Labels: +1 (upper touch), -1 (lower touch), 0 (vertical).
    """
    ...

def get_events(prices, entry_indices, config, daily_vols):
    """
    Generate triple-barrier labeled events from a price series (AFML Ch. 3).

    For each entry index, finds the first barrier touch (upper profit-take,
    lower stop-loss, or vertical max-holding) and records the event.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series.
    entry_indices : list[int]
        Indices at which to enter positions.
    config : TripleBarrierConfig
        Barrier configuration (upper, lower, max holding period).
    daily_vols : numpy.ndarray
        Daily volatility estimates (same length as prices), used to scale barriers.

    Returns
    -------
    list[Event]
        Labeled events with entry/exit indices, touch type, and return.
    """
    ...

def get_meta_bins(events, primary_predictions):
    """
    Generate meta-labels that correct a primary model's predictions (AFML Ch. 3).

    A meta-label is 1 if the primary prediction's direction matches the
    actual outcome, 0 otherwise.

    Parameters
    ----------
    events : list[Event]
        Labeled events from ``get_events``.
    primary_predictions : list[int]
        Primary model's directional predictions (+1 or -1).

    Returns
    -------
    numpy.ndarray[int32]
        Meta-labels (0 or 1).
    """
    ...

def parkinson_volatility(bars, window):
    """
    Parkinson volatility estimator using high-low range.

    More efficient than close-to-close volatility since it uses
    intra-bar price range information.

    Parameters
    ----------
    bars : list[OhlcvBar]
        OHLCV bars.
    window : int
        Rolling window size.

    Returns
    -------
    numpy.ndarray
        Rolling Parkinson volatility estimates.
    """
    ...

def trend_scanning_label_series(prices, max_window):
    """
    Trend scanning label series (+1, -1, 0) for an entire price series.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series.
    max_window : int
        Maximum look-ahead window.

    Returns
    -------
    numpy.ndarray[int32]
        Labels for each index where a label could be computed.
    """
    ...

def trend_scanning_labels(prices, t_events=None, max_window=20, min_window=None):
    """
    Trend scanning labels using t-statistic regression (AFML Ch. 3.5).

    For each event index, fits linear regressions over multiple forward-looking
    windows and selects the window with the highest |t-statistic|.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series.
    t_events : list[int], optional
        Indices at which to compute labels. If None, uses all indices.
    max_window : int, default 20
        Maximum look-ahead window.
    min_window : int, optional
        Minimum look-ahead window (default: 3).

    Returns
    -------
    list[TrendScanResult]
        Per-event results with t-stat, label, best window, and R-squared.
    """
    ...

def yang_zhang_volatility(bars, window):
    """
    Yang-Zhang volatility estimator.

    Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
    components for a minimum-variance, drift-independent estimator.

    Parameters
    ----------
    bars : list[OhlcvBar]
        OHLCV bars.
    window : int
        Rolling window size.

    Returns
    -------
    numpy.ndarray
        Rolling Yang-Zhang volatility estimates.
    """
    ...
