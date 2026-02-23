from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class DollarBarAggregator:
    """
    Aggregate ticks into bars when cumulative dollar volume reaches a threshold.

    Parameters
    ----------
    dollar_threshold : float
        Dollar volume threshold per bar.
    """
    def __init__(self, dollar_threshold): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.

        Parameters
        ----------
        tick : TickData
            A single market tick.

        Returns
        -------
        OhlcvBar or None
            A completed bar if the threshold was reached, otherwise None.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.

        Parameters
        ----------
        ticks : list[TickData]
            Sequence of market ticks.

        Returns
        -------
        list[OhlcvBar]
            All bars completed during the batch.
        """
        ...

class DollarImbalanceBarAggregator:
    """
    Dollar imbalance bars (DIB) — sample when signed dollar volume imbalance exceeds an EWMA threshold.

    Parameters
    ----------
    initial_expected : float
        Initial expected imbalance.
    ewma_span : float
        EWMA decay span for threshold adaptation.
    """
    def __init__(self, initial_expected, ewma_span): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

class DollarRunsBarAggregator:
    """
    Dollar runs bars — sample when dollar volume of the dominant direction exceeds an EWMA threshold.

    Parameters
    ----------
    initial_expected : float
        Initial expected run dollar volume.
    ewma_span : float
        EWMA decay span for threshold adaptation.
    """
    def __init__(self, initial_expected, ewma_span): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

class TickBarAggregator:
    """
    Aggregate ticks into bars with a fixed number of ticks per bar.

    Parameters
    ----------
    bar_size : int
        Number of ticks per bar.
    """
    def __init__(self, bar_size): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.

        Parameters
        ----------
        tick : TickData
            A single market tick.

        Returns
        -------
        OhlcvBar or None
            A completed bar if the threshold was reached, otherwise None.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.

        Parameters
        ----------
        ticks : list[TickData]
            Sequence of market ticks.

        Returns
        -------
        list[OhlcvBar]
            All bars completed during the batch.
        """
        ...

class TickImbalanceBarAggregator:
    """
    Tick imbalance bars (TIB) — sample when tick direction imbalance exceeds an EWMA threshold.

    Parameters
    ----------
    initial_expected : float
        Initial expected imbalance.
    ewma_span : float
        EWMA decay span for threshold adaptation.
    """
    def __init__(self, initial_expected, ewma_span): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

class TickRunsBarAggregator:
    """
    Tick runs bars — sample when the longest run of same-sign ticks exceeds an EWMA threshold.

    Parameters
    ----------
    initial_expected : float
        Initial expected run length.
    ewma_span : float
        EWMA decay span for threshold adaptation.
    """
    def __init__(self, initial_expected, ewma_span): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

class TimeBarAggregator:
    """
    Aggregate ticks into bars at fixed time intervals.

    Parameters
    ----------
    interval_seconds : int
        Bar duration in seconds.
    """
    def __init__(self, interval_seconds): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

class VolumeBarAggregator:
    """
    Aggregate ticks into bars when cumulative volume reaches a threshold.

    Parameters
    ----------
    volume_threshold : float
        Volume threshold per bar.
    """
    def __init__(self, volume_threshold): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.

        Parameters
        ----------
        tick : TickData
            A single market tick.

        Returns
        -------
        OhlcvBar or None
            A completed bar if the threshold was reached, otherwise None.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.

        Parameters
        ----------
        ticks : list[TickData]
            Sequence of market ticks.

        Returns
        -------
        list[OhlcvBar]
            All bars completed during the batch.
        """
        ...

class VolumeImbalanceBarAggregator:
    """
    Volume imbalance bars (VIB) — sample when signed volume imbalance exceeds an EWMA threshold.

    Parameters
    ----------
    initial_expected : float
        Initial expected imbalance.
    ewma_span : float
        EWMA decay span for threshold adaptation.
    """
    def __init__(self, initial_expected, ewma_span): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

class VolumeRunsBarAggregator:
    """
    Volume runs bars — sample when volume of the dominant direction exceeds an EWMA threshold.

    Parameters
    ----------
    initial_expected : float
        Initial expected run volume.
    ewma_span : float
        EWMA decay span for threshold adaptation.
    """
    def __init__(self, initial_expected, ewma_span): ...
    def __repr__(self) -> str: ...
    def process_tick(self, /, tick):
        """
        Process a single tick and return a completed bar, if any.
        """
        ...
    def process_ticks(self, /, ticks):
        """
        Process a batch of ticks and return all completed bars.
        """
        ...

def cusum_filter(values, threshold):
    """
    CUSUM event filter for detecting structural shifts (AFML Ch. 2).

    Detects indices where the cumulative sum of deviations from the
    running mean exceeds a symmetric threshold.

    Parameters
    ----------
    values : numpy.ndarray
        Input series (e.g. log returns or price differences).
    threshold : float
        Symmetric threshold for positive and negative CUSUM.

    Returns
    -------
    list[int]
        Indices where CUSUM events are detected.
    """
    ...

def etf_trick(prices, weights):
    """
    ETF trick for combining multiple product series into a single tradeable index.

    Parameters
    ----------
    prices : list[list[float]]
        Per-product price series (products x time steps).
    weights : list[list[float]]
        Per-product allocation weights (products x time steps).

    Returns
    -------
    numpy.ndarray
        Synthetic ETF price series.
    """
    ...

def linspace_sample(start, end, n):
    """
    Sample n evenly spaced indices in a range.

    Parameters
    ----------
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).
    n : int
        Number of samples.

    Returns
    -------
    list[int]
        Evenly spaced indices.
    """
    ...

def non_negative_rolled(prices, roll_dates):
    """
    Build a non-negative rolled price series by adjusting for roll gaps.

    Parameters
    ----------
    prices : numpy.ndarray
        Raw futures prices.
    roll_dates : list[int]
        Indices where contract rolls occur.

    Returns
    -------
    numpy.ndarray
        Adjusted non-negative price series.
    """
    ...

def pca_weights(cov_matrix, risk_target=None):
    """
    PCA-based portfolio weights from a covariance matrix.

    Allocates risk proportionally to principal components. Optionally
    targets a specific risk distribution.

    Parameters
    ----------
    cov_matrix : numpy.ndarray
        Covariance matrix (n x n).
    risk_target : float, optional
        Target risk fraction for the first component. If None, uses
        equal risk allocation across all components.

    Returns
    -------
    numpy.ndarray
        Portfolio weights (length n, sums to 1).
    """
    ...

def roll_gaps(prices, roll_dates):
    """
    Compute roll gaps for a single-future continuous series.

    Parameters
    ----------
    prices : numpy.ndarray
        Raw futures prices.
    roll_dates : list[int]
        Indices where contract rolls occur.

    Returns
    -------
    numpy.ndarray
        Cumulative roll gap adjustments.
    """
    ...

def uniform_sample(n, total, seed):
    """
    Sample n random indices from a range.

    Parameters
    ----------
    n : int
        Number of samples.
    total : int
        Upper bound of the range (exclusive).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[int]
        Randomly sampled indices (sorted, with replacement).
    """
    ...
