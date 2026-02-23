from __future__ import annotations
import numpy as np

from . import core as core
from . import data as data
from . import labeling as labeling
from . import sampling as sampling
from . import features as features
from . import modeling as modeling
from . import backtesting as backtesting

class TickData:
    """
    A single market tick with timestamp, price, and volume.

    Parameters
    ----------
    timestamp : float
        Unix timestamp in seconds (fractional seconds supported).
    price : float
        Trade price.
    volume : float
        Trade volume.

    Examples
    --------
    >>> tick = TickData(1609459200.0, 100.5, 10.0)
    >>> tick.price
    100.5
    """
    def __init__(self, timestamp, price, volume): ...
    def __repr__(self) -> str: ...
    price: ...
    timestamp: ...
    volume: ...

class OhlcvBar:
    """
    An OHLCV bar (Open, High, Low, Close, Volume) with VWAP.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the bar open.
    open : float
        Opening price.
    high : float
        Highest price during the bar.
    low : float
        Lowest price during the bar.
    close : float
        Closing price.
    volume : float
        Total volume traded.
    vwap : float, optional
        Volume-weighted average price (default 0.0).

    Attributes
    ----------
    mid_price() : float
        (high + low) / 2
    typical_price() : float
        (high + low + close) / 3
    dollar_volume() : float
        vwap * volume
    """
    def __init__(self, timestamp, open, high, low, close, volume, vwap=None): ...
    def __repr__(self) -> str: ...
    close: ...
    def dollar_volume(self, /):
        """
        Return the dollar volume: vwap * volume.
        """
        ...
    high: ...
    low: ...
    def mid_price(self, /):
        """
        Return the mid price: (high + low) / 2.
        """
        ...
    open: ...
    timestamp: ...
    def typical_price(self, /):
        """
        Return the typical price: (high + low + close) / 3.
        """
        ...
    volume: ...
    vwap: ...

class TripleBarrierConfig:
    """
    Configuration for the triple-barrier labeling method.

    Defines the upper profit-taking barrier, lower stop-loss barrier,
    and maximum holding period for event labeling (AFML Ch. 3).

    Parameters
    ----------
    upper_barrier : float, optional
        Profit-taking threshold as a fraction of daily volatility.
    lower_barrier : float, optional
        Stop-loss threshold as a fraction of daily volatility.
    max_holding_period : int, optional
        Maximum number of bars before forced exit (vertical barrier).

    Examples
    --------
    >>> config = TripleBarrierConfig(upper_barrier=2.0, lower_barrier=2.0, max_holding_period=10)
    """
    def __init__(self, upper_barrier=None, lower_barrier=None, max_holding_period=None): ...
    def __repr__(self) -> str: ...
    lower_barrier: ...
    max_holding_period: ...
    upper_barrier: ...

class Event:
    """
    A labeled event from the triple-barrier method.

    Attributes
    ----------
    entry_idx : int
        Index of the entry bar.
    exit_idx : int
        Index of the exit bar.
    touch_type : str
        Which barrier was touched first: ``"upper"``, ``"lower"``, or ``"vertical"``.
    return_value : float
        Return from entry to exit.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    entry_idx: ...
    exit_idx: ...
    return_value: ...
    touch_type: ...

class TrendScanResult:
    """
    Result of trend scanning label detection.

    Attributes
    ----------
    t_stat : float
        t-statistic of the best linear fit.
    label : int
        Trend direction: +1 (up), -1 (down), or 0 (no trend).
    best_window : int
        Look-ahead window that produced the highest |t-stat|.
    r_squared : float
        R-squared of the best linear fit.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    best_window: ...
    label: ...
    r_squared: ...
    t_stat: ...

class DrawdownResult:
    """
    Result of drawdown analysis on a return series.

    Attributes
    ----------
    max_drawdown : float
        Maximum peak-to-trough decline.
    max_drawdown_duration : int
        Longest drawdown duration in bars.
    drawdown_series : list[float]
        Per-bar drawdown values.
    time_under_water : list[int]
        Per-bar count of consecutive bars in drawdown.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    drawdown_series: ...
    max_drawdown: ...
    max_drawdown_duration: ...
    time_under_water: ...

class CscvResult:
    """
    Result of Combinatorially Symmetric Cross-Validation (CSCV).

    Attributes
    ----------
    pbo : float
        Probability of Backtest Overfitting.
    rank_logits : list[float]
        Log-odds of each strategy's rank degradation.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    pbo: ...
    rank_logits: ...

class KMeansResult:
    """
    Result of K-means clustering.

    Attributes
    ----------
    labels : list[int]
        Cluster assignment for each data point.
    n_iterations : int
        Number of iterations until convergence.
    centroids : list[list[float]]
        Cluster centroids (k x n_features).
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    centroids: ...
    labels: ...
    n_iterations: ...

class OncResult:
    """
    Result of Optimal Number of Clusters (ONC) analysis.

    Attributes
    ----------
    labels : list[int]
        Cluster assignment for each feature/variable.
    silhouette : float
        Silhouette score of the optimal clustering.
    n_clusters : int
        Optimal number of clusters found.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    labels: ...
    n_clusters: ...
    silhouette: ...

class AllocationComparison:
    """
    Comparison of HRP, CLA, and IVP portfolio allocations.

    Attributes
    ----------
    hrp_sharpe : float
        Out-of-sample Sharpe ratio for HRP.
    cla_sharpe : float
        Out-of-sample Sharpe ratio for CLA (min-variance).
    ivp_sharpe : float
        Out-of-sample Sharpe ratio for Inverse Variance.
    hrp_variance : float
        Out-of-sample portfolio variance for HRP.
    cla_variance : float
        Out-of-sample portfolio variance for CLA.
    ivp_variance : float
        Out-of-sample portfolio variance for IVP.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    cla_sharpe: ...
    cla_variance: ...
    hrp_sharpe: ...
    hrp_variance: ...
    ivp_sharpe: ...
    ivp_variance: ...

class BootstrapComparison:
    """
    Comparison of sequential vs. standard bootstrap uniqueness.

    Attributes
    ----------
    seq_uniqueness : float
        Average uniqueness from sequential bootstrap.
    std_uniqueness : float
        Average uniqueness from standard (IID) bootstrap.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    seq_uniqueness: ...
    std_uniqueness: ...

class FoldIndices:
    """
    Train/test split indices for a single cross-validation fold.

    Attributes
    ----------
    train : list[int]
        Indices of training samples.
    test : list[int]
        Indices of test samples.
    """
    def __init__(self, ): ...
    def __repr__(self) -> str: ...
    test: ...
    train: ...
