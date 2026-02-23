# Types

Shared data types used across all pymlfinance modules.

| Type | Description |
|------|-------------|
| `TickData` | A single market tick with timestamp, price, and volume |
| `OhlcvBar` | An OHLCV bar with VWAP |
| `TripleBarrierConfig` | Configuration for the triple-barrier labeling method |
| `Event` | A labeled event from the triple-barrier method |
| `TrendScanResult` | Result of trend scanning label detection |
| `DrawdownResult` | Result of drawdown analysis |
| `CscvResult` | Result of Combinatorially Symmetric Cross-Validation |
| `KMeansResult` | Result of K-means clustering |
| `OncResult` | Result of Optimal Number of Clusters analysis |
| `AllocationComparison` | Comparison of HRP, CLA, and IVP allocations |
| `BootstrapComparison` | Comparison of sequential vs. standard bootstrap uniqueness |
| `FoldIndices` | Train/test split indices for a cross-validation fold |

All types are available at the root level:

```python
import pymlfinance as ml

tick = ml.TickData(timestamp=1000, price=100.0, volume=50.0)
config = ml.TripleBarrierConfig(profit_taking=2.0, stop_loss=2.0, max_holding=20)
```

::: pymlfinance
    options:
      members:
        - TickData
        - OhlcvBar
        - TripleBarrierConfig
        - Event
        - TrendScanResult
        - DrawdownResult
        - CscvResult
        - KMeansResult
        - OncResult
        - AllocationComparison
        - BootstrapComparison
        - FoldIndices
