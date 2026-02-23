# Data

Bar aggregation (10 bar types), CUSUM filtering, sampling, ETF trick, and PCA weights.

## Bar Aggregators

All bar aggregators share the same streaming interface:

```python
import pymlfinance as ml

agg = ml.data.TickBarAggregator(bar_size=100)
tick = ml.TickData(timestamp=1000, price=100.0, volume=50.0)
bars = agg.process_tick(tick)  # returns list[OhlcvBar]
```

Available aggregators:

- **Standard bars**: `TickBarAggregator`, `VolumeBarAggregator`, `DollarBarAggregator`, `TimeBarAggregator`
- **Imbalance bars**: `TickImbalanceBarAggregator`, `VolumeImbalanceBarAggregator`, `DollarImbalanceBarAggregator`
- **Runs bars**: `TickRunsBarAggregator`, `VolumeRunsBarAggregator`, `DollarRunsBarAggregator`

::: pymlfinance.data
