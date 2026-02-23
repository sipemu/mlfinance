# Getting Started

## Installation

pymlfinance is built from Rust source using [maturin](https://www.maturin.rs/).

### Prerequisites

- Python 3.9+
- Rust toolchain (install via [rustup](https://rustup.rs/))
- maturin: `pip install maturin`

### Build from source

```bash
cd crates/pymlfinance
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy
maturin develop --release

# Optional: install Polars for the expression plugin
pip install polars
```

The `--release` flag enables Rust optimizations — strongly recommended for numerical workloads.

## Module Overview

pymlfinance is organized into 7 submodules matching the mlfinance Rust workspace:

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `core` | Math and statistics | `ewma`, `log_returns`, `correlation_matrix`, `power_iteration_eig` |
| `data` | Bar aggregation and sampling | `TickBarAggregator`, `VolumeBarAggregator`, `cusum_filter`, `etf_trick` |
| `labeling` | Event labeling | `get_events`, `get_bins`, `daily_volatility`, `trend_scanning_labels` |
| `sampling` | Differentiation and bootstrap | `frac_diff_ffd`, `seq_bootstrap`, `average_uniqueness` |
| `features` | Feature engineering | `sadf`, `lempel_ziv_complexity`, `vpin`, `hrp_weights`, `kmeans` |
| `modeling` | Model validation | `PurgedKFold`, `cv_score`, `mean_decrease_impurity` |
| `backtesting` | Performance analysis | `sharpe_ratio`, `deflated_sharpe_ratio`, `probability_of_backtest_overfitting` |

Shared data types (`TickData`, `OhlcvBar`, `Event`, etc.) are available at the root level:

```python
import pymlfinance as ml

tick = ml.TickData(timestamp=1000, price=100.0, volume=50.0)
```

## First Steps

### 1. Build bars from tick data

```python
import pymlfinance as ml

# Create a tick bar aggregator (100 ticks per bar)
agg = ml.data.TickBarAggregator(bar_size=100)

# Feed ticks one at a time
for ts, price, vol in tick_stream:
    tick = ml.TickData(timestamp=ts, price=price, volume=vol)
    bars = agg.process_tick(tick)
    for bar in bars:
        print(f"Bar: O={bar.open} H={bar.high} L={bar.low} C={bar.close}")
```

### 2. Label events with the triple-barrier method

```python
import numpy as np
import pymlfinance as ml

prices = np.array([...])  # your close prices
vol = ml.labeling.daily_volatility(prices, span=50)

config = ml.TripleBarrierConfig(
    profit_taking=2.0,
    stop_loss=2.0,
    max_holding=20,
)

# Entry indices (e.g., from CUSUM filter)
entries = ml.data.cusum_filter(prices, threshold=vol.mean())
events = ml.labeling.get_events(prices, entries, config, volatility=vol)
labels = ml.labeling.get_bins(events, prices)
```

### 3. Fractional differentiation

```python
import numpy as np
import pymlfinance as ml

prices = np.array([...])

# Find minimum d for stationarity
min_d = ml.sampling.find_min_d(prices, p_value_threshold=0.05, max_d=1.0)
print(f"Minimum d for stationarity: {min_d:.3f}")

# Apply FFD
stationary = ml.sampling.frac_diff_ffd(prices, d=min_d, threshold=1e-4)
```

### 4. Portfolio construction with HRP

```python
import numpy as np
import pymlfinance as ml

# Returns matrix: rows=observations, columns=assets
returns = np.random.randn(252, 10)
cov = ml.core.covariance_matrix(returns)
corr = ml.core.correlation_matrix(returns)

weights = ml.features.hrp_weights(cov, corr)
print(f"HRP weights: {weights}")
```

## NumPy Integration

All functions accept and return NumPy arrays. Data is exchanged via zero-copy buffers where possible:

```python
import numpy as np
import pymlfinance as ml

x = np.random.randn(1000)
result = ml.core.ewma(x, span=20)  # returns np.ndarray
assert isinstance(result, np.ndarray)
```

## Polars Integration

If you work with Polars DataFrames, pymlfinance includes a native expression plugin:

```python
import polars as pl
import pymlfinance.polars  # registers the .ml namespace

df = pl.DataFrame({"price": [100.0, 102.0, 101.0, 105.0, 103.0]})
result = df.with_columns(
    pl.col("price").ml.ewma(span=3).alias("ewma"),
    pl.col("price").ml.log_returns().alias("log_ret"),
)
```

See the [Polars Integration](polars.md) guide for the full API reference.

## Getting Help

Every function has a NumPy-style docstring:

```python
help(ml.core.ewma)
help(ml.TripleBarrierConfig)
```
