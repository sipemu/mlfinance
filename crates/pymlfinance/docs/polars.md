# Polars Integration

pymlfinance includes a native Polars expression plugin, giving you access to financial ML functions directly in Polars lazy/eager pipelines with zero-copy performance.

## Installation

```bash
pip install pymlfinance[polars]
```

Or if building from source, ensure `polars` is installed in your environment.

## Quick Start

```python
import polars as pl
import pymlfinance.polars  # registers the .ml namespace

df = pl.DataFrame({"price": [100.0, 102.0, 101.0, 105.0, 103.0]})

result = df.with_columns(
    pl.col("price").ml.ewma(span=3).alias("ewma"),
    pl.col("price").ml.log_returns().alias("log_ret"),
    pl.col("price").ml.simple_returns().alias("ret"),
)
```

## Tier 1: Expression Namespace (`.ml`)

All single-column functions are available through the `.ml` expression namespace. Just `import pymlfinance.polars` to register it.

### Core

| Method | Description |
|--------|-------------|
| `.ml.ewma(span)` | Exponential weighted moving average |
| `.ml.ewma_std(span)` | EWMA standard deviation |
| `.ml.cumsum()` | Cumulative sum |
| `.ml.log_returns()` | Logarithmic returns |
| `.ml.simple_returns()` | Simple (arithmetic) returns |

### Sampling

| Method | Description |
|--------|-------------|
| `.ml.frac_diff_ffd(d, threshold)` | Fractional differentiation (fixed-width window) |
| `.ml.frac_diff_expanding(d, threshold)` | Fractional differentiation (expanding window) |
| `.ml.find_min_d(max_d, step_size, threshold)` | Find minimum d for stationarity (scalar) |

### Labeling

| Method | Description |
|--------|-------------|
| `.ml.daily_volatility(span)` | Daily volatility estimate |
| `.ml.trend_scanning_label_series(max_window)` | Trend scanning labels |

### Features: Structural Breaks

| Method | Description |
|--------|-------------|
| `.ml.adf_test(max_lags)` | Augmented Dickey-Fuller test (scalar) |
| `.ml.sadf(min_window, max_lags)` | Sequential ADF test |

### Features: Encoding

| Method | Description |
|--------|-------------|
| `.ml.binary_encode()` | Binary (0/1) encoding |
| `.ml.quantile_encode(n_bins)` | Quantile-based encoding |
| `.ml.sigma_encode(n_bands)` | Standard deviation band encoding |

### Features: Entropy

| Method | Description |
|--------|-------------|
| `.ml.lempel_ziv_complexity()` | Lempel-Ziv complexity (scalar) |
| `.ml.kontoyiannis_entropy(window)` | Kontoyiannis entropy estimator (scalar) |
| `.ml.shannon_entropy()` | Shannon entropy (scalar) |
| `.ml.plugin_entropy()` | Plugin entropy estimator (scalar) |

### Features: Microstructure

| Method | Description |
|--------|-------------|
| `.ml.tick_rule_classify()` | Tick rule trade classification |

### Backtesting: Statistics

| Method | Description |
|--------|-------------|
| `.ml.sharpe_ratio(risk_free_rate, periods_per_year)` | Annualized Sharpe ratio (scalar) |
| `.ml.hit_ratio()` | Win rate (scalar) |
| `.ml.hhi()` | Herfindahl-Hirschman Index (scalar) |
| `.ml.compute_drawdowns()` | Drawdown series |

### Backtesting: Bet Sizing

| Method | Description |
|--------|-------------|
| `.ml.sigmoid_bet_size(num_classes)` | Sigmoid-based bet sizing |
| `.ml.power_bet_size(num_classes, exponent)` | Power-law bet sizing |
| `.ml.discrete_signal(step_size)` | Signal discretization |

## Tier 2: Multi-Column Functions

Functions that require multiple columns are available as standalone functions:

```python
from pymlfinance.polars._lib import corwin_schultz_spread, parkinson_volatility

df = pl.DataFrame({
    "high": [102.0, 103.0, 104.0],
    "low":  [99.0, 100.0, 101.0],
})

result = df.with_columns(
    corwin_schultz_spread("high", "low").alias("spread"),
    parkinson_volatility("high", "low", window=2).alias("pvol"),
)
```

### Microstructure

| Function | Arguments | Description |
|----------|-----------|-------------|
| `amihud_lambda(returns, dollar_volumes)` | 2 columns | Amihud's lambda (scalar) |
| `amihud_lambda_rolling(returns, dollar_volumes, window)` | 2 columns | Rolling Amihud lambda |
| `kyle_lambda(returns, signed_volume)` | 2 columns | Kyle's lambda (scalar) |
| `roll_spread_rolling(prices, window)` | 1 column | Roll spread estimator |
| `corwin_schultz_spread(highs, lows)` | 2 columns | Corwin-Schultz spread |
| `vpin(volumes, prices, bucket_size, n_buckets)` | 2 columns | VPIN estimator |

### Volatility Estimators

| Function | Arguments | Description |
|----------|-----------|-------------|
| `parkinson_volatility(highs, lows, window)` | 2 columns | Parkinson volatility |
| `garman_klass_volatility(opens, highs, lows, closes, window)` | 4 columns | Garman-Klass volatility |
| `yang_zhang_volatility(opens, highs, lows, closes, window)` | 4 columns | Yang-Zhang volatility |

## NumPy vs Polars API

Both APIs call the same Rust functions. Choose based on your data pipeline:

| Aspect | NumPy API | Polars API |
|--------|-----------|------------|
| Import | `from pymlfinance import core` | `import pymlfinance.polars` |
| Input | `np.ndarray` | `pl.Expr` / column names |
| Execution | Eager | Lazy-compatible |
| Multi-column | Separate arrays | Column expressions |
| Best for | Research notebooks | Production pipelines |
