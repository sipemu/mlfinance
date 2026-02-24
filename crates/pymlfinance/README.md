# pymlfinance

[![PyPI](https://img.shields.io/pypi/v/pymlfinance)](https://pypi.org/project/pymlfinance/)
[![BUSL-1.1 licensed](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](https://github.com/sipemu/mlfinance/blob/main/LICENSE)
![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)

Python bindings for [mlfinance](https://github.com/sipemu/mlfinance) — a Rust implementation of concepts from *Advances in Financial Machine Learning* by Marcos L&oacute;pez de Prado.

## Installation

```sh
pip install pymlfinance
```

With Polars plugin support:

```sh
pip install pymlfinance[polars]
```

## Quick Start

```python
import numpy as np
import pymlfinance as ml

# EWMA smoothing
prices = np.array([100.0, 101.5, 99.8, 102.3, 101.0])
smoothed = ml.core.ewma(prices, 3)

# CUSUM event filter
events = ml.data.cusum_filter(prices, 1.5)

# Triple-barrier labeling
from pymlfinance import TripleBarrierConfig
config = TripleBarrierConfig(upper_barrier=0.02, lower_barrier=0.02, max_holding_period=10)

# Fractional differentiation
stationary = ml.sampling.frac_diff_ffd(prices, 0.5, 1e-4)

# HRP portfolio allocation
returns = np.random.randn(100, 5) * 0.02
weights = ml.features.hrp_weights(returns)

# Sharpe ratio
ret = np.random.randn(252) * 0.01
sr = ml.backtesting.sharpe_ratio(ret)
```

## Modules

| Module | Description |
|---|---|
| `pymlfinance.core` | EWMA, cumsum, returns, matrix utilities |
| `pymlfinance.data` | Bar generation (tick, volume, dollar, information-driven), CUSUM filter |
| `pymlfinance.labeling` | Triple-barrier, meta-labeling, trend-scanning |
| `pymlfinance.sampling` | Sequential bootstrapping, sample weights, fractional differentiation |
| `pymlfinance.features` | Structural breaks, entropy, microstructure, HRP, denoising |
| `pymlfinance.modeling` | Bet sizing, cross-validation, feature importance |
| `pymlfinance.backtesting` | Backtest statistics, overfitting detection, strategy risk |

## Polars Plugin

```python
import polars as pl
import pymlfinance  # registers .ml namespace

df = pl.DataFrame({"price": [100.0, 101.5, 99.8, 102.3, 101.0]})
df.with_columns(pl.col("price").ml.ewma(span=3).alias("smoothed"))
```

## Documentation

- [Python API Documentation](https://sipemu.github.io/mlfinance/)
- [GitHub Repository](https://github.com/sipemu/mlfinance)

## Disclaimer

This software is provided for educational and research purposes only. It does not constitute financial advice. Use of this code for trading or investment decisions is entirely at your own risk. The authors accept no liability for any financial losses incurred.

## License

BUSL-1.1 — see [LICENSE](https://github.com/sipemu/mlfinance/blob/main/LICENSE).
