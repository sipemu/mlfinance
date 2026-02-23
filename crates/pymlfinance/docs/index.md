# pymlfinance

**Python bindings for the mlfinance AFML toolkit** — a high-performance Rust implementation of the methods from *Advances in Financial Machine Learning* by Marcos López de Prado.

## Overview

pymlfinance provides ~140 functions and 20+ classes across 7 modules, covering:

- **Core** — Statistical primitives: EWMA, returns, correlation/covariance matrices, eigendecomposition
- **Data** — 10 bar aggregation methods (tick, volume, dollar, imbalance, runs) plus sampling and ETF trick
- **Labeling** — Triple-barrier method, meta-labeling, volatility estimators, trend scanning
- **Sampling** — Fractional differentiation, sequential bootstrap, concurrency, sample weights
- **Features** — Structural breaks (SADF/GSADF), entropy, microstructure, RMT denoising, portfolio allocation (HRP/CLA/IVP), clustering (ONC), codependence measures
- **Modeling** — Purged K-fold CV, feature importance (MDI/MDA/SFI), hyperparameter search, scoring
- **Backtesting** — Performance statistics (Sharpe, PSR, DSR), overfitting detection (PBO/CSCV), strategy risk, bet sizing, synthetic data generation

All computation happens in Rust with zero-copy NumPy array interchange via PyO3.

## Quick Example

```python
import numpy as np
import pymlfinance as ml

# Generate log returns
prices = np.array([100.0, 101.5, 99.8, 102.3, 101.0, 103.5])
returns = ml.core.log_returns(prices)

# Compute daily volatility with EWMA
vol = ml.core.ewma_std(returns, span=3)

# Fractional differentiation to preserve memory while achieving stationarity
min_d = ml.sampling.find_min_d(prices, p_value_threshold=0.05, max_d=1.0)
stationary = ml.sampling.frac_diff_ffd(prices, d=min_d, threshold=1e-4)

# Sharpe ratio
sr = ml.backtesting.sharpe_ratio(returns, periods_per_year=252)
```

## Installation

```bash
# From source (requires Rust toolchain)
cd crates/pymlfinance
pip install maturin
maturin develop --release
```

## License

BUSL-1.1
