# mlfinance

[![CI](https://github.com/sipemu/mlfinance/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/mlfinance/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sipemu/mlfinance/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/mlfinance)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Rust implementation of concepts from *Advances in Financial Machine Learning* by Marcos López de Prado.

## Crates

| Crate | Description |
|---|---|
| `mlfinance-core` | Shared types, traits, matrix utilities |
| `mlfinance-parallel` | Rayon-based parallel helpers |
| `mlfinance-data` | Bar generation (tick, volume, dollar, information-driven) |
| `mlfinance-labeling` | Triple-barrier, meta-labeling, trend-scanning |
| `mlfinance-sampling` | Sequential bootstrapping, sample weights, purging |
| `mlfinance-features` | Fractional differentiation, structural breaks, entropy |
| `mlfinance-modeling` | Bet sizing, cross-validation, feature importance |
| `mlfinance-backtesting` | Backtesting and performance metrics |
| `mlfinance` | Facade re-exporting all sub-crates |
| `pymlfinance` | Python bindings via PyO3 (all modules) |

## Quick Start

### Rust

```rust
// Cargo.toml
[dependencies]
mlfinance = { git = "https://github.com/sipemu/mlfinance" }
```

```sh
cargo build
cargo test --workspace
cargo doc --workspace --no-deps --open
```

### Python

```sh
cd crates/pymlfinance
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

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

The Python package exposes 7 submodules (`core`, `data`, `labeling`, `sampling`, `features`, `modeling`, `backtesting`) wrapping ~140 functions with zero-copy NumPy conversion. Requires Python >= 3.9.

**[Python API Documentation](https://sipemu.github.io/mlfinance/)**

## Examples

Self-contained examples covering the main workflows (see `crates/mlfinance/examples/`):

```bash
cargo run --example basic_pipeline          # tick bars, CUSUM filter, triple-barrier labeling
cargo run --example portfolio_construction  # HRP, IVP, equal-weight allocation with Sharpe comparison
cargo run --example model_evaluation        # purged k-fold CV, bet sizing, backtest statistics
cargo run --example fracdiff                # FFD weights, expanding window, minimum-d search
cargo run --example volatility_estimators   # Parkinson, Garman-Klass, Yang-Zhang vs daily vol
cargo run --example sample_weights          # indicator matrix, uniqueness, sequential bootstrap, time decay
cargo run --example structural_breaks       # ADF, SADF, GSADF, Brown-Durbin-Evans, Chu-Stinchcombe-White
cargo run --example entropy                 # Shannon, plug-in, Lempel-Ziv, Kontoyiannis with 3 encodings
cargo run --example microstructure          # VPIN, Amihud/Kyle lambda, Roll & Corwin-Schultz spreads
cargo run --example denoising               # Marcenko-Pastur, RMT denoising, detoning, optimal portfolio
cargo run --example meta_labeling           # trend scanning, meta-labels, bet sizing pipeline
cargo run --example overfitting_detection   # PBO, CSCV, Bonferroni/Holm, PSR, deflated Sharpe
cargo run --example strategy_risk           # SR from precision, failure probability, O-U simulation
```

## Benchmarks

Criterion benchmarks across 5 crates (10 suites):

```bash
cargo bench --workspace                     # run all benchmarks

# Individual suites
cargo bench -p mlfinance-data               # bars_bench, cusum_bench
cargo bench -p mlfinance-labeling           # barriers_bench
cargo bench -p mlfinance-sampling           # fracdiff_bench, bootstrap_bench
cargo bench -p mlfinance-features           # entropy_bench, hrp_bench, sadf_bench
cargo bench -p mlfinance-backtesting        # bet_sizing_bench, statistics_bench
```

## Development

```sh
# Activate pre-commit hooks
git config core.hooksPath .githooks

# Run checks manually
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## MSRV

The minimum supported Rust version is **1.75**.

## Disclaimer

This software is provided for educational and research purposes only. It does not constitute financial advice. Use of this code for trading or investment decisions is entirely at your own risk. The authors accept no liability for any financial losses incurred.

## License

Rust crates: MIT License — see [LICENSE](LICENSE).
Python bindings (`pymlfinance`): BUSL-1.1.
