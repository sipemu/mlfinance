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

## Quick Start

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

## Examples

Self-contained examples covering the main workflows (see `crates/mlfinance/examples/`):

```bash
cargo run --example basic_pipeline          # tick bars, CUSUM filter, triple-barrier labeling
cargo run --example portfolio_construction  # HRP, IVP, equal-weight allocation with Sharpe comparison
cargo run --example model_evaluation        # purged k-fold CV, bet sizing, backtest statistics
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

## License

MIT License — see [LICENSE](LICENSE).
