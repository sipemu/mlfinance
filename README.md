# mlfinance

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

## Development

```sh
# Activate pre-commit hooks
git config core.hooksPath .githooks

# Run checks manually
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## MSRV

The minimum supported Rust version is **1.75**.

## License

MIT
