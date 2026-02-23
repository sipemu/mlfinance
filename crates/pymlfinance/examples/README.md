# pymlfinance Examples

Python examples covering key chapters from *Advances in Financial Machine Learning* (AFML) by Marcos Lopez de Prado, implemented using the `pymlfinance` library.

Inspired by the [BlackArbsCEO AFML Exercises](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises).

## Setup

```bash
cd crates/pymlfinance
source .venv/bin/activate
pip install matplotlib jupyter
maturin develop --release
```

## Table of Contents

| Chapter | Topic | Notebook | Script | Polars API |
|---------|-------|----------|--------|------------|
| Ch 2 | Financial Data Structures | [notebook](notebooks/ch02_financial_data_structures.ipynb) | [script](scripts/ch02_financial_data_structures.py) | — |
| Ch 3 | Labeling | [notebook](notebooks/ch03_labeling.ipynb) | [script](scripts/ch03_labeling.py) | `.ml.daily_volatility()`, `.ml.trend_scanning_label_series()` |
| Ch 4 | Sample Weights | [notebook](notebooks/ch04_sample_weights.ipynb) | [script](scripts/ch04_sample_weights.py) | — |
| Ch 5 | Fractional Differentiation | [notebook](notebooks/ch05_fractional_differentiation.ipynb) | [script](scripts/ch05_fractional_differentiation.py) | `.ml.frac_diff_ffd()`, `.ml.find_min_d()`, `.ml.adf_test()` |
| Ch 6 | Ensemble Methods | [notebook](notebooks/ch06_ensemble_methods.ipynb) | [script](scripts/ch06_ensemble_methods.py) | — |
| Ch 7 | Cross-Validation | [notebook](notebooks/ch07_cross_validation.ipynb) | [script](scripts/ch07_cross_validation.py) | — |
| Ch 8 | Feature Importance | [notebook](notebooks/ch08_feature_importance.ipynb) | [script](scripts/ch08_feature_importance.py) | — |
| Ch 10 | Bet Sizing | [notebook](notebooks/ch10_bet_sizing.ipynb) | [script](scripts/ch10_bet_sizing.py) | `.ml.sigmoid_bet_size()`, `.ml.power_bet_size()` |
| Ch 11+12 | Backtesting Dangers | [notebook](notebooks/ch11_backtesting_dangers.ipynb) | [script](scripts/ch11_backtesting_dangers.py) | — |
| Ch 13 | Synthetic Data | [notebook](notebooks/ch13_synthetic_data.ipynb) | [script](scripts/ch13_synthetic_data.py) | — |
| Ch 14 | Backtest Statistics | [notebook](notebooks/ch14_backtest_statistics.ipynb) | [script](scripts/ch14_backtest_statistics.py) | `.ml.sharpe_ratio()`, `.ml.hit_ratio()`, `.ml.compute_drawdowns()` |
| Ch 17 | Structural Breaks | [notebook](notebooks/ch17_structural_breaks.ipynb) | [script](scripts/ch17_structural_breaks.py) | `.ml.adf_test()`, `.ml.sadf()` |
| Ch 18 | Entropy Features | [notebook](notebooks/ch18_entropy_features.ipynb) | [script](scripts/ch18_entropy_features.py) | `.ml.binary_encode()`, `.ml.shannon_entropy()`, `.ml.lempel_ziv_complexity()` |
| Ch 19+20 | Microstructure | [notebook](notebooks/ch19_microstructure.ipynb) | [script](scripts/ch19_microstructure.py) | `.ml.tick_rule_classify()`, `_lib.vpin()`, `_lib.kyle_lambda()` |

## Running

### Scripts

```bash
# Run a single script
python examples/scripts/ch02_financial_data_structures.py

# Run all scripts
for f in examples/scripts/ch*.py; do python "$f"; done
```

### Notebooks

```bash
# Launch Jupyter
jupyter notebook examples/notebooks/

# Execute all notebooks headless
jupyter nbconvert --to notebook --execute examples/notebooks/*.ipynb
```

## Design

- **All synthetic data** — no external files or API keys required
- **Both APIs** — each example shows the NumPy API and Polars expression API (where applicable)
- **Self-contained** — each notebook/script runs independently
- **Exercises** — suggested parameter variations at the end of each example
