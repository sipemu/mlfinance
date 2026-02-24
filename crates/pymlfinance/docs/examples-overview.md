# Examples

Python examples covering key chapters from *Advances in Financial Machine Learning* (AFML) by Marcos Lopez de Prado, implemented using `pymlfinance`.

Inspired by the [BlackArbsCEO AFML Exercises](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises).

## Chapters

| Chapter | Topic | Polars Expressions |
|---------|-------|--------------------|
| [Ch 2](examples/ch02_financial_data_structures.ipynb) | Financial Data Structures | — |
| [Ch 3](examples/ch03_labeling.ipynb) | Labeling | `.ml.daily_volatility()`, `.ml.trend_scanning_label_series()` |
| [Ch 4](examples/ch04_sample_weights.ipynb) | Sample Weights | — |
| [Ch 5](examples/ch05_fractional_differentiation.ipynb) | Fractional Differentiation | `.ml.frac_diff_ffd()`, `.ml.find_min_d()`, `.ml.adf_test()` |
| [Ch 6](examples/ch06_ensemble_methods.ipynb) | Ensemble Methods | — |
| [Ch 7](examples/ch07_cross_validation.ipynb) | Cross-Validation | — |
| [Ch 8](examples/ch08_feature_importance.ipynb) | Feature Importance | — |
| [Ch 10](examples/ch10_bet_sizing.ipynb) | Bet Sizing | `.ml.sigmoid_bet_size()`, `.ml.power_bet_size()` |
| [Ch 11-12](examples/ch11_backtesting_dangers.ipynb) | Backtesting Dangers | — |
| [Ch 13](examples/ch13_synthetic_data.ipynb) | Synthetic Data | — |
| [Ch 14](examples/ch14_backtest_statistics.ipynb) | Backtest Statistics | `.ml.sharpe_ratio()`, `.ml.hit_ratio()`, `.ml.compute_drawdowns()` |
| [Ch 17](examples/ch17_structural_breaks.ipynb) | Structural Breaks | `.ml.adf_test()`, `.ml.sadf()` |
| [Ch 18](examples/ch18_entropy_features.ipynb) | Entropy Features | `.ml.binary_encode()`, `.ml.shannon_entropy()`, `.ml.lempel_ziv_complexity()` |
| [Ch 19+20](examples/ch19_microstructure.ipynb) | Microstructure | `.ml.tick_rule_classify()`, `_lib.vpin()`, `_lib.kyle_lambda()` |

## Design

- **All synthetic data** — no external files or API keys required
- **Both APIs** — each example shows the NumPy API and Polars expression API (where applicable)
- **Self-contained** — each notebook runs independently
- **Exercises** — suggested parameter variations at the end of each example
