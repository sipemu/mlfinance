"""
Chapter 14: Backtest Statistics
================================
AFML Ch. 14 — Sharpe ratio adjustments and performance metrics.

Demonstrates:
- Sharpe ratio
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Drawdown analysis
- HHI concentration
- Hit ratio
"""

import numpy as np
import polars as pl
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 14: Backtest Statistics")
print("=" * 60)

# --- Generate synthetic strategy returns ---
n = 500
# Strategy with slight positive drift
returns = np.random.randn(n) * 0.01 + 0.0003
# Add some fat tails
returns[np.random.choice(n, 20, replace=False)] *= 3.0

print(f"\nGenerated {n} daily returns")
print(f"  Mean: {np.mean(returns):.6f}, Std: {np.std(returns):.6f}")

# --- Sharpe Ratio ---
sr = pymlfinance.backtesting.sharpe_ratio(returns)
sr_monthly = pymlfinance.backtesting.sharpe_ratio(returns, periods_per_year=12.0)
print(f"\n--- Sharpe Ratio ---")
print(f"  Annualized (daily):   {sr:.4f}")
print(f"  Annualized (monthly): {sr_monthly:.4f}")

# --- Probabilistic Sharpe Ratio ---
skew = pymlfinance.core.skewness(returns)
kurt = pymlfinance.core.kurtosis(returns)
print(f"\n--- Higher Moments ---")
print(f"  Skewness: {skew:.4f}")
print(f"  Excess Kurtosis: {kurt:.4f}")

psr = pymlfinance.backtesting.probabilistic_sharpe_ratio(
    observed_sr=sr,
    benchmark_sr=0.0,  # benchmark: zero SR
    n_observations=n,
    skewness=skew,
    kurtosis=kurt
)
print(f"\n--- Probabilistic Sharpe Ratio ---")
print(f"  PSR(SR > 0): {psr:.4f}")
print(f"  Interpretation: {psr:.0%} probability that true SR > 0")

# PSR with different benchmarks
for bench in [0.0, 0.5, 1.0, 1.5]:
    p = pymlfinance.backtesting.probabilistic_sharpe_ratio(
        sr, bench, n, skew, kurt
    )
    print(f"  PSR(SR > {bench}): {p:.4f}")

# --- Deflated Sharpe Ratio ---
# Simulate testing multiple strategies
n_trials = 20
trial_srs = [pymlfinance.backtesting.sharpe_ratio(
    np.random.randn(n) * 0.01 + 0.0002
) for _ in range(n_trials - 1)]
trial_srs.append(sr)  # include our strategy
sr_std = np.std(trial_srs)

dsr = pymlfinance.backtesting.deflated_sharpe_ratio(
    observed_sr=sr,
    sr_std=sr_std,
    n_observations=n,
    n_trials=n_trials,
    skewness=skew,
    kurtosis=kurt
)
print(f"\n--- Deflated Sharpe Ratio ---")
print(f"  After testing {n_trials} strategies:")
print(f"  DSR = {dsr:.4f}")
print(f"  (Accounts for multiple testing bias)")

# --- Drawdown Analysis ---
dd = pymlfinance.backtesting.compute_drawdowns(returns)
print(f"\n--- Drawdown Analysis ---")
print(f"  Max drawdown: {dd.max_drawdown:.4f} ({dd.max_drawdown:.2%})")
print(f"  Max drawdown duration: {dd.max_drawdown_duration} bars")
dd_series = np.array(dd.drawdown_series)
print(f"  Average drawdown: {np.mean(dd_series):.4f}")
tuw = np.array(dd.time_under_water)
print(f"  Avg time under water: {np.mean(tuw):.1f} bars")

# --- Hit Ratio ---
hr = pymlfinance.backtesting.hit_ratio(returns)
print(f"\n--- Hit Ratio ---")
print(f"  Win rate: {hr:.4f} ({hr:.1%})")

# --- HHI Concentration ---
hhi_val = pymlfinance.backtesting.hhi(np.abs(returns) / np.sum(np.abs(returns)))
pos_hhi, neg_hhi = pymlfinance.backtesting.hhi_concentration(returns)
print(f"\n--- HHI Concentration ---")
print(f"  HHI of |returns|: {hhi_val:.6f}")
print(f"  Positive return HHI: {pos_hhi:.6f}")
print(f"  Negative return HHI: {neg_hhi:.6f}")
print(f"  (Lower HHI = more diversified across time)")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars

df = pl.DataFrame({"returns": returns})
stats = df.select(
    pl.col("returns").ml.sharpe_ratio().alias("sharpe"),
    pl.col("returns").ml.hit_ratio().alias("hit_ratio"),
    pl.col("returns").ml.hhi().alias("hhi"),
)
print(stats)

dd_df = df.with_columns(
    pl.col("returns").ml.compute_drawdowns().alias("drawdown"),
)
print(f"  Drawdown series (first 5): {dd_df['drawdown'].head(5).to_list()}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Increase n_trials in DSR and observe how it penalizes the Sharpe ratio")
print("2. Compare PSR for strategies with different skewness/kurtosis")
print("3. Generate a strategy with high hit ratio but negative Sharpe (large losses)")
