"""
Chapter 11 & 12: Backtesting Dangers & CSCV
=============================================
AFML Ch. 11-12 — Detecting overfitting in backtests.

Demonstrates:
- Probability of Backtest Overfitting (PBO)
- Combinatorially Symmetric Cross-Validation (CSCV)
- Bonferroni and Holm corrections for multiple testing
"""

import numpy as np
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 11 & 12: Backtesting Dangers")
print("=" * 60)

# --- Generate synthetic strategy returns ---
n_periods = 500
n_strategies = 10

# Mostly noise strategies, with 1-2 having slight edge
returns_matrix = np.random.randn(n_periods, n_strategies) * 0.01
# Give strategy 0 a slight edge
returns_matrix[:, 0] += 0.0005
# Give strategy 1 a larger edge but more volatile
returns_matrix[:, 1] += 0.001
returns_matrix[:, 1] *= 2.0

print(f"\nGenerated {n_strategies} strategy returns over {n_periods} periods")
for i in range(n_strategies):
    sr = pymlfinance.backtesting.sharpe_ratio(returns_matrix[:, i])
    print(f"  Strategy {i}: Sharpe = {sr:.4f}")

# --- PBO ---
pbo = pymlfinance.backtesting.probability_of_backtest_overfitting(
    returns_matrix, num_partitions=10, seed=42
)
print(f"\n--- Probability of Backtest Overfitting ---")
print(f"  PBO = {pbo:.4f}")
print(f"  Interpretation: {pbo:.0%} chance that the best IS strategy")
print(f"  underperforms OOS")

# --- CSCV ---
cscv_result = pymlfinance.backtesting.cscv(returns_matrix, num_groups=8)
print(f"\n--- CSCV Analysis ---")
print(f"  PBO (via CSCV): {cscv_result.pbo:.4f}")
print(f"  Rank logits: [{', '.join(f'{x:.3f}' for x in cscv_result.rank_logits[:5])}...]")

# --- Multiple Testing Corrections ---
print(f"\n--- Multiple Testing Corrections ---")
# Simulate p-values from testing multiple strategies
# Some are "significant" by chance
p_values = np.random.uniform(0, 1, n_strategies)
p_values[0] = 0.01  # one genuinely significant
p_values[1] = 0.04  # borderline

bonferroni = pymlfinance.backtesting.bonferroni_correction(p_values)
holm = pymlfinance.backtesting.holm_correction(p_values)

print(f"  {'Strategy':>10} {'Raw p':>10} {'Bonferroni':>12} {'Holm':>10} {'Sig (5%)':>10}")
for i in range(n_strategies):
    sig = "Yes" if holm[i] < 0.05 else "No"
    print(f"  {f'Strat {i}':>10} {p_values[i]:>10.4f} {bonferroni[i]:>12.4f} {holm[i]:>10.4f} {sig:>10}")

# --- Overfitting Simulation ---
print(f"\n--- Overfitting Simulation ---")
print(f"  Testing with increasing number of strategies:")
for n_strats in [2, 5, 10, 20, 50]:
    # All noise strategies
    noise_returns = np.random.randn(n_periods, n_strats) * 0.01
    if n_strats >= 4:
        pbo_val = pymlfinance.backtesting.probability_of_backtest_overfitting(
            noise_returns, num_partitions=min(n_strats, 10), seed=42
        )
        best_sr = max(pymlfinance.backtesting.sharpe_ratio(noise_returns[:, i])
                      for i in range(n_strats))
        print(f"  {n_strats:>3d} strategies: best SR = {best_sr:.4f}, PBO = {pbo_val:.4f}")
    else:
        best_sr = max(pymlfinance.backtesting.sharpe_ratio(noise_returns[:, i])
                      for i in range(n_strats))
        print(f"  {n_strats:>3d} strategies: best SR = {best_sr:.4f}, PBO = N/A (need >=4)")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Add strategies with genuine alpha and see if PBO decreases")
print("2. Vary the number of CSCV partitions and observe stability")
print("3. Compare Bonferroni (conservative) vs Holm (less conservative) power")
