"""
Chapter 6: Ensemble Methods
============================
AFML Ch. 6 — Bagging accuracy and ensemble effects.
Also covers HRP (Ch. 16) as a practical ensemble allocation method.

Demonstrates:
- Theoretical bagging accuracy curve
- HRP portfolio weights
- IVP (Inverse Variance) weights comparison
- Allocation comparison (HRP vs CLA vs IVP)
"""

import numpy as np
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 6: Ensemble Methods")
print("=" * 60)

# --- Bagging Accuracy ---
print(f"\n--- Theoretical Bagging Accuracy ---")
print(f"  Individual accuracy -> Ensemble accuracy (N classifiers)")
for p in [0.45, 0.50, 0.51, 0.55, 0.60, 0.70]:
    accs = []
    for n in [3, 11, 51, 101, 501]:
        acc = pymlfinance.modeling.bagging_accuracy(n, p)
        accs.append(f"N={n}: {acc:.4f}")
    print(f"  p={p:.2f} -> {', '.join(accs)}")

print(f"\n  Key insight: For p>0.5, ensemble accuracy -> 1.0 as N increases")
print(f"  For p<0.5, ensemble accuracy -> 0.0 (worse than individual!)")
print(f"  For p=0.5, ensemble accuracy stays at 0.5 regardless of N")

# --- HRP Portfolio Weights ---
print(f"\n--- Hierarchical Risk Parity (HRP) ---")
n_assets = 5
n_periods = 252
# Simulate correlated asset returns
factors = np.random.randn(n_periods, 2)
betas = np.random.randn(2, n_assets) * 0.3
returns = factors @ betas + np.random.randn(n_periods, n_assets) * 0.01

hrp_w = pymlfinance.features.hrp_weights(returns)
print(f"  {n_assets} assets, {n_periods} periods")
print(f"  HRP weights: [{', '.join(f'{w:.4f}' for w in hrp_w)}]")
print(f"  Sum: {np.sum(hrp_w):.4f}")

# --- IVP Weights ---
cov = pymlfinance.core.covariance_matrix(returns)
ivp_w = pymlfinance.features.inverse_variance_weights(cov)
print(f"\n--- Inverse Variance Portfolio (IVP) ---")
print(f"  IVP weights: [{', '.join(f'{w:.4f}' for w in ivp_w)}]")
print(f"  Sum: {np.sum(ivp_w):.4f}")

# --- Compare HRP vs IVP ---
print(f"\n--- Weight Comparison ---")
print(f"  {'Asset':<8} {'HRP':>8} {'IVP':>8} {'Diff':>8}")
for i in range(n_assets):
    diff = hrp_w[i] - ivp_w[i]
    print(f"  {f'Asset {i}':<8} {hrp_w[i]:>8.4f} {ivp_w[i]:>8.4f} {diff:>+8.4f}")

# --- Monte Carlo Allocation Comparison ---
print(f"\n--- Allocation Comparison (Monte Carlo) ---")
comparison = pymlfinance.features.compare_allocations(returns, n_simulations=50, seed=42)
print(f"  HRP  — Sharpe: {comparison.hrp_sharpe:.4f}, Variance: {comparison.hrp_variance:.6f}")
print(f"  CLA  — Sharpe: {comparison.cla_sharpe:.4f}, Variance: {comparison.cla_variance:.6f}")
print(f"  IVP  — Sharpe: {comparison.ivp_sharpe:.4f}, Variance: {comparison.ivp_variance:.6f}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Increase the number of assets and observe how HRP scales vs IVP")
print("2. Vary individual classifier accuracy p from 0.45 to 0.60 and plot the ensemble curve")
print("3. Add a highly correlated asset pair and compare how HRP and IVP handle it")
