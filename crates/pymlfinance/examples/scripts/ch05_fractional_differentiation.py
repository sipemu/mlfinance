"""
Chapter 5: Fractional Differentiation
======================================
AFML Ch. 5 — Achieving stationarity while preserving memory.

Demonstrates:
- FFD weights computation
- Fractional differentiation (FFD and expanding window)
- Finding minimum d for stationarity
- ADF test for stationarity verification
- Correlation preservation analysis
"""

import numpy as np
import polars as pl
import pymlfinance

np.random.seed(42)

# --- Generate synthetic non-stationary price series ---
n = 500
# Trending series with mean-reverting noise
trend = np.cumsum(np.random.randn(n) * 0.02 + 0.001)  # upward drift
prices = 100.0 * np.exp(trend)
log_prices = np.log(prices)

print("=" * 60)
print("Chapter 5: Fractional Differentiation")
print("=" * 60)
print(f"\nGenerated {n} log prices")
print(f"  Start: {log_prices[0]:.4f}, End: {log_prices[-1]:.4f}")

# --- FFD Weights ---
print(f"\n--- FFD Weights ---")
for d in [0.3, 0.5, 0.7, 1.0]:
    weights = pymlfinance.sampling.get_weights_ffd(d, threshold=1e-4)
    print(f"  d={d:.1f}: {len(weights)} weights, sum={np.sum(weights):.4f}, "
          f"first={weights[0]:.4f}, last={weights[-1]:.6f}")

# --- Fractional Differentiation at Various d ---
print(f"\n--- FFD at Various d Values ---")
d_values = [0.2, 0.4, 0.6, 0.8, 1.0]
for d in d_values:
    ffd = pymlfinance.sampling.frac_diff_ffd(log_prices, d=d, threshold=1e-4)
    corr = np.corrcoef(log_prices[:len(ffd)], ffd)[0, 1] if len(ffd) > 0 else 0.0
    adf_stat, _ = pymlfinance.features.adf_test(ffd, max_lags=1)
    print(f"  d={d:.1f}: len={len(ffd)}, corr_with_original={corr:.4f}, ADF={adf_stat:.4f}")

# --- Expanding Window vs FFD ---
print(f"\n--- Expanding Window vs FFD (d=0.5) ---")
ffd_result = pymlfinance.sampling.frac_diff_ffd(log_prices, d=0.5, threshold=1e-4)
exp_result = pymlfinance.sampling.frac_diff_expanding(log_prices, d=0.5, threshold=1e-4)
print(f"  FFD length:       {len(ffd_result)}")
print(f"  Expanding length: {len(exp_result)}")
if len(ffd_result) > 0 and len(exp_result) > 0:
    min_len = min(len(ffd_result), len(exp_result))
    diff = np.abs(ffd_result[:min_len] - exp_result[:min_len])
    print(f"  Mean absolute difference: {np.mean(diff):.6f}")

# --- Find Minimum d for Stationarity ---
min_d = pymlfinance.sampling.find_min_d(log_prices, max_d=1.0, step_size=0.1, threshold=1e-4)
print(f"\n--- Minimum d for Stationarity ---")
print(f"  min_d = {min_d:.2f}")

# Verify
ffd_min = pymlfinance.sampling.frac_diff_ffd(log_prices, d=min_d, threshold=1e-4)
if len(ffd_min) > 0:
    adf_stat, _ = pymlfinance.features.adf_test(ffd_min, max_lags=1)
    corr = np.corrcoef(log_prices[:len(ffd_min)], ffd_min)[0, 1]
    print(f"  ADF statistic at d={min_d:.2f}: {adf_stat:.4f}")
    print(f"  Correlation with original: {corr:.4f}")
    print(f"  (Compare: integer differencing d=1.0 destroys all memory)")

# --- Weight Vectors ---
print(f"\n--- Weight Vectors ---")
for d in [0.3, 0.5, 0.7]:
    w = pymlfinance.sampling.get_weights(d, size=10)
    print(f"  d={d:.1f}: weights = [{', '.join(f'{x:.4f}' for x in w[:5])}...]")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars

df = pl.DataFrame({"log_price": log_prices})
result = df.with_columns(
    pl.col("log_price").ml.frac_diff_ffd(d=0.5, threshold=1e-4).alias("ffd_0.5"),
    pl.col("log_price").ml.frac_diff_expanding(d=0.5, threshold=1e-4).alias("exp_0.5"),
)
print(f"  DataFrame shape: {result.shape}")
print(result.head(5))

# Find min d via Polars
min_d_pl = df.select(
    pl.col("log_price").ml.find_min_d(max_d=1.0, step_size=0.1, threshold=1e-4)
).item()
print(f"  Polars find_min_d: {min_d_pl:.2f}")

# ADF test via Polars
adf_pl = df.select(
    pl.col("log_price").ml.adf_test(max_lags=1)
).item()
print(f"  Polars ADF on raw log prices: {adf_pl:.4f}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Plot correlation vs d and ADF statistic vs d to find the sweet spot")
print("2. Try different thresholds (1e-3, 1e-4, 1e-5) and compare FFD output length")
print("3. Generate a stationary series (e.g., returns) and verify min_d ≈ 0")
