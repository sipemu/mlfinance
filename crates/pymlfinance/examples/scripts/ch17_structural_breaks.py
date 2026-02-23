"""
Chapter 17: Structural Breaks
===============================
AFML Ch. 17 — Detecting regime changes and explosive behavior.

Demonstrates:
- ADF test for stationarity
- SADF (Supremum ADF) for bubble detection
- GSADF for multiple bubble detection
- Brown-Durbin-Evans CUSUM test
- Chu-Stinchcombe-White test
"""

import numpy as np
import polars as pl
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 17: Structural Breaks")
print("=" * 60)

# --- Generate synthetic data with a bubble ---
n = 300
# Normal regime + bubble + crash + recovery
normal1 = np.cumsum(np.random.randn(100) * 0.01)
bubble = np.cumsum(np.random.randn(50) * 0.01 + 0.02)  # explosive growth
crash = np.cumsum(np.random.randn(50) * 0.02 - 0.03)   # sharp decline
normal2 = np.cumsum(np.random.randn(100) * 0.01)
log_prices = np.concatenate([normal1, normal1[-1] + bubble,
                              normal1[-1] + bubble[-1] + crash,
                              normal1[-1] + bubble[-1] + crash[-1] + normal2])
log_prices = log_prices + 4.6  # ~100 price level

print(f"\nGenerated {len(log_prices)} log prices with embedded bubble (bars 100-150)")

# --- ADF Test ---
adf_stat, coeffs = pymlfinance.features.adf_test(log_prices, max_lags=1)
print(f"\n--- ADF Test (full series) ---")
print(f"  ADF statistic: {adf_stat:.4f}")
print(f"  (More negative = more likely stationary)")
print(f"  Critical values: -3.43 (1%), -2.86 (5%), -2.57 (10%)")
if adf_stat < -2.86:
    print(f"  Result: STATIONARY at 5% level")
else:
    print(f"  Result: NON-STATIONARY (unit root present)")

# --- SADF Test ---
min_window = 30
sadf_series = pymlfinance.features.sadf(log_prices, min_window=min_window, max_lags=1)
sadf_max = pymlfinance.features.sadf_stat(log_prices, min_window=min_window, max_lags=1)
print(f"\n--- SADF (Supremum ADF) ---")
print(f"  SADF statistic: {sadf_max:.4f}")
print(f"  Series length: {len(sadf_series)}")
print(f"  (Positive SADF = evidence of explosive behavior)")

# Find the bubble region in SADF
if len(sadf_series) > 0:
    peak_idx = np.argmax(sadf_series)
    print(f"  Peak SADF at index {peak_idx + min_window} (value: {sadf_series[peak_idx]:.4f})")
    # Show around the bubble region
    bubble_start = max(0, 100 - min_window)
    bubble_end = min(len(sadf_series), 150 - min_window)
    if bubble_end > bubble_start:
        bubble_sadfs = sadf_series[bubble_start:bubble_end]
        print(f"  Mean SADF in bubble region: {np.mean(bubble_sadfs):.4f}")
        print(f"  Max SADF in bubble region:  {np.max(bubble_sadfs):.4f}")

# --- GSADF Test ---
gsadf_series = pymlfinance.features.gsadf(log_prices, min_window=min_window, max_lags=1)
gsadf_max = pymlfinance.features.gsadf_stat(log_prices, min_window=min_window, max_lags=1)
print(f"\n--- GSADF (Generalized SADF) ---")
print(f"  GSADF statistic: {gsadf_max:.4f}")
print(f"  Series length: {len(gsadf_series)}")

# --- Brown-Durbin-Evans CUSUM Test ---
# Use first differences as residuals
residuals = np.diff(log_prices)
cusum, critical = pymlfinance.features.brown_durbin_evans(residuals)
print(f"\n--- Brown-Durbin-Evans CUSUM ---")
print(f"  Critical value (5%): {critical:.4f}")
print(f"  Max |CUSUM|: {np.max(np.abs(cusum)):.4f}")
exceedances = np.sum(np.abs(cusum) > critical)
print(f"  Bars exceeding critical: {exceedances}")

# --- Chu-Stinchcombe-White ---
csw = pymlfinance.features.chu_stinchcombe_white(log_prices, critical_value=1.96)
print(f"\n--- Chu-Stinchcombe-White ---")
print(f"  CSW series length: {len(csw)}")
print(f"  Max CSW statistic: {np.max(csw):.4f}")
print(f"  Bars exceeding 1.96: {np.sum(csw > 1.96)}")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars

df = pl.DataFrame({"log_price": log_prices})
adf_pl = df.select(pl.col("log_price").ml.adf_test(max_lags=1)).item()
print(f"  Polars ADF: {adf_pl:.4f}")

sadf_df = df.with_columns(
    pl.col("log_price").ml.sadf(min_window=min_window, max_lags=1).alias("sadf"),
)
print(f"  Polars SADF series length: {sadf_df['sadf'].drop_nulls().len()}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Vary the bubble intensity and see how SADF responds")
print("2. Add multiple bubbles and compare SADF vs GSADF detection power")
print("3. Try different min_window sizes and observe the tradeoff")
