"""
Chapter 19 & 20: Market Microstructure
========================================
AFML Ch. 19-20 — Liquidity, price impact, and informed trading metrics.

Demonstrates:
- Tick rule classification
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Amihud lambda (price impact)
- Kyle's lambda (order flow impact)
- Roll spread estimator
- Corwin-Schultz spread estimator
- Hasbrouck lambda
"""

import numpy as np
import polars as pl
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 19 & 20: Market Microstructure")
print("=" * 60)

# --- Generate synthetic trade data ---
n = 2000
# Price with microstructure noise (bid-ask bounce)
efficient_price = 100.0 + np.cumsum(np.random.randn(n) * 0.05)
spread = 0.10
trade_prices = efficient_price + np.random.choice([-spread/2, spread/2], n)
volumes = np.abs(np.random.exponential(100, n))
dollar_volumes = trade_prices * volumes

# OHLC data (for spread estimators)
n_bars = 200
bar_size = n // n_bars
highs = np.array([trade_prices[i*bar_size:(i+1)*bar_size].max() for i in range(n_bars)])
lows = np.array([trade_prices[i*bar_size:(i+1)*bar_size].min() for i in range(n_bars)])
opens = np.array([trade_prices[i*bar_size] for i in range(n_bars)])
closes = np.array([trade_prices[(i+1)*bar_size - 1] for i in range(n_bars)])

print(f"\nGenerated {n} trades, {n_bars} OHLC bars")
print(f"  Price range: {trade_prices.min():.2f} - {trade_prices.max():.2f}")
print(f"  True spread: {spread:.2f}")

# --- Tick Rule Classification ---
tick_signs = pymlfinance.features.tick_rule_classify(trade_prices)
print(f"\n--- Tick Rule Classification ---")
print(f"  Upticks (+1): {np.sum(tick_signs > 0)}")
print(f"  Downticks (-1): {np.sum(tick_signs < 0)}")
print(f"  No change (0): {np.sum(tick_signs == 0)}")
print(f"  Buy/sell imbalance: {np.mean(tick_signs):.4f}")

# --- VPIN ---
vpin_values = pymlfinance.features.vpin(
    volumes=volumes, prices=trade_prices,
    bucket_size=float(np.sum(volumes) / 100),  # ~100 buckets
    n_buckets=10
)
print(f"\n--- VPIN ---")
print(f"  VPIN series length: {len(vpin_values)}")
if len(vpin_values) > 0:
    print(f"  Mean VPIN: {np.mean(vpin_values):.4f}")
    print(f"  Max VPIN:  {np.max(vpin_values):.4f}")
    print(f"  Min VPIN:  {np.min(vpin_values):.4f}")

# --- Amihud Lambda ---
returns = pymlfinance.core.log_returns(trade_prices)
amihud = pymlfinance.features.amihud_lambda(returns, dollar_volumes[1:])
print(f"\n--- Amihud Lambda (Price Impact) ---")
print(f"  Amihud lambda: {amihud:.8f}")
print(f"  (Higher = less liquid / greater price impact)")

# Rolling Amihud
amihud_rolling = pymlfinance.features.amihud_lambda_rolling(returns, dollar_volumes[1:], window=100)
print(f"  Rolling Amihud (window=100): mean={np.nanmean(amihud_rolling):.8f}")

# --- Kyle Lambda ---
signed_vol = tick_signs[1:].astype(np.float64) * volumes[1:]
kyle = pymlfinance.features.kyle_lambda(returns, signed_vol)
print(f"\n--- Kyle Lambda ---")
print(f"  Kyle lambda: {kyle:.8f}")
print(f"  (Permanent price impact per unit signed volume)")

# --- Roll Spread ---
roll = pymlfinance.features.roll_spread(closes)
print(f"\n--- Roll Spread Estimator ---")
print(f"  Estimated spread: {roll:.4f}")
print(f"  True spread: {spread:.4f}")
print(f"  Error: {abs(roll - spread):.4f}")

# Rolling Roll
roll_rolling = pymlfinance.features.roll_spread_rolling(closes, window=20)
print(f"  Rolling mean: {np.nanmean(roll_rolling):.4f}")

# --- Corwin-Schultz Spread ---
cs_spread = pymlfinance.features.corwin_schultz_spread(highs, lows)
print(f"\n--- Corwin-Schultz Spread ---")
cs_valid = cs_spread[~np.isnan(cs_spread)]
if len(cs_valid) > 0:
    print(f"  Mean spread: {np.mean(cs_valid):.4f}")
    print(f"  Median spread: {np.median(cs_valid):.4f}")

# --- Hasbrouck Lambda ---
trade_signs = tick_signs[1:].astype(np.float64)
# Replace zeros with previous sign
for i in range(1, len(trade_signs)):
    if trade_signs[i] == 0:
        trade_signs[i] = trade_signs[i-1]
hasbrouck = pymlfinance.features.hasbrouck_lambda(returns, trade_signs, n_iterations=100, seed=42)
print(f"\n--- Hasbrouck Lambda (Gibbs Sampling) ---")
print(f"  Hasbrouck lambda: {hasbrouck:.8f}")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars
from pymlfinance.polars._lib import (
    amihud_lambda as pl_amihud, kyle_lambda as pl_kyle,
    corwin_schultz_spread as pl_cs, vpin as pl_vpin,
    parkinson_volatility as pl_parkinson,
    garman_klass_volatility as pl_gk,
    yang_zhang_volatility as pl_yz,
)

# Tick-level Polars
tick_df = pl.DataFrame({
    "price": trade_prices,
    "volume": volumes,
})
tick_result = tick_df.with_columns(
    pl.col("price").ml.tick_rule_classify().alias("tick_sign"),
)
print(f"  Tick signs (first 5): {tick_result['tick_sign'].head(5).to_list()}")

# Bar-level Polars
bar_returns = np.concatenate([[np.nan], pymlfinance.core.log_returns(closes)])
bar_dollar_vol = closes * volumes[:n_bars]
bar_df = pl.DataFrame({
    "open": opens, "high": highs, "low": lows, "close": closes,
    "returns": bar_returns,
    "dollar_volume": bar_dollar_vol,
})

# Multi-column functions
parkinson = bar_df.with_columns(
    pl_parkinson(pl.col("high"), pl.col("low"), window=20).alias("parkinson_vol"),
)
print(f"  Parkinson vol (first non-null): "
      f"{parkinson['parkinson_vol'].drop_nulls().head(3).to_list()}")

gk = bar_df.with_columns(
    pl_gk(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"), window=20).alias("gk_vol"),
)
print(f"  Garman-Klass vol (first non-null): "
      f"{gk['gk_vol'].drop_nulls().head(3).to_list()}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Increase the true spread and verify that Roll and CS estimators track it")
print("2. Add informed trading (directional volume) and observe VPIN increase")
print("3. Compare Amihud and Kyle lambdas under different liquidity regimes")
