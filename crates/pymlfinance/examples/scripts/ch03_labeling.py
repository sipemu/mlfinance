"""
Chapter 3: Labeling
===================
AFML Ch. 3 — Triple-barrier method and trend scanning labels.

Demonstrates:
- Daily volatility estimation
- CUSUM event filter
- Triple-barrier labeling (get_events, get_bins)
- Trend scanning labels
- Meta-labeling
"""

import numpy as np
import polars as pl
import pymlfinance
from pymlfinance import TickData, TripleBarrierConfig

np.random.seed(42)

# --- Generate synthetic price series ---
n = 1000
# Mean-reverting + trending components
trend = np.linspace(0, 2, n)
noise = np.cumsum(np.random.randn(n) * 0.5)
mean_reversion = -0.05 * noise
prices = 100.0 * np.exp((trend + noise + np.cumsum(mean_reversion)) * 0.01)
timestamps = np.arange(n, dtype=np.float64) * 86400.0  # daily timestamps

print("=" * 60)
print("Chapter 3: Labeling")
print("=" * 60)
print(f"\nGenerated {n} daily prices, range: {prices.min():.2f} - {prices.max():.2f}")

# --- Daily Volatility ---
daily_vol = pymlfinance.labeling.daily_volatility(prices, timestamps.tolist(), span=50)
print(f"\n--- Daily Volatility (EWMA span=50) ---")
print(f"  Mean vol: {np.nanmean(daily_vol):.6f}")
print(f"  Min vol:  {np.nanmin(daily_vol):.6f}")
print(f"  Max vol:  {np.nanmax(daily_vol):.6f}")

# --- CUSUM Filter ---
log_ret = pymlfinance.core.log_returns(prices)
threshold = np.nanmean(daily_vol[1:]) * 1.5
entry_indices = pymlfinance.data.cusum_filter(log_ret, threshold)
print(f"\n--- CUSUM Event Filter ---")
print(f"  Threshold: {threshold:.6f}")
print(f"  Entry events: {len(entry_indices)}")

# --- Triple-Barrier Labeling ---
config = TripleBarrierConfig(
    upper_barrier=2.0,   # 2x daily vol
    lower_barrier=2.0,   # 2x daily vol
    max_holding_period=20  # 20 bars max hold
)
events = pymlfinance.labeling.get_events(prices, entry_indices, config, daily_vol)
labels = pymlfinance.labeling.get_bins(events)
print(f"\n--- Triple-Barrier Labels ---")
print(f"  Total events: {len(events)}")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    label_name = {-1: "Stop-loss", 0: "Vertical", 1: "Profit-take"}
    print(f"  {label_name.get(int(u), str(u)):>12s} ({int(u):+d}): {c}")
if events:
    print(f"  First event: entry={events[0].entry_idx}, exit={events[0].exit_idx}, "
          f"type={events[0].touch_type}, ret={events[0].return_value:.4f}")

# --- Trend Scanning Labels ---
trend_labels = pymlfinance.labeling.trend_scanning_label_series(prices, max_window=20)
print(f"\n--- Trend Scanning Labels (max_window=20) ---")
print(f"  Labels computed: {len(trend_labels)}")
t_unique, t_counts = np.unique(trend_labels, return_counts=True)
for u, c in zip(t_unique, t_counts):
    print(f"  Label {int(u):+d}: {c}")

# --- Meta-Labeling ---
meta_labeler = pymlfinance.labeling.MetaLabeler(min_probability=0.5)
# Simulate primary model predictions (random +-1)
primary_preds = [1 if np.random.random() > 0.5 else -1 for _ in range(len(events))]
meta_labels = meta_labeler.generate_labels(events, primary_preds)
print(f"\n--- Meta-Labeling ---")
print(f"  Meta-labels: {len(meta_labels)}")
print(f"  Trade signals: {np.sum(meta_labels == 1)}")
print(f"  No-trade signals: {np.sum(meta_labels == 0)}")

# Test bet sizing
bet = meta_labeler.bet_size(0.7)
print(f"  Bet size for p=0.7: {bet:.4f}")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars  # registers .ml namespace

df = pl.DataFrame({"price": prices})
vol_df = df.with_columns(
    pl.col("price").ml.daily_volatility(span=50).alias("daily_vol"),
)
print(f"  Daily vol DataFrame shape: {vol_df.shape}")
print(vol_df.head(5))

# trend_scanning returns fewer rows than input, so select separately
trend_result = df.select(
    pl.col("price").ml.trend_scanning_label_series(max_window=20).alias("trend_label"),
)
print(f"  Trend labels shape: {trend_result.shape}")
print(trend_result.head(5))

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Vary the triple-barrier widths (1x, 2x, 3x vol) and compare label distributions")
print("2. Use asymmetric barriers (wider upper, tighter lower) for trend-following")
print("3. Compare trend scanning with different max_window values (10, 20, 50)")
