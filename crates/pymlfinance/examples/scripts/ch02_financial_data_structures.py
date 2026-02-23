"""
Chapter 2: Financial Data Structures
=====================================
AFML Ch. 2 — Alternative bar types that handle irregular market activity
better than traditional time bars.

Demonstrates:
- TickBarAggregator: fixed number of ticks per bar
- VolumeBarAggregator: fixed volume per bar
- DollarBarAggregator: fixed dollar volume per bar
- TimeBarAggregator: fixed time interval
- CUSUM filter for event detection
"""

import numpy as np
import pymlfinance
from pymlfinance import TickData

np.random.seed(42)

# --- Generate synthetic tick data ---
# Simulate 10,000 ticks with varying activity (clustered volume)
n_ticks = 10_000
timestamps = np.cumsum(np.random.exponential(0.5, n_ticks))  # irregular arrivals
prices = 100.0 + np.cumsum(np.random.randn(n_ticks) * 0.01)  # random walk
# Volume clusters: some periods have 10x more activity
volume_regime = np.where(np.sin(np.arange(n_ticks) * 0.002) > 0.5, 10.0, 1.0)
volumes = np.abs(np.random.exponential(1.0, n_ticks)) * volume_regime

ticks = [TickData(float(timestamps[i]), float(prices[i]), float(volumes[i]))
         for i in range(n_ticks)]

print("=" * 60)
print("Chapter 2: Financial Data Structures")
print("=" * 60)
print(f"\nGenerated {n_ticks} synthetic ticks")
print(f"  Price range: {prices.min():.2f} - {prices.max():.2f}")
print(f"  Time span: {timestamps[-1]:.1f} seconds")
print(f"  Total volume: {volumes.sum():.0f}")

# --- Tick Bars ---
tick_agg = pymlfinance.data.TickBarAggregator(bar_size=50)
tick_bars = tick_agg.process_ticks(ticks)
print(f"\n--- Tick Bars (50 ticks/bar) ---")
print(f"  Number of bars: {len(tick_bars)}")
if tick_bars:
    print(f"  First bar: O={tick_bars[0].open:.2f} H={tick_bars[0].high:.2f} "
          f"L={tick_bars[0].low:.2f} C={tick_bars[0].close:.2f} V={tick_bars[0].volume:.1f}")

# --- Volume Bars ---
avg_vol = volumes.sum() / 200  # target ~200 bars
vol_agg = pymlfinance.data.VolumeBarAggregator(volume_threshold=avg_vol)
vol_bars = vol_agg.process_ticks(ticks)
print(f"\n--- Volume Bars (threshold={avg_vol:.0f}) ---")
print(f"  Number of bars: {len(vol_bars)}")

# --- Dollar Bars ---
dollar_volumes = prices * volumes
avg_dollar = dollar_volumes.sum() / 200
dollar_agg = pymlfinance.data.DollarBarAggregator(dollar_threshold=avg_dollar)
dollar_bars = dollar_agg.process_ticks(ticks)
print(f"\n--- Dollar Bars (threshold={avg_dollar:.0f}) ---")
print(f"  Number of bars: {len(dollar_bars)}")

# --- Time Bars ---
time_agg = pymlfinance.data.TimeBarAggregator(interval_seconds=25)
time_bars = time_agg.process_ticks(ticks)
print(f"\n--- Time Bars (25 sec interval) ---")
print(f"  Number of bars: {len(time_bars)}")

# --- Compare bar count variability ---
print(f"\n--- Bar Count Comparison ---")
print(f"  Time bars:   {len(time_bars):>4d} (fixed time, variable info)")
print(f"  Tick bars:   {len(tick_bars):>4d} (fixed ticks)")
print(f"  Volume bars: {len(vol_bars):>4d} (fixed volume)")
print(f"  Dollar bars: {len(dollar_bars):>4d} (fixed dollar volume)")

# --- CUSUM Filter ---
log_returns = pymlfinance.core.log_returns(prices)
threshold = np.std(log_returns) * 2.0
events = pymlfinance.data.cusum_filter(log_returns, threshold)
print(f"\n--- CUSUM Filter ---")
print(f"  Threshold: {threshold:.6f}")
print(f"  Events detected: {len(events)}")
if len(events) >= 3:
    print(f"  First 3 event indices: {events[:3]}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Try different tick bar sizes (10, 50, 200) and compare volatility per bar")
print("2. Increase volume clustering and observe how volume bars adapt")
print("3. Compare bar return distributions across bar types")
