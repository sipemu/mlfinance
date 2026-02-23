"""
Chapter 10: Bet Sizing
======================
AFML Ch. 10 — Sizing positions based on prediction confidence.

Demonstrates:
- Sigmoid bet sizing
- Power-law bet sizing
- Signal discretization
- Average active signals
"""

import numpy as np
import polars as pl
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 10: Bet Sizing")
print("=" * 60)

# --- Sigmoid Bet Sizing ---
print(f"\n--- Sigmoid Bet Sizing ---")
print(f"  {'Probability':>12} {'Bet Size (2-class)':>20} {'Bet Size (3-class)':>20}")
for prob in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    size_2 = pymlfinance.backtesting.sigmoid_bet_size(prob, num_classes=2)
    size_3 = pymlfinance.backtesting.sigmoid_bet_size(prob, num_classes=3)
    print(f"  {prob:>12.2f} {size_2:>20.4f} {size_3:>20.4f}")

# --- Power Bet Sizing ---
print(f"\n--- Power Bet Sizing (2-class) ---")
print(f"  {'Probability':>12} {'exp=0.5':>10} {'exp=1.0':>10} {'exp=2.0':>10} {'exp=3.0':>10}")
for prob in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    sizes = [pymlfinance.backtesting.power_bet_size(prob, 2, exp) for exp in [0.5, 1.0, 2.0, 3.0]]
    print(f"  {prob:>12.2f} {sizes[0]:>10.4f} {sizes[1]:>10.4f} {sizes[2]:>10.4f} {sizes[3]:>10.4f}")

# --- Signal Discretization ---
print(f"\n--- Signal Discretization ---")
print(f"  {'Continuous':>12} {'step=0.1':>10} {'step=0.2':>10} {'step=0.25':>10}")
for signal in [-0.73, -0.35, 0.0, 0.17, 0.42, 0.88]:
    d1 = pymlfinance.backtesting.discrete_signal(signal, 0.1)
    d2 = pymlfinance.backtesting.discrete_signal(signal, 0.2)
    d3 = pymlfinance.backtesting.discrete_signal(signal, 0.25)
    print(f"  {signal:>12.2f} {d1:>10.2f} {d2:>10.2f} {d3:>10.2f}")

# --- Average Active Signals ---
print(f"\n--- Average Active Signals ---")
n_bars = 100
# Create overlapping signals
signals = [
    (0, 30, 0.5),
    (10, 40, -0.3),
    (25, 60, 0.7),
    (50, 80, -0.4),
    (70, 99, 0.6),
]
avg_signals = pymlfinance.backtesting.avg_active_signals(signals, n_bars)
print(f"  {len(signals)} overlapping signals across {n_bars} bars")
print(f"  Mean active signal: {np.mean(avg_signals):.4f}")
print(f"  Max active signal:  {np.max(avg_signals):.4f}")
print(f"  Min active signal:  {np.min(avg_signals):.4f}")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars

probs = np.linspace(0.0, 1.0, 11)
df = pl.DataFrame({"probability": probs})
result = df.with_columns(
    pl.col("probability").ml.sigmoid_bet_size(num_classes=2).alias("sigmoid_bet"),
    pl.col("probability").ml.power_bet_size(num_classes=2, exponent=2.0).alias("power_bet"),
    pl.col("probability").ml.discrete_signal(step_size=0.1).alias("discrete"),
)
print(result)

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Compare sigmoid vs power bet sizing curves with different exponents")
print("2. Create a simulation: feed random predictions through bet sizing and track P&L")
print("3. Explore how discretization step size affects portfolio turnover")
