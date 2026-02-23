"""
Chapter 4: Sample Weights
=========================
AFML Ch. 4 — Handling overlapping labels with uniqueness-aware sampling.

Demonstrates:
- Concurrent events counting
- Average uniqueness of labels
- Indicator matrix construction
- Sequential bootstrap vs standard bootstrap
- Return attribution weights
- Time decay
"""

import numpy as np
import pymlfinance

np.random.seed(42)

# --- Generate synthetic overlapping events ---
n_bars = 500
n_events = 100

# Create events with varying overlap
entries = sorted(np.random.choice(n_bars - 20, n_events, replace=False))
durations = np.random.randint(5, 30, n_events)
events = [(int(e), min(int(e + d), n_bars - 1)) for e, d in zip(entries, durations)]

print("=" * 60)
print("Chapter 4: Sample Weights")
print("=" * 60)
print(f"\nGenerated {n_events} events across {n_bars} bars")
print(f"  Average duration: {np.mean(durations):.1f} bars")

# --- Concurrent Events ---
co_events = pymlfinance.sampling.num_co_events(events, n_bars)
print(f"\n--- Concurrent Events ---")
print(f"  Max concurrent: {max(co_events)}")
print(f"  Mean concurrent: {np.mean(co_events):.2f}")
print(f"  Bars with no events: {sum(1 for c in co_events if c == 0)}")

# --- Average Uniqueness ---
uniqueness = pymlfinance.sampling.average_uniqueness(events, n_bars)
print(f"\n--- Average Uniqueness ---")
print(f"  Mean uniqueness: {np.mean(uniqueness):.4f}")
print(f"  Min uniqueness:  {np.min(uniqueness):.4f}")
print(f"  Max uniqueness:  {np.max(uniqueness):.4f}")
print(f"  (1.0 = fully unique, lower = more overlap)")

# --- Indicator Matrix ---
ind_matrix = pymlfinance.sampling.get_indicator_matrix(events, n_bars)
print(f"\n--- Indicator Matrix ---")
print(f"  Shape: {ind_matrix.shape}  (events x bars)")
print(f"  Non-zero entries: {np.sum(ind_matrix > 0)}")
print(f"  Sparsity: {1 - np.sum(ind_matrix > 0) / ind_matrix.size:.2%}")

# --- Sequential Bootstrap ---
seq_samples = pymlfinance.sampling.seq_bootstrap(ind_matrix, num_samples=n_events, seed=42)
std_samples = pymlfinance.sampling.standard_bootstrap(n_events, n_events, seed=42)
print(f"\n--- Bootstrap Comparison ---")
print(f"  Sequential bootstrap unique samples: {len(set(seq_samples))}/{n_events}")
print(f"  Standard bootstrap unique samples:   {len(set(std_samples))}/{n_events}")

# Monte Carlo comparison
comparison = pymlfinance.sampling.compare_bootstraps(ind_matrix, n_events, num_trials=50, seed=42)
print(f"  Seq. avg uniqueness: {comparison.seq_uniqueness:.4f}")
print(f"  Std. avg uniqueness: {comparison.std_uniqueness:.4f}")

# --- Return Attribution Weights ---
# returns must be per-event (same length as events list)
event_returns = np.random.randn(n_events) * 0.01  # per-event returns
attr_weights = pymlfinance.sampling.return_attribution_weights(events, event_returns, n_bars)
print(f"\n--- Return Attribution Weights ---")
print(f"  Mean weight: {np.mean(attr_weights):.6f}")
print(f"  Std weight:  {np.std(attr_weights):.6f}")
print(f"  Min weight:  {np.min(attr_weights):.6f}")
print(f"  Max weight:  {np.max(attr_weights):.6f}")

# --- Time Decay ---
decayed_full = pymlfinance.sampling.time_decay(attr_weights, oldest_weight=0.0)
decayed_half = pymlfinance.sampling.time_decay(attr_weights, oldest_weight=0.5)
decayed_none = pymlfinance.sampling.time_decay(attr_weights, oldest_weight=1.0)
print(f"\n--- Time Decay ---")
print(f"  Full decay (oldest=0.0): first={decayed_full[0]:.4f}, last={decayed_full[-1]:.4f}")
print(f"  Half decay (oldest=0.5): first={decayed_half[0]:.4f}, last={decayed_half[-1]:.4f}")
print(f"  No decay   (oldest=1.0): first={decayed_none[0]:.4f}, last={decayed_none[-1]:.4f}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Increase event overlap (longer durations) and observe uniqueness drop")
print("2. Compare sequential vs standard bootstrap with different numbers of samples")
print("3. Try different time decay curves and see how weight distributions shift")
