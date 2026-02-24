"""
Chapter 18: Entropy Features
==============================
AFML Ch. 18 — Information-theoretic features for market microstructure.

Demonstrates:
- Binary, quantile, and sigma encoding
- Shannon entropy
- Plugin entropy estimator
- Kontoyiannis entropy estimator
- Lempel-Ziv complexity
"""

import numpy as np
import polars as pl
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 18: Entropy Features")
print("=" * 60)

# --- Generate synthetic series ---
n = 500
# Mixture: trending periods + random periods
trend_period = np.cumsum(np.ones(100) * 0.01 + np.random.randn(100) * 0.002)
random_period = np.cumsum(np.random.randn(150) * 0.01)
mean_rev = np.cumsum(-0.1 * np.random.randn(100).cumsum() * 0.01 + np.random.randn(100) * 0.005)
random_period2 = np.cumsum(np.random.randn(150) * 0.01)
series = np.concatenate([trend_period, random_period, mean_rev, random_period2])

print(f"\nGenerated {len(series)} observations")
print(f"  Regime 1 (0-100): trending")
print(f"  Regime 2 (100-250): random")
print(f"  Regime 3 (250-350): mean-reverting")
print(f"  Regime 4 (350-500): random")

# --- Encoding ---
print(f"\n--- Binary Encoding ---")
binary = pymlfinance.features.binary_encode(series)
print(f"  Length: {len(binary)}")
print(f"  True (above median): {sum(binary)}, False (below): {len(binary) - sum(binary)}")

print(f"\n--- Quantile Encoding ---")
for n_bins in [4, 8, 16]:
    quantile = pymlfinance.features.quantile_encode(series, num_bins=n_bins)
    unique = len(set(quantile))
    print(f"  {n_bins} bins: {unique} unique values, "
          f"distribution: {[quantile.count(i) for i in range(min(n_bins, 4))]}")

print(f"\n--- Sigma Encoding ---")
for n_bands in [2, 3, 5]:
    sigma = pymlfinance.features.sigma_encode(series, num_bands=n_bands)
    unique = len(set(sigma))
    print(f"  {n_bands} bands: {unique} unique values")

# --- Shannon Entropy ---
print(f"\n--- Shannon Entropy ---")
# Uniform distribution
uniform_probs = np.ones(10) / 10
h_uniform = pymlfinance.features.shannon_entropy(uniform_probs)
# Peaked distribution
peaked_probs = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0005])
peaked_probs /= peaked_probs.sum()
h_peaked = pymlfinance.features.shannon_entropy(peaked_probs)
# Binary
binary_probs = np.array([0.5, 0.5])
h_binary = pymlfinance.features.shannon_entropy(binary_probs)
print(f"  Uniform (10 symbols): {h_uniform:.4f} bits (max entropy)")
print(f"  Peaked (10 symbols):  {h_peaked:.4f} bits (low entropy)")
print(f"  Binary (50/50):       {h_binary:.4f} bits")

# --- Plugin Entropy ---
print(f"\n--- Plugin Entropy (from encoded sequences) ---")
quantile_seq = pymlfinance.features.quantile_encode(series, num_bins=8)
h_plugin = pymlfinance.features.plugin_entropy(quantile_seq, num_symbols=8)
print(f"  Full series (8 bins): {h_plugin:.4f}")

# Per-regime
for name, start, end in [("Trending", 0, 100), ("Random", 100, 250),
                          ("Mean-rev", 250, 350), ("Random2", 350, 500)]:
    seg = pymlfinance.features.quantile_encode(series[start:end], num_bins=8)
    h = pymlfinance.features.plugin_entropy(seg, num_symbols=8)
    print(f"  {name:>10s} [{start}-{end}]: {h:.4f}")

# --- Kontoyiannis Entropy ---
print(f"\n--- Kontoyiannis Entropy ---")
for window in [10, 20, 50]:
    h_kont = pymlfinance.features.kontoyiannis_entropy(quantile_seq, window=window)
    print(f"  Window={window}: {h_kont:.4f}")

# --- Lempel-Ziv Complexity ---
print(f"\n--- Lempel-Ziv Complexity ---")
lz = pymlfinance.features.lempel_ziv_complexity(binary)
print(f"  Full series: {lz} distinct patterns")
print(f"  Normalized: {lz / len(binary):.4f} patterns/symbol")

# Per-regime
for name, start, end in [("Trending", 0, 100), ("Random", 100, 250),
                          ("Mean-rev", 250, 350), ("Random2", 350, 500)]:
    seg_binary = pymlfinance.features.binary_encode(series[start:end])
    lz_seg = pymlfinance.features.lempel_ziv_complexity(seg_binary)
    print(f"  {name:>10s}: {lz_seg} patterns, normalized={lz_seg/len(seg_binary):.4f}")

# --- Polars API ---
print(f"\n--- Polars API ---")
import pymlfinance.polars

df = pl.DataFrame({"value": series})
result = df.with_columns(
    pl.col("value").ml.binary_encode().alias("binary"),
    pl.col("value").ml.quantile_encode(n_bins=8).alias("quantile"),
    pl.col("value").ml.sigma_encode(n_bands=3).alias("sigma"),
)
print(f"  Encoded DataFrame: {result.shape}")
print(result.head(5))

# Scalar entropy measures
binary_col = [bool(b) for b in binary]
entropy_df = pl.DataFrame({"binary": binary_col, "quantile": quantile_seq})
lz_pl = entropy_df.select(pl.col("binary").ml.lempel_ziv_complexity()).item()
print(f"  Polars LZ complexity: {lz_pl}")

shannon_probs = pl.DataFrame({"probs": uniform_probs.tolist()})
h_pl = shannon_probs.select(pl.col("probs").ml.shannon_entropy()).item()
print(f"  Polars Shannon entropy: {h_pl:.4f}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Generate a perfectly periodic series and measure its entropy (should be low)")
print("2. Compare entropy measures across different market regimes")
print("3. Use entropy features as inputs to a classifier for regime detection")
