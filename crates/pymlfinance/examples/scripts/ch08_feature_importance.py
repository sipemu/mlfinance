"""
Chapter 8: Feature Importance
==============================
AFML Ch. 8 — MDI, MDA, and SFI methods for understanding feature relevance.

Demonstrates:
- make_classification with known informative features
- Mean Decrease Impurity (MDI)
- Mean Decrease Accuracy (MDA)
- Single Feature Importance (SFI)
- Orthogonal features via PCA
"""

import numpy as np
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 8: Feature Importance")
print("=" * 60)

# --- Generate synthetic data with known feature structure ---
X, y = pymlfinance.modeling.make_classification(
    n_samples=500,
    n_informative=3,
    n_redundant=2,
    n_noise=5,
    seed=42
)
n_features = X.shape[1]
print(f"\nDataset: {X.shape[0]} samples, {n_features} features")
print(f"  Informative: features 0-2")
print(f"  Redundant: features 3-4")
print(f"  Noise: features 5-9")

# Create events for purged CV
events = [(i, min(i + 10, X.shape[0] - 1)) for i in range(X.shape[0])]

# --- MDI (Mean Decrease Impurity) ---
# Simulate tree importances (since we don't have sklearn)
# Each "tree" gives Gini importances for each feature
n_trees = 50
# Informative features have higher importances
base_importance = np.zeros(n_features)
base_importance[:3] = 0.2   # informative
base_importance[3:5] = 0.1  # redundant
base_importance[5:] = 0.02  # noise

importances_per_tree = []
for _ in range(n_trees):
    tree_imp = base_importance + np.abs(np.random.randn(n_features) * 0.03)
    tree_imp /= tree_imp.sum()  # normalize per tree
    importances_per_tree.append(tree_imp.tolist())

mdi = pymlfinance.modeling.mean_decrease_impurity(importances_per_tree)
print(f"\n--- Mean Decrease Impurity (MDI) ---")
for i in range(n_features):
    bar = "█" * int(mdi[i] * 100)
    label = "INFO" if i < 3 else ("REDUN" if i < 5 else "NOISE")
    print(f"  Feature {i} [{label:>5s}]: {mdi[i]:.4f} {bar}")

# --- MDA (Mean Decrease Accuracy) ---
class SimpleClassifier:
    def __init__(self):
        self.weights = None
    def fit(self, X, y, sample_weight=None):
        # Simple logistic-like: correlate each feature with y
        self.weights = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
        return self
    def predict(self, X):
        scores = X @ self.weights
        return np.where(scores > 0, 1, -1).astype(np.int32)

clf = SimpleClassifier()
clf.fit(X, y)
mda = pymlfinance.modeling.mean_decrease_accuracy(clf, X, y.astype(np.float64), seed=42)
print(f"\n--- Mean Decrease Accuracy (MDA) ---")
for i in range(n_features):
    bar = "█" * max(0, int(mda[i] * 200))
    label = "INFO" if i < 3 else ("REDUN" if i < 5 else "NOISE")
    print(f"  Feature {i} [{label:>5s}]: {mda[i]:>+.4f} {bar}")

# --- SFI (Single Feature Importance) ---
clf2 = SimpleClassifier()
sfi = pymlfinance.modeling.single_feature_importance(
    clf2, X, y.astype(np.float64), events, n_splits=3
)
print(f"\n--- Single Feature Importance (SFI) ---")
for i in range(n_features):
    bar = "█" * int(sfi[i] * 20)
    label = "INFO" if i < 3 else ("REDUN" if i < 5 else "NOISE")
    print(f"  Feature {i} [{label:>5s}]: {sfi[i]:.4f} {bar}")

# --- Orthogonal Features (PCA) ---
transformed, var_ratio = pymlfinance.modeling.orthogonal_features(X, n_components=5)
print(f"\n--- Orthogonal Features (PCA) ---")
print(f"  Top 5 components explain: {np.sum(var_ratio):.2%} of variance")
for i, vr in enumerate(var_ratio):
    bar = "█" * int(vr * 100)
    print(f"  PC{i}: {vr:.4f} {bar}")
print(f"  Transformed shape: {transformed.shape}")

# --- Importance Comparison ---
print(f"\n--- Importance Ranking Comparison ---")
mdi_rank = np.argsort(-mdi)
mda_rank = np.argsort(-mda)
sfi_rank = np.argsort(-sfi)
print(f"  {'Feature':<10} {'MDI Rank':>10} {'MDA Rank':>10} {'SFI Rank':>10}")
for i in range(n_features):
    mdi_r = np.where(mdi_rank == i)[0][0] + 1
    mda_r = np.where(mda_rank == i)[0][0] + 1
    sfi_r = np.where(sfi_rank == i)[0][0] + 1
    print(f"  Feature {i:<3d} {mdi_r:>10d} {mda_r:>10d} {sfi_r:>10d}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Increase noise features and observe how MDI becomes unreliable")
print("2. Compare MDA with a better classifier vs a weak one")
print("3. Use orthogonal features as input and re-run importance analysis")
