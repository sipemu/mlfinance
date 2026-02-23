"""
Chapter 7: Cross-Validation in Finance
=======================================
AFML Ch. 7 — Purged K-Fold cross-validation with embargo.

Demonstrates:
- PurgedKFold splitter with purging and embargo
- cv_score with a custom classifier
- Comparison of purged vs unpurged CV
"""

import numpy as np
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 7: Cross-Validation in Finance")
print("=" * 60)

# --- Generate synthetic classification data with overlapping labels ---
n_samples = 200
n_features = 5

# Create features with some signal
X, y = pymlfinance.modeling.make_classification(
    n_samples=n_samples,
    n_informative=3,
    n_redundant=1,
    n_noise=1,
    seed=42
)
print(f"\nSynthetic dataset: {n_samples} samples, {n_features} features")
print(f"  Label distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")

# Create overlapping events (entry, exit)
entries = np.arange(n_samples)
durations = np.random.randint(5, 20, n_samples)
events = [(int(e), min(int(e + d), n_samples + 50)) for e, d in zip(entries, durations)]

# --- PurgedKFold ---
print(f"\n--- PurgedKFold (5 folds, 2% embargo) ---")
pkf = pymlfinance.modeling.PurgedKFold(n_splits=5, embargo_pct=0.02)
folds = pkf.split(events, n_samples)

for i, fold in enumerate(folds):
    train_idx = fold.train
    test_idx = fold.test
    print(f"  Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}, "
          f"test_range=[{min(test_idx)}-{max(test_idx)}]")

# --- Verify purging ---
print(f"\n--- Purging Verification ---")
for i, fold in enumerate(folds):
    train_set = set(fold.train)
    test_set = set(fold.test)
    # Check no overlap
    overlap = train_set & test_set
    print(f"  Fold {i+1}: train∩test overlap = {len(overlap)} (should be 0)")

# --- CV Score with simple classifier ---
# Create a simple classifier class that follows the sklearn interface
class SimpleClassifier:
    """A minimal threshold-based classifier for demonstration."""
    def __init__(self):
        self.threshold = 0.0

    def fit(self, X, y, sample_weight=None):
        # Simple: use the mean of first feature as threshold
        pos_mean = np.mean(X[y == 1, 0]) if np.any(y == 1) else 0
        neg_mean = np.mean(X[y == 0, 0]) if np.any(y == 0) else 0
        self.threshold = (pos_mean + neg_mean) / 2
        return self

    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(np.int32)

clf = SimpleClassifier()
scores = pymlfinance.modeling.cv_score(
    classifier=clf,
    x=X,
    y=y.astype(np.float64),
    events=events,
    n_splits=5,
    embargo_pct=0.02
)
print(f"\n--- CV Scores (PurgedKFold) ---")
print(f"  Per-fold scores: [{', '.join(f'{s:.4f}' for s in scores)}]")
print(f"  Mean accuracy:   {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# --- Different embargo percentages ---
print(f"\n--- Embargo Sensitivity ---")
for embargo in [0.0, 0.01, 0.02, 0.05, 0.10]:
    clf = SimpleClassifier()
    s = pymlfinance.modeling.cv_score(
        classifier=clf, x=X, y=y.astype(np.float64),
        events=events, n_splits=5, embargo_pct=embargo
    )
    print(f"  embargo={embargo:.0%}: mean={np.mean(s):.4f} ± {np.std(s):.4f}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Increase event overlap (longer durations) and observe purging effects")
print("2. Compare 3, 5, and 10 fold CV and how it affects score stability")
print("3. Try a more sophisticated classifier and see if purged CV matters more")
