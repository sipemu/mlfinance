#!/usr/bin/env python3
"""Generate reference test fixtures from Python implementations of AFML algorithms.

Outputs JSON files to tests/fixtures/ for use by Rust integration tests.
Requires: numpy, pandas (pip install numpy pandas)

Run once locally: python scripts/generate_fixtures.py
"""

import json
import math
import os
import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(name, data):
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {path}")


# --- FFD Weights (Snippet 5.3) ---
def get_weights_ffd(d, threshold):
    """Fixed-width window fractional differentiation weights."""
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        w.append(w_next)
        k += 1
    return w


def generate_ffd_weights():
    cases = []
    for d, threshold in [(0.5, 1e-4), (0.3, 1e-5), (1.0, 1e-4), (0.7, 1e-3)]:
        weights = get_weights_ffd(d, threshold)
        cases.append({"d": d, "threshold": threshold, "weights": weights})
    save_fixture("ffd_weights.json", {"cases": cases})


# --- FFD Series (Snippet 5.3) ---
def frac_diff_ffd(series, d, threshold):
    """Apply fractional differentiation with FFD.

    Returns same-length output with NaN prefix (matching Rust implementation).
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    output = [None] * len(series)  # None for NaN prefix
    for i in range(width - 1, len(series)):
        val = 0.0
        for k, w in enumerate(weights):
            val += w * series[i - k]
        output[i] = val
    return output


def generate_ffd_series():
    np.random.seed(42)
    series = list(np.cumsum(np.random.randn(100)) + 100)
    cases = []
    for d, threshold in [(0.5, 1e-4), (0.3, 1e-5)]:
        result = frac_diff_ffd(series, d, threshold)
        cases.append({"d": d, "threshold": threshold, "series": series, "result": result})
    save_fixture("ffd_series.json", {"cases": cases})


# --- CUSUM Filter (Snippet 2.4) ---
def cusum_filter(values, threshold):
    """Symmetric CUSUM filter."""
    events = []
    s_pos, s_neg = 0.0, 0.0
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        if s_pos >= threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg <= -threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return events


def generate_cusum_events():
    np.random.seed(42)
    values = list(np.cumsum(np.random.randn(200) * 0.01))
    cases = []
    for threshold in [0.05, 0.1, 0.02]:
        events = cusum_filter(values, threshold)
        cases.append({"values": values, "threshold": threshold, "events": events})
    save_fixture("cusum_events.json", {"cases": cases})


# --- Daily Volatility (Snippet 3.1) ---
def daily_volatility(prices, span):
    """EWMA standard deviation of returns."""
    returns = [0.0]
    for i in range(1, len(prices)):
        returns.append(prices[i] / prices[i - 1] - 1.0)

    # EWMA variance
    alpha = 2.0 / (span + 1)
    ewma_var = [0.0] * len(returns)
    ewma_mean = [0.0] * len(returns)
    ewma_mean[0] = returns[0]
    ewma_var[0] = 0.0
    for i in range(1, len(returns)):
        ewma_mean[i] = alpha * returns[i] + (1 - alpha) * ewma_mean[i - 1]
        diff = returns[i] - ewma_mean[i - 1]
        ewma_var[i] = (1 - alpha) * (ewma_var[i - 1] + alpha * diff * diff)

    return [math.sqrt(max(0, v)) for v in ewma_var]


def generate_daily_volatility():
    np.random.seed(42)
    prices = list(np.cumprod(1 + np.random.randn(100) * 0.01) * 100)
    cases = []
    for span in [20, 50]:
        result = daily_volatility(prices, span)
        cases.append({"prices": prices, "span": span, "result": result})
    save_fixture("daily_volatility.json", {"cases": cases})


# --- Triple Barrier Events (Snippets 3.2-3.3) ---
def find_first_touch(prices, entry_idx, upper_barrier, lower_barrier, max_holding, daily_vol):
    """Find first barrier touch."""
    entry_price = prices[entry_idx]
    end_idx = min(entry_idx + max_holding, len(prices) - 1) if max_holding else len(prices) - 1

    for i in range(entry_idx + 1, end_idx + 1):
        ret = (prices[i] - entry_price) / entry_price
        if upper_barrier is not None and ret > upper_barrier * daily_vol:
            return {"exit_idx": i, "touch_type": "Upper", "return_value": ret}
        if lower_barrier is not None and ret < -lower_barrier * daily_vol:
            return {"exit_idx": i, "touch_type": "Lower", "return_value": ret}

    # Vertical barrier
    final_ret = (prices[end_idx] - entry_price) / entry_price
    return {"exit_idx": end_idx, "touch_type": "Vertical", "return_value": final_ret}


def generate_triple_barrier_events():
    np.random.seed(42)
    prices = list(np.cumprod(1 + np.random.randn(200) * 0.02) * 100)
    entry_indices = list(range(0, 150, 20))
    daily_vols = [0.02] * len(prices)
    events = []
    for idx in entry_indices:
        result = find_first_touch(prices, idx, 2.0, 2.0, 50, daily_vols[idx])
        events.append({"entry_idx": idx, **result})
    save_fixture(
        "triple_barrier_events.json",
        {
            "prices": prices,
            "entry_indices": entry_indices,
            "upper_barrier": 2.0,
            "lower_barrier": 2.0,
            "max_holding_period": 50,
            "daily_vols": daily_vols,
            "events": events,
        },
    )


# --- Plugin Entropy (Snippet 18.1) ---
def plugin_entropy(sequence, num_symbols):
    """Shannon entropy estimator using plug-in (maximum likelihood) method."""
    counts = [0] * num_symbols
    for s in sequence:
        counts[s] += 1
    n = len(sequence)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    return entropy


# --- Lempel-Ziv Complexity (Snippet 18.2) ---
def lempel_ziv_complexity(binary_string):
    """Lempel-Ziv complexity of a binary string.

    Matches the Rust implementation: parse from position 1, extend match
    within binary_string[0..pos+match_len], count new substrings.
    """
    n = len(binary_string)
    if n == 0:
        return 0

    complexity = 1  # First symbol is always new
    i = 1

    while i < n:
        match_len = 0
        reached_end = False

        while True:
            if i + match_len >= n:
                reached_end = True
                break

            target = binary_string[i : i + match_len + 1]
            search_end = i + match_len

            # Check if target exists as substring in binary_string[0..search_end]
            found = False
            target_len = len(target)
            if target_len <= search_end:
                for start in range(search_end - target_len + 1):
                    if binary_string[start : start + target_len] == target:
                        found = True
                        break

            if found:
                match_len += 1
            else:
                break

        if not reached_end:
            complexity += 1
        i += match_len + 1

    return complexity


def generate_entropy():
    cases = []
    # Plugin entropy
    for seq, num_sym in [
        ([0, 1, 0, 1, 0, 1, 0, 1], 2),
        ([0, 0, 0, 0, 0], 2),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10),
        (list(range(5)) * 20, 5),
    ]:
        ent = plugin_entropy(seq, num_sym)
        cases.append({"type": "plugin", "sequence": seq, "num_symbols": num_sym, "result": ent})

    # Lempel-Ziv
    for bits in [
        [True, False, True, False, True, False],
        [True, True, True, True, True],
        [True, False, True, True, False, False, True, False, True, True],
    ]:
        lz = lempel_ziv_complexity(bits)
        cases.append({"type": "lempel_ziv", "binary_string": bits, "result": lz})

    save_fixture("entropy.json", {"cases": cases})


# --- Bet Sizing (Snippet 10.1) ---
def sigmoid_bet_size(prob, num_classes):
    """Sigmoid bet sizing."""
    if num_classes == 2:
        signal = prob - 0.5
        return 2.0 * signal
    else:
        return (prob - 1.0 / num_classes) / (1.0 - 1.0 / num_classes)


def power_bet_size(prob, num_classes, exponent):
    """Power bet sizing."""
    base = sigmoid_bet_size(prob, num_classes)
    sign = 1.0 if base >= 0 else -1.0
    return sign * abs(base) ** exponent


def generate_bet_sizing():
    cases = []
    for prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for num_classes in [2, 3]:
            sig = sigmoid_bet_size(prob, num_classes)
            cases.append(
                {"type": "sigmoid", "prob": prob, "num_classes": num_classes, "result": sig}
            )
            for exp in [1.0, 2.0, 0.5]:
                pw = power_bet_size(prob, num_classes, exp)
                cases.append(
                    {
                        "type": "power",
                        "prob": prob,
                        "num_classes": num_classes,
                        "exponent": exp,
                        "result": pw,
                    }
                )
    save_fixture("bet_sizing.json", {"cases": cases})


# --- Sequential Bootstrap (Snippet 4.5) ---
def get_indicator_matrix(events, num_bars):
    """Build indicator matrix from event spans."""
    n_events = len(events)
    matrix = [[0.0] * n_events for _ in range(num_bars)]
    for j, (start, end) in enumerate(events):
        for i in range(start, end + 1):
            if i < num_bars:
                matrix[i][j] = 1.0
    return matrix


def average_uniqueness_from_matrix(ind_matrix, num_bars, n_events):
    """Compute average uniqueness for each event."""
    # Concurrency: count of events active at each bar
    concurrency = [0.0] * num_bars
    for i in range(num_bars):
        for j in range(n_events):
            concurrency[i] += ind_matrix[i][j]

    uniqueness = []
    for j in range(n_events):
        u_sum = 0.0
        count = 0
        for i in range(num_bars):
            if ind_matrix[i][j] > 0.0 and concurrency[i] > 0.0:
                u_sum += 1.0 / concurrency[i]
                count += 1
        if count > 0:
            uniqueness.append(u_sum / count)
        else:
            uniqueness.append(0.0)
    return uniqueness


def generate_seq_bootstrap_draws():
    events = [(0, 5), (3, 8), (6, 12), (10, 15), (13, 18)]
    num_bars = 20
    ind_matrix = get_indicator_matrix(events, num_bars)
    avg_uniq = average_uniqueness_from_matrix(ind_matrix, num_bars, len(events))

    save_fixture(
        "seq_bootstrap_draws.json",
        {
            "events": events,
            "num_bars": num_bars,
            "n_samples": len(events),
            "avg_uniqueness": avg_uniq,
            "note": "RNG differs between Rust/Python; validate uniqueness scores, not exact draws",
        },
    )


# --- Purged K-Fold (Snippet 7.1) ---
def purged_kfold_split(events, n_samples, n_splits, embargo_pct):
    """Split events into purged train/test folds."""
    test_size = n_samples // n_splits
    folds = []

    for fold_idx in range(n_splits):
        test_start = fold_idx * test_size
        if fold_idx == n_splits - 1:
            test_end = n_samples
        else:
            test_end = test_start + test_size

        test_indices = list(range(test_start, test_end))

        # Find earliest event start and latest event end in test set
        embargo_size = int(n_samples * embargo_pct)

        # Train = everything not in test, minus purged and embargoed
        train_indices = []
        for i in range(n_samples):
            if i in test_indices:
                continue
            # Purge: remove from train if overlaps with any test event
            purged = False
            for t_idx in test_indices:
                t_start, t_end = events[t_idx]
                i_start, i_end = events[i]
                if i_start <= t_end and i_end >= t_start:
                    purged = True
                    break
            if purged:
                continue
            # Embargo
            if test_end <= i < test_end + embargo_size:
                continue
            train_indices.append(i)

        folds.append({"train": train_indices, "test": test_indices})
    return folds


def generate_purged_kfold_splits():
    # Non-overlapping events
    events = [(i, i + 2) for i in range(0, 20, 2)]  # 10 events
    n_samples = len(events)
    cases = []
    for n_splits, embargo_pct in [(3, 0.0), (5, 0.0), (3, 0.1)]:
        folds = purged_kfold_split(events, n_samples, n_splits, embargo_pct)
        cases.append(
            {
                "events": events,
                "n_samples": n_samples,
                "n_splits": n_splits,
                "embargo_pct": embargo_pct,
                "folds": folds,
            }
        )
    save_fixture("purged_kfold_splits.json", {"cases": cases})


# --- SADF (Snippet 17.1) ---
def adf_stat(series, max_lags):
    """Compute ADF t-statistic using OLS on dy = alpha + beta*y_{t-1} + lags."""
    n = len(series)
    dy = [series[i] - series[i - 1] for i in range(1, n)]
    y_lag = series[:-1]

    # Include lagged differences
    actual_lags = min(max_lags, len(dy) - 2)
    if actual_lags < 0:
        actual_lags = 0

    start = actual_lags + 1 if actual_lags > 0 else 1
    y_dep = dy[start - 1 :]
    n_obs = len(y_dep)
    if n_obs < 3:
        return float("nan")

    # Build X matrix: [1, y_{t-1}, dy_{t-1}, ..., dy_{t-lags}]
    X = []
    for i in range(n_obs):
        row = [1.0, y_lag[start - 1 + i]]
        for lag in range(1, actual_lags + 1):
            row.append(dy[start - 1 + i - lag])
        X.append(row)

    # OLS: beta = (X'X)^-1 X'y
    k = len(X[0])
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    for i in range(n_obs):
        for j in range(k):
            Xty[j] += X[i][j] * y_dep[i]
            for l in range(k):
                XtX[j][l] += X[i][j] * X[i][l]

    # Solve via numpy for accuracy
    X_np = np.array(X)
    y_np = np.array(y_dep)
    try:
        beta = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")

    residuals = y_np - X_np @ beta
    s2 = float(np.sum(residuals**2) / (n_obs - k))
    try:
        var_beta = s2 * np.linalg.inv(X_np.T @ X_np)
    except np.linalg.LinAlgError:
        return float("nan")
    se_beta1 = float(np.sqrt(var_beta[1, 1]))
    if se_beta1 == 0:
        return float("nan")
    return float(beta[1] / se_beta1)


def sadf(series, min_window, max_lags):
    """Supremum ADF statistic."""
    stats = []
    for end in range(min_window, len(series)):
        subseries = series[: end + 1]
        stat = adf_stat(subseries, max_lags)
        stats.append(stat)
    return stats


def generate_sadf_stats():
    np.random.seed(42)
    # Random walk with slight drift (should not trigger bubble detection)
    series = list(np.cumsum(np.random.randn(60) * 0.5 + 0.01))
    stats = sadf(series, 20, 3)
    save_fixture(
        "sadf_stats.json",
        {
            "series": series,
            "min_window": 20,
            "max_lags": 3,
            "stats": stats,
            "sadf_stat": max(stats) if stats else None,
        },
    )


# --- HRP Weights (Snippet 16.3) ---
def generate_hrp_weights():
    """Generate HRP reference data.

    Note: HRP involves clustering which can have tie-breaking differences.
    We store the returns matrix and expected weight sum/positivity constraints
    rather than exact weights.
    """
    np.random.seed(42)
    n_obs, n_assets = 100, 5
    returns = np.random.randn(n_obs, n_assets) * 0.01
    # Add some correlation structure
    returns[:, 1] = returns[:, 0] * 0.8 + returns[:, 1] * 0.2
    returns[:, 3] = returns[:, 2] * 0.6 + returns[:, 3] * 0.4

    save_fixture(
        "hrp_weights.json",
        {
            "returns": returns.tolist(),
            "n_obs": n_obs,
            "n_assets": n_assets,
            "note": "Clustering tie-breaking may differ; validate weights sum to 1 and are all positive",
        },
    )


if __name__ == "__main__":
    print("Generating test fixtures...")
    generate_ffd_weights()
    generate_ffd_series()
    generate_cusum_events()
    generate_daily_volatility()
    generate_triple_barrier_events()
    generate_entropy()
    generate_bet_sizing()
    generate_seq_bootstrap_draws()
    generate_purged_kfold_splits()
    generate_sadf_stats()
    generate_hrp_weights()
    print("Done! All fixtures written to tests/fixtures/")
