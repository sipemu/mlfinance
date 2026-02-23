#!/usr/bin/env python3
"""Generate reference test fixtures from Python implementations of AFML algorithms.

Outputs JSON files to tests/fixtures/ for use by Rust integration tests.
Requires: numpy, scipy (pip install numpy scipy)

Run once locally: python scripts/generate_fixtures.py
"""

import json
import math
import os
from itertools import combinations

import numpy as np
from scipy.stats import norm

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
    """EWMA standard deviation of log returns, matching Rust implementation.

    Rust: log_returns() then ewma_std() with first-difference variance.
    """
    if len(prices) < 2:
        return [0.0] * len(prices)

    # Log returns (matching Rust math::log_returns)
    log_ret = []
    for i in range(1, len(prices)):
        if prices[i - 1] <= 0 or prices[i] <= 0:
            log_ret.append(0.0)
        else:
            log_ret.append(math.log(prices[i] / prices[i - 1]))

    # ewma_std (matching Rust math::ewma_std)
    alpha = 2.0 / (span + 1)
    variance = 0.0
    ewma_vol = [0.0]  # first element is 0.0
    for i in range(1, len(log_ret)):
        diff = log_ret[i] - log_ret[i - 1]
        variance = (1 - alpha) * variance + alpha * diff * diff
        ewma_vol.append(math.sqrt(variance))

    # Prepend 0.0 for the first price (no return available)
    return [0.0] + ewma_vol


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
    """Supremum ADF statistic (matching Rust windowing: min_window..=n)."""
    stats = []
    n = len(series)
    for end in range(min_window, n + 1):
        subseries = series[:end]
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


# --- Backtesting Statistics ---
def generate_backtesting_statistics():
    np.random.seed(42)
    returns = list(np.random.randn(100) * 0.01 + 0.0003)

    cases = []

    # sharpe_ratio
    for rf, periods in [(0.0, 252.0), (0.0001, 252.0), (0.0, 12.0)]:
        n = len(returns)
        mean_r = sum(returns) / n
        excess = mean_r - rf
        var = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(var)
        sr = (excess / std) * math.sqrt(periods) if std > 1e-15 else 0.0
        cases.append(
            {
                "type": "sharpe_ratio",
                "returns": returns,
                "risk_free_rate": rf,
                "periods_per_year": periods,
                "result": sr,
            }
        )

    # hit_ratio
    for rets in [returns, [0.01, -0.02, 0.03, -0.01, 0.005], [-0.01, -0.02, -0.03]]:
        positives = sum(1 for r in rets if r > 0.0)
        hr = positives / len(rets)
        cases.append({"type": "hit_ratio", "returns": rets, "result": hr})

    # avg_holding_period
    for pairs in [[[0, 5], [10, 15], [20, 30]], [[0, 10]], [[10, 5]]]:
        total = sum(max(exit_b - entry, 0) for entry, exit_b in pairs)
        ahp = total / len(pairs)
        cases.append({"type": "avg_holding_period", "pairs": pairs, "result": ahp})

    # turnover
    for positions in [[0.0, 1.0, 0.5, -0.5, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0]]:
        if len(positions) < 2:
            t = 0.0
        else:
            changes = sum(
                abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)
            )
            t = changes / (len(positions) - 1)
        cases.append({"type": "turnover", "positions": positions, "result": t})

    # hhi
    for weights in [
        [0.25, 0.25, 0.25, 0.25],
        [1.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    ]:
        total = sum(abs(w) for w in weights)
        h = sum((abs(w) / total) ** 2 for w in weights) if total > 1e-15 else 0.0
        cases.append({"type": "hhi", "weights": weights, "result": h})

    # hhi_concentration
    def compute_hhi(w):
        total = sum(abs(x) for x in w)
        if total < 1e-15 or not w:
            return 0.0
        return sum((abs(x) / total) ** 2 for x in w)

    for rets in [[0.1, -0.05, 0.2, -0.1, 0.05], [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]:
        positives = [r for r in rets if r > 0.0]
        negatives = [abs(r) for r in rets if r < 0.0]
        cases.append(
            {
                "type": "hhi_concentration",
                "returns": rets,
                "hhi_positive": compute_hhi(positives),
                "hhi_negative": compute_hhi(negatives),
            }
        )

    # cscv
    np.random.seed(123)
    cscv_returns = np.random.randn(8, 3) * 0.01
    num_groups = 4
    n_rows, n_strategies = cscv_returns.shape
    group_size = n_rows // num_groups
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = n_rows if g == num_groups - 1 else (g + 1) * group_size
        groups.append(list(range(start, end)))
    half = num_groups // 2
    group_indices = list(range(num_groups))
    combos = list(combinations(group_indices, half))

    rank_logits = []
    overfit_count = 0
    for is_combo in combos:
        oos_combo = [i for i in group_indices if i not in is_combo]
        is_rows = [r for g in is_combo for r in groups[g]]
        oos_rows = [r for g in oos_combo for r in groups[g]]
        if not is_rows or not oos_rows:
            continue
        is_perf = [sum(cscv_returns[r, s] for r in is_rows) for s in range(n_strategies)]
        best_is = max(range(n_strategies), key=lambda s: is_perf[s])
        oos_perf = [sum(cscv_returns[r, s] for r in oos_rows) for s in range(n_strategies)]
        best_oos_val = oos_perf[best_is]
        rank = sum(1 for p in oos_perf if p > best_oos_val) + 1
        n_s = n_strategies
        denom = n_s + 1 - rank
        logit = math.log(rank / denom) if denom > 0 else float("inf")
        rank_logits.append(logit)
        if rank > n_strategies // 2:
            overfit_count += 1

    cscv_total = max(len(rank_logits), 1)
    cscv_pbo = overfit_count / cscv_total
    cases.append(
        {
            "type": "cscv",
            "returns_matrix": cscv_returns.tolist(),
            "num_groups": num_groups,
            "pbo": cscv_pbo,
            "rank_logits": rank_logits,
        }
    )

    save_fixture("backtesting_statistics.json", {"cases": cases})


# --- Drawdowns ---
def generate_drawdowns():
    cases = []
    for label, rets in [
        ("up_then_down", [0.10, 0.05, -0.10, -0.05, 0.20]),
        ("all_positive", [0.01, 0.02, 0.03, 0.01, 0.02]),
        ("all_negative", [-0.01, -0.02, -0.03, -0.01, -0.02]),
        ("recovery", [-0.10, -0.05, 0.10, 0.10, 0.05]),
    ]:
        # Wealth curve
        wealth = []
        w = 1.0
        for r in rets:
            w *= 1.0 + r
            wealth.append(w)

        # High-water mark
        hwm = []
        current_max = float("-inf")
        for wv in wealth:
            current_max = max(current_max, wv)
            hwm.append(current_max)

        # Drawdown series
        dd_series = []
        for wv, h in zip(wealth, hwm):
            if h > 0.0:
                dd_series.append(max((h - wv) / h, 0.0))
            else:
                dd_series.append(0.0)

        # Time under water
        tuw = []
        underwater_count = 0
        for dd in dd_series:
            if dd > 1e-15:
                underwater_count += 1
            else:
                underwater_count = 0
            tuw.append(underwater_count)

        max_dd = max(dd_series)
        max_dd_duration = max(tuw)

        cases.append(
            {
                "label": label,
                "returns": rets,
                "max_drawdown": max_dd,
                "max_drawdown_duration": max_dd_duration,
                "drawdown_series": dd_series,
                "time_under_water": tuw,
            }
        )

    save_fixture("drawdowns.json", {"cases": cases})


# --- PSR and DSR ---
def generate_psr_dsr():
    cases = []

    # PSR cases
    for observed_sr, benchmark_sr, n_obs, skewness, kurtosis in [
        (2.0, 0.0, 252, 0.0, 3.0),
        (0.5, 2.0, 252, 0.0, 3.0),
        (1.0, 1.0, 252, 0.0, 3.0),
        (1.5, 1.0, 100, 0.0, 3.0),
        (1.5, 1.0, 100, -1.0, 3.0),
        (0.3, 0.0, 30, 0.0, 0.0),
        (0.8, 0.5, 100, -0.5, 4.0),
    ]:
        # Match Rust PSR formula
        n = n_obs
        sr = observed_sr
        denom_sq = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr
        if denom_sq <= 0.0:
            psr = 1.0 if observed_sr > benchmark_sr else 0.0
        else:
            denom = math.sqrt(denom_sq)
            numerator = (sr - benchmark_sr) * math.sqrt(n - 1)
            z = numerator / denom if denom > 1e-15 else (1.0 if numerator > 0 else 0.0)
            psr = float(norm.cdf(z))

        cases.append(
            {
                "type": "psr",
                "observed_sr": observed_sr,
                "benchmark_sr": benchmark_sr,
                "n_observations": n_obs,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "result": psr,
            }
        )

    # DSR cases
    euler_mascheroni = 0.5772156649015329
    for observed_sr, sr_std, n_obs, n_trials, skewness, kurtosis in [
        (1.0, 1.0, 252, 1, 0.0, 3.0),
        (1.5, 1.0, 252, 5, 0.0, 3.0),
        (1.5, 1.0, 252, 1000, 0.0, 3.0),
        (2.0, 0.5, 100, 50, -0.3, 4.0),
        (5.0, 1.0, 1000, 100, 0.0, 3.0),
    ]:
        # Expected max SR under null
        if n_trials <= 1:
            expected_max_sr = 0.0
        else:
            n_t = n_trials
            p1 = min(1.0 - 1.0 / n_t, 1.0 - 1e-15)
            p2 = min(1.0 - 1.0 / (n_t * math.e), 1.0 - 1e-15)
            z1 = float(norm.ppf(p1))
            z2 = float(norm.ppf(p2))
            expected_max_sr = sr_std * (
                (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2
            )

        # PSR formula with deflated benchmark
        sr = observed_sr
        denom_sq = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr
        if denom_sq <= 0.0:
            dsr = 1.0 if observed_sr > expected_max_sr else 0.0
        else:
            denom = math.sqrt(denom_sq)
            numerator = (sr - expected_max_sr) * math.sqrt(n_obs - 1)
            z = numerator / denom if denom > 1e-15 else (1.0 if numerator > 0 else 0.0)
            dsr = float(norm.cdf(z))

        cases.append(
            {
                "type": "dsr",
                "observed_sr": observed_sr,
                "sr_std": sr_std,
                "n_observations": n_obs,
                "n_trials": n_trials,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "result": dsr,
            }
        )

    save_fixture("psr_dsr.json", {"cases": cases})


# --- Strategy Risk ---
def generate_strategy_risk():
    cases = []
    for estimated_precision, n_obs, break_even in [
        (0.7, 1000, 0.5),
        (0.5, 1000, 0.5),
        (0.4, 1000, 0.5),
        (0.6, 10, 0.5),
        (0.6, 1000, 0.5),
        (0.55, 200, 0.5),
    ]:
        n = n_obs
        se = math.sqrt(break_even * (1 - break_even) / n)
        z = (estimated_precision - break_even) / se
        result = 1.0 - float(norm.cdf(z))
        cases.append(
            {
                "estimated_precision": estimated_precision,
                "n_observations": n_obs,
                "break_even_precision": break_even,
                "result": result,
            }
        )

    save_fixture("strategy_risk.json", {"cases": cases})


# --- Multiple Testing Corrections ---
def generate_multiple_testing():
    cases = []

    # Bonferroni
    for p_values in [[0.01, 0.04, 0.03, 0.005], [0.5, 0.3], [0.001, 0.01, 0.05]]:
        n = len(p_values)
        adjusted = [min(p * n, 1.0) for p in p_values]
        cases.append(
            {"type": "bonferroni", "p_values": p_values, "adjusted": adjusted}
        )

    # Holm
    for p_values in [[0.01, 0.04, 0.03, 0.005], [0.5, 0.6], [0.001, 0.01, 0.05]]:
        n = len(p_values)
        # Sort by p-value
        indices = sorted(range(n), key=lambda i: p_values[i])
        adjusted_sorted = [0.0] * n
        running_max = 0.0
        for rank, orig_idx in enumerate(indices):
            multiplier = n - rank
            adj = min(p_values[orig_idx] * multiplier, 1.0)
            running_max = max(running_max, adj)
            adjusted_sorted[rank] = running_max
        # Map back
        result = [0.0] * n
        for rank, orig_idx in enumerate(indices):
            result[orig_idx] = adjusted_sorted[rank]
        cases.append({"type": "holm", "p_values": p_values, "adjusted": result})

    save_fixture("multiple_testing.json", {"cases": cases})


# --- Structural Breaks ---
def generate_structural_breaks():
    cases = []

    # ADF test
    np.random.seed(42)
    # Stationary AR(1) process
    n_adf = 200
    ar_series = [0.0] * n_adf
    for i in range(1, n_adf):
        noise = math.sin(i * 1.7) * 0.5
        ar_series[i] = 0.5 * ar_series[i - 1] + noise

    adf_val = adf_stat(ar_series, 1)
    cases.append(
        {
            "type": "adf",
            "series": ar_series,
            "max_lags": 1,
            "adf_stat": adf_val,
            "n_betas": 3,  # intercept + lagged level + 1 lag
        }
    )

    # Brown-Durbin-Evans
    residuals = [0.1] * 20 + [-0.5] * 20
    n_bde = len(residuals)
    mean = sum(residuals) / n_bde
    var_bde = sum((r - mean) ** 2 for r in residuals) / (n_bde - 1)
    std_bde = math.sqrt(var_bde)
    standardized = [(r - mean) / std_bde for r in residuals]
    cumulative = 0.0
    cusum = []
    for s in standardized:
        cumulative += s
        cusum.append(cumulative)
    scale = 1.0 / math.sqrt(n_bde)
    cusum = [c * scale for c in cusum]
    cases.append(
        {
            "type": "brown_durbin_evans",
            "residuals": residuals,
            "cusum": cusum,
            "critical": 1.358,
        }
    )

    # Chu-Stinchcombe-White
    log_prices = [4.6 + i * 0.001 + math.sin(i * 0.1) * 0.01 for i in range(50)]
    returns_csw = [log_prices[i + 1] - log_prices[i] for i in range(len(log_prices) - 1)]
    half_csw = len(returns_csw) // 2
    mean_ret_csw = sum(returns_csw[:half_csw]) / half_csw
    var_csw = sum((r - mean_ret_csw) ** 2 for r in returns_csw[:half_csw]) / (
        half_csw - 1
    )
    sigma_csw = math.sqrt(var_csw)
    csw_stats = []
    s_n = 0.0
    n_returns = len(returns_csw)
    for i, ret in enumerate(returns_csw):
        s_n += (ret - mean_ret_csw) / sigma_csw
        t_frac = (i + 1) / n_returns
        boundary = 1.96 * math.sqrt(i + 1) * (1.0 + 2.0 * t_frac)
        csw_stats.append(abs(s_n) / boundary)
    cases.append(
        {
            "type": "chu_stinchcombe_white",
            "log_prices": log_prices,
            "critical_value": 1.96,
            "stats": csw_stats,
        }
    )

    # GSADF (small series for tractability)
    np.random.seed(42)
    gsadf_series = list(np.cumsum(np.random.randn(30) * 0.5 + 0.01))
    gsadf_min_window = 15
    gsadf_max_lags = 1
    gsadf_stats = []
    n_gs = len(gsadf_series)
    for start in range(n_gs):
        for end in range(start + gsadf_min_window, n_gs + 1):
            window = gsadf_series[start:end]
            stat = adf_stat(window, gsadf_max_lags)
            if math.isfinite(stat):
                gsadf_stats.append(stat)
    gsadf_stat_val = max(gsadf_stats) if gsadf_stats else float("-inf")
    cases.append(
        {
            "type": "gsadf",
            "series": gsadf_series,
            "min_window": gsadf_min_window,
            "max_lags": gsadf_max_lags,
            "n_stats": len(gsadf_stats),
            "gsadf_stat": gsadf_stat_val,
        }
    )

    save_fixture("structural_breaks.json", {"cases": cases})


# --- Volatility Estimators ---
def generate_ohlc_bars(n, seed=42):
    """Generate deterministic OHLC bars."""
    np.random.seed(seed)
    bars = []
    price = 100.0
    for _ in range(n):
        open_price = price
        ret = np.random.randn() * 0.02
        close_price = open_price * (1 + ret)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.01)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.01)
        bars.append(
            {
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
            }
        )
        price = close_price
    return bars


def sample_variance(data):
    """Sample variance with (n-1) denominator, matching Rust."""
    n = len(data)
    if n <= 1:
        return 0.0
    mean = sum(data) / n
    return sum((v - mean) ** 2 for v in data) / (n - 1)


def generate_volatility_estimators():
    bars = generate_ohlc_bars(30)

    cases = []

    for window in [5, 10]:
        # Parkinson
        n = len(bars)
        hl_sq = [math.log(b["high"] / b["low"]) ** 2 for b in bars]
        coeff = 1.0 / (4.0 * window * math.log(2))
        pk_result = [None] * n
        s = sum(hl_sq[:window])
        pk_result[window - 1] = math.sqrt(coeff * s)
        for i in range(window, n):
            s += hl_sq[i] - hl_sq[i - window]
            pk_result[i] = math.sqrt(coeff * s)
        cases.append(
            {
                "type": "parkinson",
                "bars": bars,
                "window": window,
                "result": pk_result,
            }
        )

        # Garman-Klass
        co_coeff = 2.0 * math.log(2) - 1.0
        gk_terms = []
        for b in bars:
            hl = math.log(b["high"] / b["low"])
            co = math.log(b["close"] / b["open"])
            gk_terms.append(0.5 * hl * hl - co_coeff * co * co)
        gk_result = [None] * n
        s = sum(gk_terms[:window])
        variance = s / window
        gk_result[window - 1] = math.sqrt(variance) if variance > 0 else 0.0
        for i in range(window, n):
            s += gk_terms[i] - gk_terms[i - window]
            variance = s / window
            gk_result[i] = math.sqrt(variance) if variance > 0 else 0.0
        cases.append(
            {
                "type": "garman_klass",
                "bars": bars,
                "window": window,
                "result": gk_result,
            }
        )

        # Yang-Zhang
        yz_result = [None] * n
        if n >= window + 1:
            overnight = [0.0] * n
            cc_ret = [0.0] * n
            rs_term = [0.0] * n
            for i in range(1, n):
                overnight[i] = math.log(bars[i]["open"] / bars[i - 1]["close"])
                cc_ret[i] = math.log(bars[i]["close"] / bars[i - 1]["close"])
            for i in range(n):
                ho = math.log(bars[i]["high"] / bars[i]["open"])
                hc = math.log(bars[i]["high"] / bars[i]["close"])
                lo = math.log(bars[i]["low"] / bars[i]["open"])
                lc = math.log(bars[i]["low"] / bars[i]["close"])
                rs_term[i] = ho * hc + lo * lc

            if window == 1:
                k = 0.0
            else:
                k = 0.34 / (1.34 + (window + 1) / (window - 1))

            for offset in range(n - window):
                start = offset + 1
                end = offset + window + 1
                on_slice = overnight[start:end]
                cc_slice = cc_ret[start:end]
                rs_slice = rs_term[start:end]
                sigma_overnight_sq = sample_variance(on_slice)
                sigma_close_sq = sample_variance(cc_slice)
                sigma_rs_sq = sum(rs_slice) / window
                sigma_yz_sq = (
                    sigma_overnight_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq
                )
                yz_result[window + offset] = (
                    math.sqrt(sigma_yz_sq) if sigma_yz_sq > 0 else 0.0
                )

        cases.append(
            {
                "type": "yang_zhang",
                "bars": bars,
                "window": window,
                "result": yz_result,
            }
        )

    save_fixture("volatility_estimators.json", {"cases": cases})


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
    generate_backtesting_statistics()
    generate_drawdowns()
    generate_psr_dsr()
    generate_strategy_risk()
    generate_multiple_testing()
    generate_structural_breaks()
    generate_volatility_estimators()
    print("Done! All fixtures written to tests/fixtures/")
