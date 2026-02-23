from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def adf_test(series, max_lags):
    """
    Augmented Dickey-Fuller unit root test.

    Tests whether a time series is stationary by fitting an autoregressive model.

    Parameters
    ----------
    series : numpy.ndarray
        Time series to test.
    max_lags : int
        Maximum number of autoregressive lags.

    Returns
    -------
    tuple[float, numpy.ndarray]
        (adf_statistic, regression_coefficients).
    """
    ...

def amihud_lambda(returns, dollar_volumes):
    """
    Amihud illiquidity measure (AFML Ch. 19).

    Measures price impact as the average ratio of absolute return to
    dollar volume.

    Parameters
    ----------
    returns : numpy.ndarray
        Return series.
    dollar_volumes : numpy.ndarray
        Dollar volume series (same length as returns).

    Returns
    -------
    float
        Amihud lambda (higher = less liquid).
    """
    ...

def amihud_lambda_rolling(returns, dollar_volumes, window):
    """
    Rolling Amihud lambda over a sliding window.

    Parameters
    ----------
    returns : numpy.ndarray
        Return series.
    dollar_volumes : numpy.ndarray
        Dollar volume series.
    window : int
        Rolling window size.

    Returns
    -------
    numpy.ndarray
        Rolling Amihud lambda values.
    """
    ...

def angular_distance(x, y):
    """
    Angular distance derived from Pearson correlation.

    ``d = sqrt(0.5 * (1 - rho))``

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.

    Returns
    -------
    float
        Angular distance in [0, 1].
    """
    ...

def binary_encode(values):
    """
    Binary encode a real-valued series (above/below median).

    Parameters
    ----------
    values : numpy.ndarray
        Input series.

    Returns
    -------
    list[bool]
        True where value >= median, False otherwise.
    """
    ...

def brown_durbin_evans(residuals):
    """
    Brown-Durbin-Evans CUSUM test for parameter instability.

    Parameters
    ----------
    residuals : numpy.ndarray
        OLS regression residuals.

    Returns
    -------
    tuple[numpy.ndarray, float]
        (cusum_series, critical_value) — the CUSUM path and 5% significance boundary.
    """
    ...

def chu_stinchcombe_white(log_prices, critical_value):
    """
    Chu-Stinchcombe-White CUSUM test for structural breaks in log prices.

    Parameters
    ----------
    log_prices : numpy.ndarray
        Log price series.
    critical_value : float
        Significance threshold.

    Returns
    -------
    numpy.ndarray
        CUSUM statistic series.
    """
    ...

def cla_max_sharpe(expected_returns, cov):
    """
    CLA maximum Sharpe ratio portfolio.

    Parameters
    ----------
    expected_returns : numpy.ndarray
        Expected return per asset (n,).
    cov : numpy.ndarray
        Covariance matrix (n x n).

    Returns
    -------
    numpy.ndarray
        Max-Sharpe weights (n,).
    """
    ...

def cla_min_variance(cov):
    """
    Critical Line Algorithm (CLA) minimum-variance portfolio.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix (n x n).

    Returns
    -------
    numpy.ndarray
        Minimum-variance weights (n,).
    """
    ...

def cluster_kmeans_base(corr, max_clusters=None, min_clusters=None, n_init=None, seed=None):
    """
    Base-level K-means clustering over a range of k values.

    Tries multiple cluster counts and selects the one with the best
    silhouette score.

    Parameters
    ----------
    corr : numpy.ndarray
        Correlation matrix.
    max_clusters : int, optional
        Maximum k to try.
    min_clusters : int, optional
        Minimum k to try.
    n_init : int, optional
        Initializations per k.
    seed : int, optional
        Random seed.

    Returns
    -------
    OncResult
        Labels, silhouette score, and optimal cluster count.
    """
    ...

def cluster_kmeans_top(corr, max_clusters=None, min_clusters=None, n_init=None, seed=None):
    """
    Top-level ONC (Optimal Number of Clusters) algorithm (AFML Ch. 16).

    Two-step approach: first clusters, then re-clusters to find the
    optimal grouping.

    Parameters
    ----------
    corr : numpy.ndarray
        Correlation matrix.
    max_clusters : int, optional
        Maximum k.
    min_clusters : int, optional
        Minimum k.
    n_init : int, optional
        Initializations per k.
    seed : int, optional
        Random seed.

    Returns
    -------
    OncResult
        Labels, silhouette score, and optimal cluster count.
    """
    ...

def compare_allocations(returns, n_simulations, seed):
    """
    Monte Carlo comparison of HRP, CLA, and IVP allocation methods.

    Simulates random correlation matrices and compares out-of-sample
    Sharpe ratios and variances.

    Parameters
    ----------
    returns : numpy.ndarray
        Return matrix (n_periods, n_assets).
    n_simulations : int
        Number of Monte Carlo trials.
    seed : int
        Random seed.

    Returns
    -------
    AllocationComparison
        Sharpe ratios and variances for each method.
    """
    ...

def corr_to_cov(corr, std):
    """
    Convert a correlation matrix + standard deviations back to a covariance matrix.

    Parameters
    ----------
    corr : numpy.ndarray
        Correlation matrix (n x n).
    std : numpy.ndarray
        Standard deviations (n,).

    Returns
    -------
    numpy.ndarray
        Covariance matrix (n x n).
    """
    ...

def corwin_schultz_spread(highs, lows):
    """
    Corwin-Schultz spread estimator from high-low prices.

    Estimates the bid-ask spread from consecutive high-low price pairs.

    Parameters
    ----------
    highs : numpy.ndarray
        High price series.
    lows : numpy.ndarray
        Low price series.

    Returns
    -------
    numpy.ndarray
        Estimated spread series.
    """
    ...

def cov_to_corr(cov):
    """
    Convert a covariance matrix to a correlation matrix + standard deviations.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix (n x n).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (correlation_matrix, std_devs).
    """
    ...

def denoise_corr(corr, q, bandwidth=None, shrinkage=False, alpha=None):
    """
    Denoise a correlation matrix using Random Matrix Theory (AFML Ch. 2).

    Shrinks eigenvalues below the Marcenko-Pastur bound toward their average,
    removing noise while preserving the signal.

    Parameters
    ----------
    corr : numpy.ndarray
        Empirical correlation matrix.
    q : float
        Ratio T/N.
    bandwidth : float, optional
        KDE bandwidth for eigenvalue fitting. Auto-selected if None.
    shrinkage : bool, default False
        Use shrinkage-based denoising instead of constant residual eigenvalue.
    alpha : float, optional
        Shrinkage intensity (0 to 1). Only used if ``shrinkage=True``.

    Returns
    -------
    numpy.ndarray
        Denoised correlation matrix.
    """
    ...

def denoise_cov(cov, q, bandwidth=None):
    """
    Denoise a covariance matrix using RMT.

    Converts to correlation, denoises, then converts back.

    Parameters
    ----------
    cov : numpy.ndarray
        Empirical covariance matrix.
    q : float
        Ratio T/N.
    bandwidth : float, optional
        KDE bandwidth.

    Returns
    -------
    numpy.ndarray
        Denoised covariance matrix.
    """
    ...

def dependence_matrix(data, method):
    """
    Compute a pairwise dependence matrix using the specified method.

    Parameters
    ----------
    data : numpy.ndarray
        Data matrix (n_observations, n_variables).
    method : str
        One of: ``"pearson"``, ``"spearman"``, ``"distance_correlation"``,
        ``"mutual_information"``, ``"variation_of_information"``.

    Returns
    -------
    numpy.ndarray
        Symmetric dependence matrix (n_variables x n_variables).
    """
    ...

def detone_corr(corr, n_components):
    """
    Remove the market component from a correlation matrix (detoning).

    Subtracts the first ``n_components`` principal components to
    remove common factors (e.g. the market mode).

    Parameters
    ----------
    corr : numpy.ndarray
        Correlation matrix.
    n_components : int
        Number of leading eigenvectors to remove (usually 1 for market mode).

    Returns
    -------
    numpy.ndarray
        Detoned correlation matrix.
    """
    ...

def distance_correlation(x, y):
    """
    Distance correlation — a measure of dependence for non-linear relationships.

    Unlike Pearson correlation, distance correlation is zero if and only if
    the variables are independent.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.

    Returns
    -------
    float
        Distance correlation in [0, 1].
    """
    ...

def distance_matrix(corr, metric):
    """
    Convert a correlation matrix to a distance matrix.

    Parameters
    ----------
    corr : numpy.ndarray
        Correlation matrix.
    metric : str
        One of: ``"angular"``, ``"absolute_angular"``, ``"squared_angular"``.

    Returns
    -------
    numpy.ndarray
        Distance matrix (n x n).
    """
    ...

def entropy_implied_vol(entropy):
    """
    Implied volatility from Gaussian entropy.

    Inverts the Gaussian entropy formula to recover the standard deviation.

    Parameters
    ----------
    entropy : float
        Gaussian entropy value.

    Returns
    -------
    float
        Implied volatility (standard deviation).
    """
    ...

def fit_kde(observations, bandwidth, eval_points):
    """
    Kernel Density Estimation (KDE) for eigenvalue distribution fitting.

    Parameters
    ----------
    observations : numpy.ndarray
        Observed eigenvalues.
    bandwidth : float
        Gaussian kernel bandwidth.
    eval_points : numpy.ndarray
        Points at which to evaluate the KDE.

    Returns
    -------
    numpy.ndarray
        KDE density values at the evaluation points.
    """
    ...

def gaussian_entropy(variance):
    """
    Gaussian entropy for a given variance.

    ``H = 0.5 * log(2 * pi * e * variance)``

    Parameters
    ----------
    variance : float
        Variance of the Gaussian distribution.

    Returns
    -------
    float
        Differential entropy in nats.
    """
    ...

def get_feature_clusters(data, max_clusters=None, seed=None):
    """
    Cluster features using ONC on their correlation structure.

    Parameters
    ----------
    data : numpy.ndarray
        Data matrix (n_samples, n_features).
    max_clusters : int, optional
        Maximum number of feature clusters.
    seed : int, optional
        Random seed.

    Returns
    -------
    OncResult
        Feature cluster labels and quality metrics.
    """
    ...

def gnpr_distance(x, y, theta, n_bins=None):
    """
    GNPR (Generalized Non-Parametric Rank) distance.

    Combines rank correlation with an information-theoretic component.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.
    theta : float
        Co-movement threshold.
    n_bins : int, optional
        Number of bins for the information component.

    Returns
    -------
    float
        GNPR distance.
    """
    ...

def gpr_distance(x, y, theta):
    """
    GPR (Gerber-Podolskij-Reisenhofer) distance with threshold.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.
    theta : float
        Co-movement threshold.

    Returns
    -------
    float
        GPR distance.
    """
    ...

def gsadf(series, min_window, max_lags):
    """
    Generalized SADF (GSADF) test series (AFML Ch. 17).

    Tests for explosive behavior using flexible start/end windows,
    providing higher power than SADF for detecting multiple bubbles.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.
    min_window : int
        Minimum regression window.
    max_lags : int
        Maximum lags.

    Returns
    -------
    numpy.ndarray
        GSADF statistic series.
    """
    ...

def gsadf_stat(series, min_window, max_lags):
    """
    Generalized SADF scalar statistic.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.
    min_window : int
        Minimum regression window.
    max_lags : int
        Maximum lags.

    Returns
    -------
    float
        GSADF test statistic.
    """
    ...

def hasbrouck_lambda(returns, trade_signs, n_iterations, seed):
    """
    Hasbrouck's lambda via Gibbs sampling (AFML Ch. 19).

    Estimates permanent price impact accounting for trade sign uncertainty
    using a Bayesian approach.

    Parameters
    ----------
    returns : numpy.ndarray
        Return series.
    trade_signs : numpy.ndarray
        Signed trade indicators (+1 or -1).
    n_iterations : int
        Number of Gibbs sampling iterations.
    seed : int
        Random seed.

    Returns
    -------
    float
        Hasbrouck lambda estimate.
    """
    ...

def hrp_weights(returns):
    """
    Hierarchical Risk Parity (HRP) portfolio weights (AFML Ch. 16).

    Uses hierarchical clustering on the correlation matrix to build
    a diversified portfolio that is more stable than mean-variance.

    Parameters
    ----------
    returns : numpy.ndarray
        Return matrix (n_periods, n_assets).

    Returns
    -------
    numpy.ndarray
        Portfolio weights (n_assets,), sums to 1.
    """
    ...

def inverse_variance_weights(cov):
    """
    Inverse Variance Portfolio (IVP) weights.

    Weights each asset inversely proportional to its variance (diagonal of cov).

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix (n x n).

    Returns
    -------
    numpy.ndarray
        IVP weights (n,), sums to 1.
    """
    ...

def kmeans(data, k, max_iter=300, n_init=10, seed=42):
    """
    K-means clustering with multiple initializations.

    Parameters
    ----------
    data : numpy.ndarray
        Data matrix (n_samples, n_features).
    k : int
        Number of clusters.
    max_iter : int, default 300
        Maximum iterations per run.
    n_init : int, default 10
        Number of random initializations (best is kept).
    seed : int, default 42
        Random seed.

    Returns
    -------
    KMeansResult
        Cluster labels, centroids, and iteration count.
    """
    ...

def kontoyiannis_entropy(sequence, window):
    """
    Kontoyiannis entropy estimator using longest-match lengths (AFML Ch. 18).

    A non-parametric entropy estimator based on how far back one must
    look to find a match for each substring.

    Parameters
    ----------
    sequence : list[int]
        Discrete symbol sequence.
    window : int
        Maximum look-back window.

    Returns
    -------
    float
        Estimated entropy rate.
    """
    ...

def kyle_lambda(returns, signed_volume):
    """
    Kyle's lambda — price impact from signed order flow (AFML Ch. 19).

    Regresses returns on signed volume to estimate the permanent price
    impact of trades.

    Parameters
    ----------
    returns : numpy.ndarray
        Return series.
    signed_volume : numpy.ndarray
        Net signed volume (buy - sell).

    Returns
    -------
    float
        Kyle lambda coefficient.
    """
    ...

def lempel_ziv_complexity(binary_string):
    """
    Lempel-Ziv complexity of a binary string (AFML Ch. 18).

    Counts the number of distinct substrings encountered during a
    sequential parse — a measure of randomness/compressibility.

    Parameters
    ----------
    binary_string : list[bool]
        Binary sequence.

    Returns
    -------
    int
        Number of distinct patterns (Lempel-Ziv complexity).
    """
    ...

def marcenko_pastur_pdf(var, q, pts):
    """
    Marcenko-Pastur probability density function.

    Theoretical distribution of eigenvalues for a random correlation matrix
    with ratio q = T/N.

    Parameters
    ----------
    var : float
        Variance of the random matrix entries.
    q : float
        Ratio T/N (observations / variables).
    pts : int
        Number of evaluation points.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (x_values, pdf_values).
    """
    ...

def mutual_information(x, y, n_bins=None, normalize=False):
    """
    Mutual information between two continuous variables.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.
    n_bins : int, optional
        Number of histogram bins. Auto-selected if None.
    normalize : bool, default False
        If True, normalize to [0, 1] range.

    Returns
    -------
    float
        Mutual information (non-negative).
    """
    ...

def optimal_portfolio(cov, mu=None):
    """
    Optimal portfolio weights from a (denoised) covariance matrix.

    Computes the minimum-variance or max-Sharpe portfolio.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix.
    mu : numpy.ndarray, optional
        Expected returns. If None, computes minimum-variance portfolio.

    Returns
    -------
    numpy.ndarray
        Portfolio weights (sums to 1).
    """
    ...

def optimal_transport_dependence(x, y):
    """
    Optimal transport dependence measure.

    Based on the Wasserstein distance between joint and product marginal
    distributions.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.

    Returns
    -------
    float
        Optimal transport dependence (non-negative).
    """
    ...

def plugin_entropy(sequence, num_symbols):
    """
    Plug-in (maximum likelihood) entropy estimator.

    Estimates entropy from a discrete symbol sequence using empirical frequencies.

    Parameters
    ----------
    sequence : list[int]
        Discrete symbol sequence.
    num_symbols : int
        Number of distinct symbols in the alphabet.

    Returns
    -------
    float
        Estimated entropy.
    """
    ...

def quantile_encode(values, num_bins):
    """
    Quantile-based discretization of a continuous series.

    Parameters
    ----------
    values : numpy.ndarray
        Input series.
    num_bins : int
        Number of quantile bins.

    Returns
    -------
    list[int]
        Bin index (0 to num_bins-1) for each value.
    """
    ...

def roll_spread(prices):
    """
    Roll model bid-ask spread estimator.

    Estimates the effective spread from the autocovariance of price changes.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series.

    Returns
    -------
    float
        Estimated bid-ask spread.
    """
    ...

def roll_spread_rolling(prices, window):
    """
    Rolling Roll model spread estimate.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series.
    window : int
        Rolling window size.

    Returns
    -------
    numpy.ndarray
        Rolling spread estimates.
    """
    ...

def sadf(series, min_window, max_lags):
    """
    Supremum Augmented Dickey-Fuller (SADF) test series (AFML Ch. 17).

    Computes a sequence of ADF statistics with expanding windows starting
    from ``min_window``.

    Parameters
    ----------
    series : numpy.ndarray
        Time series (e.g. log prices).
    min_window : int
        Minimum regression window.
    max_lags : int
        Maximum lags per ADF regression.

    Returns
    -------
    numpy.ndarray
        SADF statistic series.
    """
    ...

def sadf_stat(series, min_window, max_lags):
    """
    Supremum ADF scalar statistic — the maximum of the SADF series.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.
    min_window : int
        Minimum regression window.
    max_lags : int
        Maximum lags.

    Returns
    -------
    float
        SADF test statistic.
    """
    ...

def shannon_entropy(probs):
    """
    Shannon entropy from a probability distribution.

    Parameters
    ----------
    probs : numpy.ndarray
        Probability vector (should sum to 1).

    Returns
    -------
    float
        Shannon entropy in nats (natural log).
    """
    ...

def sigma_encode(values, num_bands):
    """
    Sigma-based encoding using standard deviation bands.

    Parameters
    ----------
    values : numpy.ndarray
        Input series.
    num_bands : int
        Number of sigma bands on each side of the mean.

    Returns
    -------
    list[int]
        Band index for each value.
    """
    ...

def silhouette_score(data, labels):
    """
    Silhouette score measuring clustering quality.

    Ranges from -1 (poor) to +1 (excellent).

    Parameters
    ----------
    data : numpy.ndarray
        Data matrix (n_samples, n_features).
    labels : list[int]
        Cluster labels for each sample.

    Returns
    -------
    float
        Mean silhouette score.
    """
    ...

def sm_exp(series):
    """
    Sub/super-martingale test with exponential kernel.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.

    Returns
    -------
    float
        Test statistic.
    """
    ...

def sm_poly(series, degree):
    """
    Sub/super-martingale test with polynomial kernel.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.
    degree : int
        Polynomial degree for the test.

    Returns
    -------
    float
        Test statistic.
    """
    ...

def sm_power(series, power):
    """
    Sub/super-martingale test with power kernel.

    Parameters
    ----------
    series : numpy.ndarray
        Time series.
    power : float
        Power exponent for the kernel.

    Returns
    -------
    float
        Test statistic.
    """
    ...

def spearmans_rho(x, y):
    """
    Spearman's rank correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.

    Returns
    -------
    float
        Spearman's rho in [-1, 1].
    """
    ...

def tick_rule_classify(prices):
    """
    Classify trades using the tick rule.

    Assigns +1 (uptick), -1 (downtick), or 0 (no change) to each trade.

    Parameters
    ----------
    prices : numpy.ndarray
        Trade price series.

    Returns
    -------
    numpy.ndarray
        Trade sign series (+1, -1, or 0).
    """
    ...

def variation_of_information(x, y, n_bins=None, normalize=False):
    """
    Variation of information — a metric-space distance based on entropy.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.
    n_bins : int, optional
        Number of histogram bins.
    normalize : bool, default False
        If True, normalize to [0, 1] range.

    Returns
    -------
    float
        Variation of information (non-negative).
    """
    ...

def vpin(volumes, prices, bucket_size, n_buckets):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    Estimates the probability of informed trading from volume-bucketed data.

    Parameters
    ----------
    volumes : numpy.ndarray
        Volume series.
    prices : numpy.ndarray
        Price series.
    bucket_size : float
        Volume per bucket.
    n_buckets : int
        Number of buckets for the rolling VPIN estimate.

    Returns
    -------
    numpy.ndarray
        VPIN estimates at each bucket boundary.
    """
    ...
