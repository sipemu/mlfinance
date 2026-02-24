use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;
use crate::types::*;

// ═══════════════════════════════════════════════════════════════════════════
// Structural breaks
// ═══════════════════════════════════════════════════════════════════════════

/// Augmented Dickey-Fuller unit root test.
///
/// Tests whether a time series is stationary by fitting an autoregressive model.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series to test.
/// max_lags : int
///     Maximum number of autoregressive lags.
///
/// Returns
/// -------
/// tuple[float, numpy.ndarray]
///     (adf_statistic, regression_coefficients).
#[pyfunction]
fn adf_test(
    py: Python<'_>,
    series: PyReadonlyArray1<'_, f64>,
    max_lags: usize,
) -> (f64, Py<PyArray1<f64>>) {
    let s = py_to_vec(series);
    let (stat, coeffs) = mlfinance::features::structural_breaks::adf::adf_test(&s, max_lags);
    (stat, vec_to_py_array(py, coeffs))
}

/// Supremum Augmented Dickey-Fuller (SADF) test series (AFML Ch. 17).
///
/// Computes a sequence of ADF statistics with expanding windows starting
/// from ``min_window``.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series (e.g. log prices).
/// min_window : int
///     Minimum regression window.
/// max_lags : int
///     Maximum lags per ADF regression.
///
/// Returns
/// -------
/// numpy.ndarray
///     SADF statistic series.
#[pyfunction]
fn sadf(
    py: Python<'_>,
    series: PyReadonlyArray1<'_, f64>,
    min_window: usize,
    max_lags: usize,
) -> Py<PyArray1<f64>> {
    let s = py_to_vec(series);
    let result = mlfinance::features::structural_breaks::sadf::sadf(&s, min_window, max_lags);
    vec_to_py_array(py, result)
}

/// Supremum ADF scalar statistic — the maximum of the SADF series.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series.
/// min_window : int
///     Minimum regression window.
/// max_lags : int
///     Maximum lags.
///
/// Returns
/// -------
/// float
///     SADF test statistic.
#[pyfunction]
fn sadf_stat(series: PyReadonlyArray1<'_, f64>, min_window: usize, max_lags: usize) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sadf::sadf_stat(&s, min_window, max_lags)
}

/// Generalized SADF (GSADF) test series (AFML Ch. 17).
///
/// Tests for explosive behavior using flexible start/end windows,
/// providing higher power than SADF for detecting multiple bubbles.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series.
/// min_window : int
///     Minimum regression window.
/// max_lags : int
///     Maximum lags.
///
/// Returns
/// -------
/// numpy.ndarray
///     GSADF statistic series.
#[pyfunction]
fn gsadf(
    py: Python<'_>,
    series: PyReadonlyArray1<'_, f64>,
    min_window: usize,
    max_lags: usize,
) -> Py<PyArray1<f64>> {
    let s = py_to_vec(series);
    let result = mlfinance::features::structural_breaks::gsadf::gsadf(&s, min_window, max_lags);
    vec_to_py_array(py, result)
}

/// Generalized SADF scalar statistic.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series.
/// min_window : int
///     Minimum regression window.
/// max_lags : int
///     Maximum lags.
///
/// Returns
/// -------
/// float
///     GSADF test statistic.
#[pyfunction]
fn gsadf_stat(series: PyReadonlyArray1<'_, f64>, min_window: usize, max_lags: usize) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::gsadf::gsadf_stat(&s, min_window, max_lags)
}

/// Brown-Durbin-Evans CUSUM test for parameter instability.
///
/// Parameters
/// ----------
/// residuals : numpy.ndarray
///     OLS regression residuals.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, float]
///     (cusum_series, critical_value) — the CUSUM path and 5% significance boundary.
#[pyfunction]
fn brown_durbin_evans(
    py: Python<'_>,
    residuals: PyReadonlyArray1<'_, f64>,
) -> (Py<PyArray1<f64>>, f64) {
    let r = py_to_vec(residuals);
    let (cusum, sig) = mlfinance::features::structural_breaks::cusum_tests::brown_durbin_evans(&r);
    (vec_to_py_array(py, cusum), sig)
}

/// Chu-Stinchcombe-White CUSUM test for structural breaks in log prices.
///
/// Parameters
/// ----------
/// log_prices : numpy.ndarray
///     Log price series.
/// critical_value : float
///     Significance threshold.
///
/// Returns
/// -------
/// numpy.ndarray
///     CUSUM statistic series.
#[pyfunction]
fn chu_stinchcombe_white(
    py: Python<'_>,
    log_prices: PyReadonlyArray1<'_, f64>,
    critical_value: f64,
) -> Py<PyArray1<f64>> {
    let p = py_to_vec(log_prices);
    let result = mlfinance::features::structural_breaks::cusum_tests::chu_stinchcombe_white(
        &p,
        critical_value,
    );
    vec_to_py_array(py, result)
}

/// Sub/super-martingale test with polynomial kernel.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series.
/// degree : int
///     Polynomial degree for the test.
///
/// Returns
/// -------
/// float
///     Test statistic.
#[pyfunction]
fn sm_poly(series: PyReadonlyArray1<'_, f64>, degree: usize) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sub_super_martingale::sm_poly(&s, degree)
}

/// Sub/super-martingale test with exponential kernel.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series.
///
/// Returns
/// -------
/// float
///     Test statistic.
#[pyfunction]
fn sm_exp(series: PyReadonlyArray1<'_, f64>) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sub_super_martingale::sm_exp(&s)
}

/// Sub/super-martingale test with power kernel.
///
/// Parameters
/// ----------
/// series : numpy.ndarray
///     Time series.
/// power : float
///     Power exponent for the kernel.
///
/// Returns
/// -------
/// float
///     Test statistic.
#[pyfunction]
fn sm_power(series: PyReadonlyArray1<'_, f64>, power: f64) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sub_super_martingale::sm_power(&s, power)
}

// ═══════════════════════════════════════════════════════════════════════════
// Entropy
// ═══════════════════════════════════════════════════════════════════════════

/// Binary encode a real-valued series (above/below median).
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input series.
///
/// Returns
/// -------
/// list[bool]
///     True where value >= median, False otherwise.
#[pyfunction]
fn binary_encode(py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyAny>> {
    let v = py_to_vec(values);
    let result = mlfinance::features::entropy::encoding::binary_encode(&v);
    vec_bool_to_list(py, result)
}

/// Quantile-based discretization of a continuous series.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input series.
/// num_bins : int
///     Number of quantile bins.
///
/// Returns
/// -------
/// list[int]
///     Bin index (0 to num_bins-1) for each value.
#[pyfunction]
fn quantile_encode(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    num_bins: usize,
) -> PyResult<Py<PyAny>> {
    let v = py_to_vec(values);
    let result = mlfinance::features::entropy::encoding::quantile_encode(&v, num_bins);
    vec_usize_to_list(py, result)
}

/// Sigma-based encoding using standard deviation bands.
///
/// Parameters
/// ----------
/// values : numpy.ndarray
///     Input series.
/// num_bands : int
///     Number of sigma bands on each side of the mean.
///
/// Returns
/// -------
/// list[int]
///     Band index for each value.
#[pyfunction]
fn sigma_encode(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    num_bands: usize,
) -> PyResult<Py<PyAny>> {
    let v = py_to_vec(values);
    let result = mlfinance::features::entropy::encoding::sigma_encode(&v, num_bands);
    vec_usize_to_list(py, result)
}

/// Shannon entropy from a probability distribution.
///
/// Parameters
/// ----------
/// probs : numpy.ndarray
///     Probability vector (should sum to 1).
///
/// Returns
/// -------
/// float
///     Shannon entropy in bits (log base 2).
#[pyfunction]
fn shannon_entropy(probs: PyReadonlyArray1<'_, f64>) -> f64 {
    let p = py_to_vec(probs);
    mlfinance::features::entropy::shannon::shannon_entropy(&p)
}

/// Plug-in (maximum likelihood) entropy estimator.
///
/// Estimates entropy from a discrete symbol sequence using empirical frequencies.
///
/// Parameters
/// ----------
/// sequence : list[int]
///     Discrete symbol sequence.
/// num_symbols : int
///     Number of distinct symbols in the alphabet.
///
/// Returns
/// -------
/// float
///     Estimated entropy.
#[pyfunction]
fn plugin_entropy(sequence: Vec<usize>, num_symbols: usize) -> f64 {
    mlfinance::features::entropy::plugin::plugin_entropy(&sequence, num_symbols)
}

/// Lempel-Ziv complexity of a binary string (AFML Ch. 18).
///
/// Counts the number of distinct substrings encountered during a
/// sequential parse — a measure of randomness/compressibility.
///
/// Parameters
/// ----------
/// binary_string : list[bool]
///     Binary sequence.
///
/// Returns
/// -------
/// int
///     Number of distinct patterns (Lempel-Ziv complexity).
#[pyfunction]
fn lempel_ziv_complexity(binary_string: Vec<bool>) -> usize {
    mlfinance::features::entropy::lempel_ziv::lempel_ziv_complexity(&binary_string)
}

/// Kontoyiannis entropy estimator using longest-match lengths (AFML Ch. 18).
///
/// A non-parametric entropy estimator based on how far back one must
/// look to find a match for each substring.
///
/// Parameters
/// ----------
/// sequence : list[int]
///     Discrete symbol sequence.
/// window : int
///     Maximum look-back window.
///
/// Returns
/// -------
/// float
///     Estimated entropy rate.
#[pyfunction]
fn kontoyiannis_entropy(sequence: Vec<usize>, window: usize) -> f64 {
    mlfinance::features::entropy::kontoyiannis::kontoyiannis_entropy(&sequence, window)
}

/// Gaussian entropy for a given variance.
///
/// ``H = 0.5 * log2(2 * pi * e * variance)``
///
/// Parameters
/// ----------
/// variance : float
///     Variance of the Gaussian distribution.
///
/// Returns
/// -------
/// float
///     Differential entropy in bits.
#[pyfunction]
fn gaussian_entropy(variance: f64) -> f64 {
    mlfinance::features::entropy::gaussian_entropy::gaussian_entropy(variance)
}

/// Implied volatility from Gaussian entropy.
///
/// Inverts the Gaussian entropy formula to recover the standard deviation.
///
/// Parameters
/// ----------
/// entropy : float
///     Gaussian entropy value.
///
/// Returns
/// -------
/// float
///     Implied volatility (standard deviation).
#[pyfunction]
fn entropy_implied_vol(entropy: f64) -> f64 {
    mlfinance::features::entropy::gaussian_entropy::entropy_implied_vol(entropy)
}

// ═══════════════════════════════════════════════════════════════════════════
// Microstructure
// ═══════════════════════════════════════════════════════════════════════════

/// Amihud illiquidity measure (AFML Ch. 19).
///
/// Measures price impact as the average ratio of absolute return to
/// dollar volume.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return series.
/// dollar_volumes : numpy.ndarray
///     Dollar volume series (same length as returns).
///
/// Returns
/// -------
/// float
///     Amihud lambda (higher = less liquid).
#[pyfunction]
fn amihud_lambda(
    returns: PyReadonlyArray1<'_, f64>,
    dollar_volumes: PyReadonlyArray1<'_, f64>,
) -> f64 {
    let r = py_to_vec(returns);
    let dv = py_to_vec(dollar_volumes);
    mlfinance::features::microstructure::amihud_lambda::amihud_lambda(&r, &dv)
}

/// Rolling Amihud lambda over a sliding window.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return series.
/// dollar_volumes : numpy.ndarray
///     Dollar volume series.
/// window : int
///     Rolling window size.
///
/// Returns
/// -------
/// numpy.ndarray
///     Rolling Amihud lambda values.
#[pyfunction]
fn amihud_lambda_rolling(
    py: Python<'_>,
    returns: PyReadonlyArray1<'_, f64>,
    dollar_volumes: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> Py<PyArray1<f64>> {
    let r = py_to_vec(returns);
    let dv = py_to_vec(dollar_volumes);
    let result =
        mlfinance::features::microstructure::amihud_lambda::amihud_lambda_rolling(&r, &dv, window);
    vec_to_py_array(py, result)
}

/// Kyle's lambda — price impact from signed order flow (AFML Ch. 19).
///
/// Regresses returns on signed volume to estimate the permanent price
/// impact of trades.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return series.
/// signed_volume : numpy.ndarray
///     Net signed volume (buy - sell).
///
/// Returns
/// -------
/// float
///     Kyle lambda coefficient.
#[pyfunction]
fn kyle_lambda(
    returns: PyReadonlyArray1<'_, f64>,
    signed_volume: PyReadonlyArray1<'_, f64>,
) -> f64 {
    let r = py_to_vec(returns);
    let sv = py_to_vec(signed_volume);
    mlfinance::features::microstructure::kyle_lambda::kyle_lambda(&r, &sv)
}

/// Hasbrouck's lambda via Gibbs sampling (AFML Ch. 19).
///
/// Estimates permanent price impact accounting for trade sign uncertainty
/// using a Bayesian approach.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return series.
/// trade_signs : numpy.ndarray
///     Signed trade indicators (+1 or -1).
/// n_iterations : int
///     Number of Gibbs sampling iterations.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// float
///     Hasbrouck lambda estimate.
#[pyfunction]
fn hasbrouck_lambda(
    returns: PyReadonlyArray1<'_, f64>,
    trade_signs: PyReadonlyArray1<'_, f64>,
    n_iterations: usize,
    seed: u64,
) -> f64 {
    let r = py_to_vec(returns);
    let ts = py_to_vec(trade_signs);
    mlfinance::features::microstructure::hasbrouck_lambda::hasbrouck_lambda(
        &r,
        &ts,
        n_iterations,
        seed,
    )
}

/// Roll model bid-ask spread estimator.
///
/// Estimates the effective spread from the autocovariance of price changes.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series.
///
/// Returns
/// -------
/// float
///     Estimated bid-ask spread.
#[pyfunction]
fn roll_spread(prices: PyReadonlyArray1<'_, f64>) -> f64 {
    let p = py_to_vec(prices);
    mlfinance::features::microstructure::roll_model::roll_spread(&p)
}

/// Rolling Roll model spread estimate.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Price series.
/// window : int
///     Rolling window size.
///
/// Returns
/// -------
/// numpy.ndarray
///     Rolling spread estimates.
#[pyfunction]
fn roll_spread_rolling(
    py: Python<'_>,
    prices: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> Py<PyArray1<f64>> {
    let p = py_to_vec(prices);
    let result = mlfinance::features::microstructure::roll_model::roll_spread_rolling(&p, window);
    vec_to_py_array(py, result)
}

/// Corwin-Schultz spread estimator from high-low prices.
///
/// Estimates the bid-ask spread from consecutive high-low price pairs.
///
/// Parameters
/// ----------
/// highs : numpy.ndarray
///     High price series.
/// lows : numpy.ndarray
///     Low price series.
///
/// Returns
/// -------
/// numpy.ndarray
///     Estimated spread series.
#[pyfunction]
fn corwin_schultz_spread(
    py: Python<'_>,
    highs: PyReadonlyArray1<'_, f64>,
    lows: PyReadonlyArray1<'_, f64>,
) -> Py<PyArray1<f64>> {
    let h = py_to_vec(highs);
    let l = py_to_vec(lows);
    let result = mlfinance::features::microstructure::corwin_schultz::corwin_schultz_spread(&h, &l);
    vec_to_py_array(py, result)
}

/// Volume-Synchronized Probability of Informed Trading (VPIN).
///
/// Estimates the probability of informed trading from volume-bucketed data.
///
/// Parameters
/// ----------
/// volumes : numpy.ndarray
///     Volume series.
/// prices : numpy.ndarray
///     Price series.
/// bucket_size : float
///     Volume per bucket.
/// n_buckets : int
///     Number of buckets for the rolling VPIN estimate.
///
/// Returns
/// -------
/// numpy.ndarray
///     VPIN estimates at each bucket boundary.
#[pyfunction]
fn vpin(
    py: Python<'_>,
    volumes: PyReadonlyArray1<'_, f64>,
    prices: PyReadonlyArray1<'_, f64>,
    bucket_size: f64,
    n_buckets: usize,
) -> Py<PyArray1<f64>> {
    let v = py_to_vec(volumes);
    let p = py_to_vec(prices);
    let result = mlfinance::features::microstructure::vpin::vpin(&v, &p, bucket_size, n_buckets);
    vec_to_py_array(py, result)
}

/// Classify trades using the tick rule.
///
/// Assigns +1 (uptick), -1 (downtick), or 0 (no change) to each trade.
///
/// Parameters
/// ----------
/// prices : numpy.ndarray
///     Trade price series.
///
/// Returns
/// -------
/// numpy.ndarray
///     Trade sign series (+1, -1, or 0).
#[pyfunction]
fn tick_rule_classify(py: Python<'_>, prices: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let p = py_to_vec(prices);
    let result = mlfinance::features::microstructure::tick_rule::tick_rule_classify(&p);
    vec_to_py_array(py, result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Denoising (RMT)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a covariance matrix to a correlation matrix + standard deviations.
///
/// Parameters
/// ----------
/// cov : numpy.ndarray
///     Covariance matrix (n x n).
///
/// Returns
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     (correlation_matrix, std_devs).
#[pyfunction]
fn cov_to_corr(
    py: Python<'_>,
    cov: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let m = py_to_array2(cov);
    let (corr, std) = to_pyresult(mlfinance::features::denoising::rmt::cov_to_corr(&m))?;
    Ok((array2_to_py(py, corr), array1_to_py(py, std)))
}

/// Convert a correlation matrix + standard deviations back to a covariance matrix.
///
/// Parameters
/// ----------
/// corr : numpy.ndarray
///     Correlation matrix (n x n).
/// std : numpy.ndarray
///     Standard deviations (n,).
///
/// Returns
/// -------
/// numpy.ndarray
///     Covariance matrix (n x n).
#[pyfunction]
fn corr_to_cov(
    py: Python<'_>,
    corr: PyReadonlyArray2<'_, f64>,
    std: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let c = py_to_array2(corr);
    let s = py_to_array1(std);
    let result = to_pyresult(mlfinance::features::denoising::rmt::corr_to_cov(&c, &s))?;
    Ok(array2_to_py(py, result))
}

/// Marcenko-Pastur probability density function.
///
/// Theoretical distribution of eigenvalues for a random correlation matrix
/// with ratio q = T/N.
///
/// Parameters
/// ----------
/// var : float
///     Variance of the random matrix entries.
/// q : float
///     Ratio T/N (observations / variables).
/// pts : int
///     Number of evaluation points.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     (x_values, pdf_values).
#[pyfunction]
fn marcenko_pastur_pdf(
    py: Python<'_>,
    var: f64,
    q: f64,
    pts: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (x, pdf) = to_pyresult(mlfinance::features::denoising::rmt::marcenko_pastur_pdf(
        var, q, pts,
    ))?;
    Ok((array1_to_py(py, x), array1_to_py(py, pdf)))
}

/// Kernel Density Estimation (KDE) for eigenvalue distribution fitting.
///
/// Parameters
/// ----------
/// observations : numpy.ndarray
///     Observed eigenvalues.
/// bandwidth : float
///     Gaussian kernel bandwidth.
/// eval_points : numpy.ndarray
///     Points at which to evaluate the KDE.
///
/// Returns
/// -------
/// numpy.ndarray
///     KDE density values at the evaluation points.
#[pyfunction]
fn fit_kde(
    py: Python<'_>,
    observations: PyReadonlyArray1<'_, f64>,
    bandwidth: f64,
    eval_points: PyReadonlyArray1<'_, f64>,
) -> Py<PyArray1<f64>> {
    let obs = py_to_vec(observations);
    let pts = py_to_array1(eval_points);
    let result = mlfinance::features::denoising::rmt::fit_kde(&obs, bandwidth, &pts);
    array1_to_py(py, result)
}

/// Denoise a correlation matrix using Random Matrix Theory (AFML Ch. 2).
///
/// Shrinks eigenvalues below the Marcenko-Pastur bound toward their average,
/// removing noise while preserving the signal.
///
/// Parameters
/// ----------
/// corr : numpy.ndarray
///     Empirical correlation matrix.
/// q : float
///     Ratio T/N.
/// bandwidth : float, optional
///     KDE bandwidth for eigenvalue fitting. Auto-selected if None.
/// shrinkage : bool, default False
///     Use shrinkage-based denoising instead of constant residual eigenvalue.
/// alpha : float, optional
///     Shrinkage intensity (0 to 1). Only used if ``shrinkage=True``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Denoised correlation matrix.
#[pyfunction]
#[pyo3(signature = (corr, q, bandwidth=None, shrinkage=false, alpha=None))]
fn denoise_corr(
    py: Python<'_>,
    corr: PyReadonlyArray2<'_, f64>,
    q: f64,
    bandwidth: Option<f64>,
    shrinkage: bool,
    alpha: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let c = py_to_array2(corr);
    let result = to_pyresult(mlfinance::features::denoising::rmt::denoise_corr(
        &c, q, bandwidth, shrinkage, alpha,
    ))?;
    Ok(array2_to_py(py, result))
}

/// Denoise a covariance matrix using RMT.
///
/// Converts to correlation, denoises, then converts back.
///
/// Parameters
/// ----------
/// cov : numpy.ndarray
///     Empirical covariance matrix.
/// q : float
///     Ratio T/N.
/// bandwidth : float, optional
///     KDE bandwidth.
///
/// Returns
/// -------
/// numpy.ndarray
///     Denoised covariance matrix.
#[pyfunction]
#[pyo3(signature = (cov, q, bandwidth=None))]
fn denoise_cov(
    py: Python<'_>,
    cov: PyReadonlyArray2<'_, f64>,
    q: f64,
    bandwidth: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let m = py_to_array2(cov);
    let result = to_pyresult(mlfinance::features::denoising::rmt::denoise_cov(
        &m, q, bandwidth,
    ))?;
    Ok(array2_to_py(py, result))
}

/// Remove the market component from a correlation matrix (detoning).
///
/// Subtracts the first ``n_components`` principal components to
/// remove common factors (e.g. the market mode).
///
/// Parameters
/// ----------
/// corr : numpy.ndarray
///     Correlation matrix.
/// n_components : int
///     Number of leading eigenvectors to remove (usually 1 for market mode).
///
/// Returns
/// -------
/// numpy.ndarray
///     Detoned correlation matrix.
#[pyfunction]
fn detone_corr(
    py: Python<'_>,
    corr: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let c = py_to_array2(corr);
    let result = to_pyresult(mlfinance::features::denoising::rmt::detone_corr(
        &c,
        n_components,
    ))?;
    Ok(array2_to_py(py, result))
}

/// Optimal portfolio weights from a (denoised) covariance matrix.
///
/// Computes the minimum-variance or max-Sharpe portfolio.
///
/// Parameters
/// ----------
/// cov : numpy.ndarray
///     Covariance matrix.
/// mu : numpy.ndarray, optional
///     Expected returns. If None, computes minimum-variance portfolio.
///
/// Returns
/// -------
/// numpy.ndarray
///     Portfolio weights (sums to 1).
#[pyfunction]
#[pyo3(signature = (cov, mu=None))]
fn optimal_portfolio(
    py: Python<'_>,
    cov: PyReadonlyArray2<'_, f64>,
    mu: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let c = py_to_array2(cov);
    let mu_arr = mu.map(|m| py_to_array1(m));
    let result = to_pyresult(mlfinance::features::denoising::rmt::optimal_portfolio(
        &c,
        mu_arr.as_ref(),
    ))?;
    Ok(array1_to_py(py, result))
}

// ═══════════════════════════════════════════════════════════════════════════
// Allocation
// ═══════════════════════════════════════════════════════════════════════════

/// Hierarchical Risk Parity (HRP) portfolio weights (AFML Ch. 16).
///
/// Uses hierarchical clustering on the correlation matrix to build
/// a diversified portfolio that is more stable than mean-variance.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return matrix (n_periods, n_assets).
///
/// Returns
/// -------
/// numpy.ndarray
///     Portfolio weights (n_assets,), sums to 1.
#[pyfunction]
fn hrp_weights(py: Python<'_>, returns: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let r = py_to_array2(returns);
    let result = to_pyresult(mlfinance::features::allocation::hrp::hrp::hrp_weights(&r))?;
    Ok(array1_to_py(py, result))
}

/// Critical Line Algorithm (CLA) minimum-variance portfolio.
///
/// Parameters
/// ----------
/// cov : numpy.ndarray
///     Covariance matrix (n x n).
///
/// Returns
/// -------
/// numpy.ndarray
///     Minimum-variance weights (n,).
#[pyfunction]
fn cla_min_variance(py: Python<'_>, cov: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let c = py_to_array2(cov);
    let result = to_pyresult(mlfinance::features::allocation::cla::cla_min_variance(&c))?;
    Ok(array1_to_py(py, result))
}

/// CLA maximum Sharpe ratio portfolio.
///
/// Parameters
/// ----------
/// expected_returns : numpy.ndarray
///     Expected return per asset (n,).
/// cov : numpy.ndarray
///     Covariance matrix (n x n).
///
/// Returns
/// -------
/// numpy.ndarray
///     Max-Sharpe weights (n,).
#[pyfunction]
fn cla_max_sharpe(
    py: Python<'_>,
    expected_returns: PyReadonlyArray1<'_, f64>,
    cov: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let mu = py_to_array1(expected_returns);
    let c = py_to_array2(cov);
    let result = to_pyresult(mlfinance::features::allocation::cla::cla_max_sharpe(
        &mu, &c,
    ))?;
    Ok(array1_to_py(py, result))
}

/// Inverse Variance Portfolio (IVP) weights.
///
/// Weights each asset inversely proportional to its variance (diagonal of cov).
///
/// Parameters
/// ----------
/// cov : numpy.ndarray
///     Covariance matrix (n x n).
///
/// Returns
/// -------
/// numpy.ndarray
///     IVP weights (n,), sums to 1.
#[pyfunction]
fn inverse_variance_weights(py: Python<'_>, cov: PyReadonlyArray2<'_, f64>) -> Py<PyArray1<f64>> {
    let c = py_to_array2(cov);
    let result = mlfinance::features::allocation::ivp::inverse_variance_weights(&c);
    array1_to_py(py, result)
}

/// Monte Carlo comparison of HRP, CLA, and IVP allocation methods.
///
/// Simulates random correlation matrices and compares out-of-sample
/// Sharpe ratios and variances.
///
/// Parameters
/// ----------
/// returns : numpy.ndarray
///     Return matrix (n_periods, n_assets).
/// n_simulations : int
///     Number of Monte Carlo trials.
/// seed : int
///     Random seed.
///
/// Returns
/// -------
/// AllocationComparison
///     Sharpe ratios and variances for each method.
#[pyfunction]
fn compare_allocations(
    returns: PyReadonlyArray2<'_, f64>,
    n_simulations: usize,
    seed: u64,
) -> PyResult<PyAllocationComparison> {
    let r = py_to_array2(returns);
    let result = to_pyresult(
        mlfinance::features::allocation::monte_carlo::compare_allocations(&r, n_simulations, seed),
    )?;
    Ok(PyAllocationComparison {
        hrp_sharpe: result.hrp_sharpe,
        cla_sharpe: result.cla_sharpe,
        ivp_sharpe: result.ivp_sharpe,
        hrp_variance: result.hrp_variance,
        cla_variance: result.cla_variance,
        ivp_variance: result.ivp_variance,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Clustering
// ═══════════════════════════════════════════════════════════════════════════

/// K-means clustering with multiple initializations.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Data matrix (n_samples, n_features).
/// k : int
///     Number of clusters.
/// max_iter : int, default 300
///     Maximum iterations per run.
/// n_init : int, default 10
///     Number of random initializations (best is kept).
/// seed : int, default 42
///     Random seed.
///
/// Returns
/// -------
/// KMeansResult
///     Cluster labels, centroids, and iteration count.
#[pyfunction]
#[pyo3(signature = (data, k, max_iter=300, n_init=10, seed=42))]
fn kmeans(
    data: PyReadonlyArray2<'_, f64>,
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> PyResult<PyKMeansResult> {
    let d = py_to_array2(data);
    let result = to_pyresult(mlfinance::features::clustering::onc::kmeans(
        &d, k, max_iter, n_init, seed,
    ))?;
    let centroids_data: Vec<Vec<f64>> = result
        .centroids
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    Ok(PyKMeansResult {
        labels: result.labels,
        n_iterations: result.n_iterations,
        centroids_data,
    })
}

/// Silhouette score measuring clustering quality.
///
/// Ranges from -1 (poor) to +1 (excellent).
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Data matrix (n_samples, n_features).
/// labels : list[int]
///     Cluster labels for each sample.
///
/// Returns
/// -------
/// float
///     Mean silhouette score.
#[pyfunction]
fn silhouette_score(data: PyReadonlyArray2<'_, f64>, labels: Vec<usize>) -> f64 {
    let d = py_to_array2(data);
    mlfinance::features::clustering::onc::silhouette_score(&d, &labels)
}

/// Base-level K-means clustering over a range of k values.
///
/// Tries multiple cluster counts and selects the one with the best
/// silhouette score.
///
/// Parameters
/// ----------
/// corr : numpy.ndarray
///     Correlation matrix.
/// max_clusters : int, optional
///     Maximum k to try.
/// min_clusters : int, optional
///     Minimum k to try.
/// n_init : int, optional
///     Initializations per k.
/// seed : int, optional
///     Random seed.
///
/// Returns
/// -------
/// OncResult
///     Labels, silhouette score, and optimal cluster count.
#[pyfunction]
#[pyo3(signature = (corr, max_clusters=None, min_clusters=None, n_init=None, seed=None))]
fn cluster_kmeans_base(
    corr: PyReadonlyArray2<'_, f64>,
    max_clusters: Option<usize>,
    min_clusters: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> PyResult<PyOncResult> {
    let c = py_to_array2(corr);
    let result = to_pyresult(mlfinance::features::clustering::onc::cluster_kmeans_base(
        &c,
        max_clusters,
        min_clusters,
        n_init,
        seed,
    ))?;
    Ok(PyOncResult {
        labels: result.labels,
        silhouette: result.silhouette,
        n_clusters: result.n_clusters,
    })
}

/// Top-level ONC (Optimal Number of Clusters) algorithm (AFML Ch. 16).
///
/// Two-step approach: first clusters, then re-clusters to find the
/// optimal grouping.
///
/// Parameters
/// ----------
/// corr : numpy.ndarray
///     Correlation matrix.
/// max_clusters : int, optional
///     Maximum k.
/// min_clusters : int, optional
///     Minimum k.
/// n_init : int, optional
///     Initializations per k.
/// seed : int, optional
///     Random seed.
///
/// Returns
/// -------
/// OncResult
///     Labels, silhouette score, and optimal cluster count.
#[pyfunction]
#[pyo3(signature = (corr, max_clusters=None, min_clusters=None, n_init=None, seed=None))]
fn cluster_kmeans_top(
    corr: PyReadonlyArray2<'_, f64>,
    max_clusters: Option<usize>,
    min_clusters: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> PyResult<PyOncResult> {
    let c = py_to_array2(corr);
    let result = to_pyresult(mlfinance::features::clustering::onc::cluster_kmeans_top(
        &c,
        max_clusters,
        min_clusters,
        n_init,
        seed,
    ))?;
    Ok(PyOncResult {
        labels: result.labels,
        silhouette: result.silhouette,
        n_clusters: result.n_clusters,
    })
}

/// Cluster features using ONC on their correlation structure.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Data matrix (n_samples, n_features).
/// max_clusters : int, optional
///     Maximum number of feature clusters.
/// seed : int, optional
///     Random seed.
///
/// Returns
/// -------
/// OncResult
///     Feature cluster labels and quality metrics.
#[pyfunction]
#[pyo3(signature = (data, max_clusters=None, seed=None))]
fn get_feature_clusters(
    data: PyReadonlyArray2<'_, f64>,
    max_clusters: Option<usize>,
    seed: Option<u64>,
) -> PyResult<PyOncResult> {
    let d = py_to_array2(data);
    let result = to_pyresult(mlfinance::features::clustering::onc::get_feature_clusters(
        &d,
        max_clusters,
        seed,
    ))?;
    Ok(PyOncResult {
        labels: result.labels,
        silhouette: result.silhouette,
        n_clusters: result.n_clusters,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Codependence
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a pairwise dependence matrix using the specified method.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Data matrix (n_observations, n_variables).
/// method : str
///     One of: ``"pearson"``, ``"spearman"``, ``"distance_correlation"``,
///     ``"mutual_information"``, ``"variation_of_information"``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Symmetric dependence matrix (n_variables x n_variables).
#[pyfunction]
fn dependence_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    method: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    use mlfinance::features::codependence::codependence_matrix::DependenceMethod;
    let m = py_to_array2(data);
    let method = match method {
        "pearson" => DependenceMethod::Pearson,
        "spearman" => DependenceMethod::Spearman,
        "distance_correlation" => DependenceMethod::DistanceCorrelation,
        "mutual_information" => DependenceMethod::MutualInformation,
        "variation_of_information" => DependenceMethod::VariationOfInformation,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown method: {other}. Use one of: pearson, spearman, distance_correlation, mutual_information, variation_of_information"
            )))
        }
    };
    let result = to_pyresult(
        mlfinance::features::codependence::codependence_matrix::dependence_matrix(&m, method),
    )?;
    Ok(array2_to_py(py, result))
}

/// Convert a correlation matrix to a distance matrix.
///
/// Parameters
/// ----------
/// corr : numpy.ndarray
///     Correlation matrix.
/// metric : str
///     One of: ``"angular"``, ``"absolute_angular"``, ``"squared_angular"``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Distance matrix (n x n).
#[pyfunction]
fn distance_matrix(
    py: Python<'_>,
    corr: PyReadonlyArray2<'_, f64>,
    metric: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    use mlfinance::features::codependence::codependence_matrix::DistanceMetric;
    let c = py_to_array2(corr);
    let metric = match metric {
        "angular" => DistanceMetric::Angular,
        "absolute_angular" => DistanceMetric::AbsoluteAngular,
        "squared_angular" => DistanceMetric::SquaredAngular,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown metric: {other}. Use one of: angular, absolute_angular, squared_angular"
            )))
        }
    };
    let result = to_pyresult(
        mlfinance::features::codependence::codependence_matrix::distance_matrix(&c, metric),
    )?;
    Ok(array2_to_py(py, result))
}

/// Spearman's rank correlation coefficient.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
///
/// Returns
/// -------
/// float
///     Spearman's rho in [-1, 1].
#[pyfunction]
fn spearmans_rho(x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::gnpr_distance::spearmans_rho(&xv, &yv))
}

/// Distance correlation — a measure of dependence for non-linear relationships.
///
/// Unlike Pearson correlation, distance correlation is zero if and only if
/// the variables are independent.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
///
/// Returns
/// -------
/// float
///     Distance correlation in [0, 1].
#[pyfunction]
fn distance_correlation(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::correlation::distance_correlation(&xv, &yv))
}

/// Mutual information between two continuous variables.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
/// n_bins : int, optional
///     Number of histogram bins. Auto-selected if None.
/// normalize : bool, default False
///     If True, normalize to [0, 1] range.
///
/// Returns
/// -------
/// float
///     Mutual information (non-negative).
#[pyfunction]
#[pyo3(signature = (x, y, n_bins=None, normalize=false))]
fn mutual_information(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    n_bins: Option<usize>,
    normalize: bool,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(
        mlfinance::features::codependence::information::mutual_information(
            &xv, &yv, n_bins, normalize,
        ),
    )
}

/// Variation of information — a metric-space distance based on entropy.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
/// n_bins : int, optional
///     Number of histogram bins.
/// normalize : bool, default False
///     If True, normalize to [0, 1] range.
///
/// Returns
/// -------
/// float
///     Variation of information (non-negative).
#[pyfunction]
#[pyo3(signature = (x, y, n_bins=None, normalize=false))]
fn variation_of_information(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    n_bins: Option<usize>,
    normalize: bool,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(
        mlfinance::features::codependence::information::variation_of_information(
            &xv, &yv, n_bins, normalize,
        ),
    )
}

/// Optimal transport dependence measure.
///
/// Based on the Wasserstein distance between joint and product marginal
/// distributions.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
///
/// Returns
/// -------
/// float
///     Optimal transport dependence (non-negative).
#[pyfunction]
fn optimal_transport_dependence(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(
        mlfinance::features::codependence::optimal_transport::optimal_transport_dependence(
            &xv, &yv,
        ),
    )
}

/// Angular distance derived from Pearson correlation.
///
/// ``d = sqrt(0.5 * (1 - rho))``
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
///
/// Returns
/// -------
/// float
///     Angular distance in [0, 1].
#[pyfunction]
fn angular_distance(x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::correlation::angular_distance(&xv, &yv))
}

/// GPR (Gerber-Podolskij-Reisenhofer) distance with threshold.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
/// theta : float
///     Co-movement threshold.
///
/// Returns
/// -------
/// float
///     GPR distance.
#[pyfunction]
fn gpr_distance(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    theta: f64,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::gnpr_distance::gpr_distance(&xv, &yv, theta))
}

/// GNPR (Generalized Non-Parametric Rank) distance.
///
/// Combines rank correlation with an information-theoretic component.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
/// theta : float
///     Co-movement threshold.
/// n_bins : int, optional
///     Number of bins for the information component.
///
/// Returns
/// -------
/// float
///     GNPR distance.
#[pyfunction]
#[pyo3(signature = (x, y, theta, n_bins=None))]
fn gnpr_distance(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    theta: f64,
    n_bins: Option<usize>,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(
        mlfinance::features::codependence::gnpr_distance::gnpr_distance(&xv, &yv, theta, n_bins),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Register all
// ═══════════════════════════════════════════════════════════════════════════

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Structural breaks
    register_functions!(
        m,
        adf_test,
        sadf,
        sadf_stat,
        gsadf,
        gsadf_stat,
        brown_durbin_evans,
        chu_stinchcombe_white,
        sm_poly,
        sm_exp,
        sm_power,
    );
    // Entropy
    register_functions!(
        m,
        binary_encode,
        quantile_encode,
        sigma_encode,
        shannon_entropy,
        plugin_entropy,
        lempel_ziv_complexity,
        kontoyiannis_entropy,
        gaussian_entropy,
        entropy_implied_vol,
    );
    // Microstructure
    register_functions!(
        m,
        amihud_lambda,
        amihud_lambda_rolling,
        kyle_lambda,
        hasbrouck_lambda,
        roll_spread,
        roll_spread_rolling,
        corwin_schultz_spread,
        vpin,
        tick_rule_classify,
    );
    // Denoising
    register_functions!(
        m,
        cov_to_corr,
        corr_to_cov,
        marcenko_pastur_pdf,
        fit_kde,
        denoise_corr,
        denoise_cov,
        detone_corr,
        optimal_portfolio,
    );
    // Allocation
    register_functions!(
        m,
        hrp_weights,
        cla_min_variance,
        cla_max_sharpe,
        inverse_variance_weights,
        compare_allocations,
    );
    // Clustering
    register_functions!(
        m,
        kmeans,
        silhouette_score,
        cluster_kmeans_base,
        cluster_kmeans_top,
        get_feature_clusters,
    );
    // Codependence
    register_functions!(
        m,
        dependence_matrix,
        distance_matrix,
        spearmans_rho,
        distance_correlation,
        mutual_information,
        variation_of_information,
        optimal_transport_dependence,
        angular_distance,
        gpr_distance,
        gnpr_distance,
    );
    Ok(())
}
