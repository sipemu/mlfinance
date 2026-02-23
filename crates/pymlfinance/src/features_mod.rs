use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::*;
use crate::types::*;

// ═══════════════════════════════════════════════════════════════════════════
// Structural breaks
// ═══════════════════════════════════════════════════════════════════════════

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

#[pyfunction]
fn sadf_stat(series: PyReadonlyArray1<'_, f64>, min_window: usize, max_lags: usize) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sadf::sadf_stat(&s, min_window, max_lags)
}

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

#[pyfunction]
fn gsadf_stat(series: PyReadonlyArray1<'_, f64>, min_window: usize, max_lags: usize) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::gsadf::gsadf_stat(&s, min_window, max_lags)
}

#[pyfunction]
fn brown_durbin_evans(
    py: Python<'_>,
    residuals: PyReadonlyArray1<'_, f64>,
) -> (Py<PyArray1<f64>>, f64) {
    let r = py_to_vec(residuals);
    let (cusum, sig) = mlfinance::features::structural_breaks::cusum_tests::brown_durbin_evans(&r);
    (vec_to_py_array(py, cusum), sig)
}

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

#[pyfunction]
fn sm_poly(series: PyReadonlyArray1<'_, f64>, degree: usize) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sub_super_martingale::sm_poly(&s, degree)
}

#[pyfunction]
fn sm_exp(series: PyReadonlyArray1<'_, f64>) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sub_super_martingale::sm_exp(&s)
}

#[pyfunction]
fn sm_power(series: PyReadonlyArray1<'_, f64>, power: f64) -> f64 {
    let s = py_to_vec(series);
    mlfinance::features::structural_breaks::sub_super_martingale::sm_power(&s, power)
}

// ═══════════════════════════════════════════════════════════════════════════
// Entropy
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
fn binary_encode(py: Python<'_>, values: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyAny>> {
    let v = py_to_vec(values);
    let result = mlfinance::features::entropy::encoding::binary_encode(&v);
    vec_bool_to_list(py, result)
}

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

#[pyfunction]
fn shannon_entropy(probs: PyReadonlyArray1<'_, f64>) -> f64 {
    let p = py_to_vec(probs);
    mlfinance::features::entropy::shannon::shannon_entropy(&p)
}

#[pyfunction]
fn plugin_entropy(sequence: Vec<usize>, num_symbols: usize) -> f64 {
    mlfinance::features::entropy::plugin::plugin_entropy(&sequence, num_symbols)
}

#[pyfunction]
fn lempel_ziv_complexity(binary_string: Vec<bool>) -> usize {
    mlfinance::features::entropy::lempel_ziv::lempel_ziv_complexity(&binary_string)
}

#[pyfunction]
fn kontoyiannis_entropy(sequence: Vec<usize>, window: usize) -> f64 {
    mlfinance::features::entropy::kontoyiannis::kontoyiannis_entropy(&sequence, window)
}

#[pyfunction]
fn gaussian_entropy(variance: f64) -> f64 {
    mlfinance::features::entropy::gaussian_entropy::gaussian_entropy(variance)
}

#[pyfunction]
fn entropy_implied_vol(entropy: f64) -> f64 {
    mlfinance::features::entropy::gaussian_entropy::entropy_implied_vol(entropy)
}

// ═══════════════════════════════════════════════════════════════════════════
// Microstructure
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
fn amihud_lambda(
    returns: PyReadonlyArray1<'_, f64>,
    dollar_volumes: PyReadonlyArray1<'_, f64>,
) -> f64 {
    let r = py_to_vec(returns);
    let dv = py_to_vec(dollar_volumes);
    mlfinance::features::microstructure::amihud_lambda::amihud_lambda(&r, &dv)
}

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

#[pyfunction]
fn kyle_lambda(
    returns: PyReadonlyArray1<'_, f64>,
    signed_volume: PyReadonlyArray1<'_, f64>,
) -> f64 {
    let r = py_to_vec(returns);
    let sv = py_to_vec(signed_volume);
    mlfinance::features::microstructure::kyle_lambda::kyle_lambda(&r, &sv)
}

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

#[pyfunction]
fn roll_spread(prices: PyReadonlyArray1<'_, f64>) -> f64 {
    let p = py_to_vec(prices);
    mlfinance::features::microstructure::roll_model::roll_spread(&p)
}

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

#[pyfunction]
fn tick_rule_classify(py: Python<'_>, prices: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    let p = py_to_vec(prices);
    let result = mlfinance::features::microstructure::tick_rule::tick_rule_classify(&p);
    vec_to_py_array(py, result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Denoising (RMT)
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
fn cov_to_corr(
    py: Python<'_>,
    cov: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let m = py_to_array2(cov);
    let (corr, std) = to_pyresult(mlfinance::features::denoising::rmt::cov_to_corr(&m))?;
    Ok((array2_to_py(py, corr), array1_to_py(py, std)))
}

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

#[pyfunction]
fn hrp_weights(py: Python<'_>, returns: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let r = py_to_array2(returns);
    let result = to_pyresult(mlfinance::features::allocation::hrp::hrp::hrp_weights(&r))?;
    Ok(array1_to_py(py, result))
}

#[pyfunction]
fn cla_min_variance(py: Python<'_>, cov: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let c = py_to_array2(cov);
    let result = to_pyresult(mlfinance::features::allocation::cla::cla_min_variance(&c))?;
    Ok(array1_to_py(py, result))
}

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

#[pyfunction]
fn inverse_variance_weights(py: Python<'_>, cov: PyReadonlyArray2<'_, f64>) -> Py<PyArray1<f64>> {
    let c = py_to_array2(cov);
    let result = mlfinance::features::allocation::ivp::inverse_variance_weights(&c);
    array1_to_py(py, result)
}

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

#[pyfunction]
fn silhouette_score(data: PyReadonlyArray2<'_, f64>, labels: Vec<usize>) -> f64 {
    let d = py_to_array2(data);
    mlfinance::features::clustering::onc::silhouette_score(&d, &labels)
}

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

#[pyfunction]
fn spearmans_rho(x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::gnpr_distance::spearmans_rho(&xv, &yv))
}

#[pyfunction]
fn distance_correlation(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::correlation::distance_correlation(&xv, &yv))
}

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

#[pyfunction]
fn angular_distance(x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    to_pyresult(mlfinance::features::codependence::correlation::angular_distance(&xv, &yv))
}

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
    m.add_function(wrap_pyfunction!(adf_test, m)?)?;
    m.add_function(wrap_pyfunction!(sadf, m)?)?;
    m.add_function(wrap_pyfunction!(sadf_stat, m)?)?;
    m.add_function(wrap_pyfunction!(gsadf, m)?)?;
    m.add_function(wrap_pyfunction!(gsadf_stat, m)?)?;
    m.add_function(wrap_pyfunction!(brown_durbin_evans, m)?)?;
    m.add_function(wrap_pyfunction!(chu_stinchcombe_white, m)?)?;
    m.add_function(wrap_pyfunction!(sm_poly, m)?)?;
    m.add_function(wrap_pyfunction!(sm_exp, m)?)?;
    m.add_function(wrap_pyfunction!(sm_power, m)?)?;
    // Entropy
    m.add_function(wrap_pyfunction!(binary_encode, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_encode, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_encode, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(lempel_ziv_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(kontoyiannis_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_implied_vol, m)?)?;
    // Microstructure
    m.add_function(wrap_pyfunction!(amihud_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(amihud_lambda_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(kyle_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(hasbrouck_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(roll_spread, m)?)?;
    m.add_function(wrap_pyfunction!(roll_spread_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(corwin_schultz_spread, m)?)?;
    m.add_function(wrap_pyfunction!(vpin, m)?)?;
    m.add_function(wrap_pyfunction!(tick_rule_classify, m)?)?;
    // Denoising
    m.add_function(wrap_pyfunction!(cov_to_corr, m)?)?;
    m.add_function(wrap_pyfunction!(corr_to_cov, m)?)?;
    m.add_function(wrap_pyfunction!(marcenko_pastur_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(fit_kde, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_corr, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_cov, m)?)?;
    m.add_function(wrap_pyfunction!(detone_corr, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_portfolio, m)?)?;
    // Allocation
    m.add_function(wrap_pyfunction!(hrp_weights, m)?)?;
    m.add_function(wrap_pyfunction!(cla_min_variance, m)?)?;
    m.add_function(wrap_pyfunction!(cla_max_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_variance_weights, m)?)?;
    m.add_function(wrap_pyfunction!(compare_allocations, m)?)?;
    // Clustering
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_kmeans_base, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_kmeans_top, m)?)?;
    m.add_function(wrap_pyfunction!(get_feature_clusters, m)?)?;
    // Codependence
    m.add_function(wrap_pyfunction!(dependence_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(spearmans_rho, m)?)?;
    m.add_function(wrap_pyfunction!(distance_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(variation_of_information, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_transport_dependence, m)?)?;
    m.add_function(wrap_pyfunction!(angular_distance, m)?)?;
    m.add_function(wrap_pyfunction!(gpr_distance, m)?)?;
    m.add_function(wrap_pyfunction!(gnpr_distance, m)?)?;
    Ok(())
}
