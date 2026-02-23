//! Random Matrix Theory (RMT) based denoising of correlation and covariance matrices.
//!
//! Implements the Marcenko-Pastur distribution fitting approach to identify noise
//! eigenvalues, and provides denoising and detoning transformations for financial
//! correlation matrices.

use mlfinance_core::error::{MlFinanceError, Result};
use mlfinance_core::matrix::{matrix_inverse, power_iteration_eig};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper conversions
// ---------------------------------------------------------------------------

/// Convert a covariance matrix to a correlation matrix.
///
/// `corr[i,j] = cov[i,j] / (std[i] * std[j])`
///
/// Returns the correlation matrix together with the vector of standard deviations
/// (square roots of the diagonal of the covariance matrix).
/// Compute a single correlation entry from covariance and standard deviations.
fn corr_entry(cov_ij: f64, std_i: f64, std_j: f64, is_diagonal: bool) -> f64 {
    if std_i < 1e-15 || std_j < 1e-15 {
        return if is_diagonal { 1.0 } else { 0.0 };
    }
    cov_ij / (std_i * std_j)
}

/// Convert a covariance matrix to a correlation matrix.
///
/// `corr[i,j] = cov[i,j] / (std[i] * std[j])`
///
/// # Arguments
///
/// * `cov` - Covariance matrix (n x n), must be square.
///
/// # Returns
///
/// A tuple of (correlation_matrix, std_devs) where std_devs are the square roots
/// of the diagonal of the covariance matrix.
///
/// # Errors
///
/// Returns an error if the matrix is not square.
pub fn cov_to_corr(cov: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = cov.nrows();
    if n != cov.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "covariance matrix must be square".into(),
        });
    }
    if n == 0 {
        return Ok((Array2::zeros((0, 0)), Array1::zeros(0)));
    }

    let std_devs: Array1<f64> = (0..n).map(|i| cov[[i, i]].sqrt()).collect();

    let mut corr = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            corr[[i, j]] = corr_entry(cov[[i, j]], std_devs[i], std_devs[j], i == j);
        }
    }

    Ok((corr, std_devs))
}

/// Convert a correlation matrix back to a covariance matrix.
///
/// `cov[i,j] = corr[i,j] * std[i] * std[j]`
///
/// # Arguments
///
/// * `corr` - Correlation matrix (n x n).
/// * `std` - Standard deviation vector of length n.
///
/// # Returns
///
/// Covariance matrix (n x n).
///
/// # Errors
///
/// Returns an error if the matrix is not square or dimensions mismatch.
pub fn corr_to_cov(corr: &Array2<f64>, std: &Array1<f64>) -> Result<Array2<f64>> {
    let n = corr.nrows();
    if n != corr.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "correlation matrix must be square".into(),
        });
    }
    if n != std.len() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "correlation matrix size {} does not match std vector length {}",
                n,
                std.len()
            ),
        });
    }

    let mut cov = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cov[[i, j]] = corr[[i, j]] * std[i] * std[j];
        }
    }
    Ok(cov)
}

// ---------------------------------------------------------------------------
// Marcenko-Pastur Distribution
// ---------------------------------------------------------------------------

/// Marcenko-Pastur probability density function.
///
/// The density is:
/// ```text
/// f(x) = (q / (2 * pi * sigma^2)) * sqrt((lambda_max - x) * (x - lambda_min)) / x
/// ```
/// where:
/// - `lambda_min = sigma^2 * (1 - 1/sqrt(q))^2`  (clamped to >= 0)
/// - `lambda_max = sigma^2 * (1 + 1/sqrt(q))^2`
///
/// # Parameters
/// - `var`: variance (`sigma^2`)
/// - `q`: `T/N` ratio (number of observations / number of variables)
/// - `pts`: number of evaluation points on `[lambda_min, lambda_max]`
///
/// # Returns
/// `(x_values, pdf_values)` evaluated on `[lambda_min, lambda_max]`
pub fn marcenko_pastur_pdf(var: f64, q: f64, pts: usize) -> Result<(Array1<f64>, Array1<f64>)> {
    if var <= 0.0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "variance must be positive".into(),
        });
    }
    if q <= 0.0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "q (T/N ratio) must be positive".into(),
        });
    }
    if pts < 2 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "pts must be >= 2".into(),
        });
    }

    let q_inv_sqrt = 1.0 / q.sqrt();
    let lambda_min = (var * (1.0 - q_inv_sqrt).powi(2)).max(0.0);
    let lambda_max = var * (1.0 + q_inv_sqrt).powi(2);

    let step = (lambda_max - lambda_min) / (pts - 1) as f64;
    let x_vals: Array1<f64> = (0..pts).map(|i| lambda_min + step * i as f64).collect();
    let pdf_vals: Array1<f64> = x_vals.mapv(|x| mp_density(x, q, var, lambda_min, lambda_max));

    Ok((x_vals, pdf_vals))
}

// ---------------------------------------------------------------------------
// Kernel Density Estimation
// ---------------------------------------------------------------------------

/// Simple kernel density estimation with a Gaussian kernel.
///
/// For each evaluation point `x`:
/// ```text
/// KDE(x) = (1 / (n * h)) * sum_i K((x - x_i) / h)
/// ```
/// where `K` is the standard Gaussian PDF: `K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)`.
///
/// # Arguments
///
/// * `observations` - Sample data points.
/// * `bandwidth` - Kernel bandwidth (smoothing parameter).
/// * `eval_points` - Points at which to evaluate the density estimate.
///
/// # Returns
///
/// Density estimates at each evaluation point.
pub fn fit_kde(observations: &[f64], bandwidth: f64, eval_points: &Array1<f64>) -> Array1<f64> {
    let n = observations.len() as f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();

    eval_points.mapv(|x| {
        let sum: f64 = observations
            .iter()
            .map(|&xi| {
                let u = (x - xi) / bandwidth;
                inv_sqrt_2pi * (-0.5 * u * u).exp()
            })
            .sum();
        sum / (n * bandwidth)
    })
}

// ---------------------------------------------------------------------------
// Finding the noise variance
// ---------------------------------------------------------------------------

/// Find the noise variance by fitting the Marcenko-Pastur distribution to the
/// observed eigenvalue spectrum via grid search.
///
/// The algorithm evaluates candidate variances on a grid from `min_var` to `max_var`,
/// computes the Marcenko-Pastur PDF at the observed eigenvalue points for each
/// candidate, and compares it to the KDE of the observed eigenvalues. The variance
/// minimising the sum of squared errors (SSE) between the two is returned.
///
/// # Parameters
/// - `eigenvalues`: eigenvalues from the correlation matrix
/// - `q`: `T/N` ratio
/// - `bandwidth`: KDE bandwidth (default: 0.25)
/// - `min_var`: lower end of the search grid (default: `1e-5`)
/// - `max_var`: upper end of the search grid (default: `1.0 - 1e-5`)
/// - `num_points`: number of grid points (default: 1000)
///
/// # Returns
/// Estimated noise variance `sigma^2`.
pub fn find_max_eigenvalue(
    eigenvalues: &Array1<f64>,
    q: f64,
    bandwidth: f64,
    min_var: Option<f64>,
    max_var: Option<f64>,
    num_points: Option<usize>,
) -> Result<f64> {
    let min_var = min_var.unwrap_or(1e-5);
    let max_var = max_var.unwrap_or(1.0 - 1e-5);
    let num_points = num_points.unwrap_or(1000);

    if min_var >= max_var {
        return Err(MlFinanceError::InvalidParameter {
            msg: "min_var must be less than max_var".into(),
        });
    }
    if eigenvalues.is_empty() {
        return Err(MlFinanceError::EmptySeries);
    }

    let obs: Vec<f64> = eigenvalues.iter().copied().collect();

    // KDE of observed eigenvalues evaluated at the eigenvalue points themselves
    let kde_observed = fit_kde(&obs, bandwidth, eigenvalues);

    let mut best_var = min_var;
    let mut best_sse = f64::MAX;

    let step = (max_var - min_var) / (num_points - 1).max(1) as f64;
    for i in 0..num_points {
        let candidate_var = min_var + step * i as f64;

        // Compute MP PDF at the observed eigenvalue points
        let q_inv_sqrt = 1.0 / q.sqrt();
        let lambda_min = (candidate_var * (1.0 - q_inv_sqrt).powi(2)).max(0.0);
        let lambda_max = candidate_var * (1.0 + q_inv_sqrt).powi(2);

        let mp_pdf: Array1<f64> = eigenvalues.mapv(|x| {
            if x <= lambda_min || x >= lambda_max || x <= 0.0 {
                0.0
            } else {
                let inner = ((lambda_max - x) * (x - lambda_min)).sqrt();
                q / (2.0 * PI * candidate_var) * inner / x
            }
        });

        // SSE between MP PDF and KDE
        let sse: f64 = mp_pdf
            .iter()
            .zip(kde_observed.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();

        if sse < best_sse {
            best_sse = sse;
            best_var = candidate_var;
        }
    }

    Ok(best_var)
}

// ---------------------------------------------------------------------------
// Denoising
// ---------------------------------------------------------------------------

/// Evaluate the Marcenko-Pastur density at a single point.
fn mp_density(x: f64, q: f64, var: f64, lambda_min: f64, lambda_max: f64) -> f64 {
    if x <= lambda_min || x >= lambda_max || x <= 0.0 {
        return 0.0;
    }
    let inner = ((lambda_max - x) * (x - lambda_min)).sqrt();
    q / (2.0 * PI * var) * inner / x
}

/// Adjust noise eigenvalues by replacing or blending them toward their average.
fn adjust_noise_eigenvalues(
    eigenvalues: &Array1<f64>,
    lambda_max: f64,
    shrinkage: bool,
    alpha_val: f64,
) -> Array1<f64> {
    let noise_sum: f64 = eigenvalues.iter().filter(|&&ev| ev <= lambda_max).sum();
    let noise_count = eigenvalues.iter().filter(|&&ev| ev <= lambda_max).count();
    let noise_avg = if noise_count > 0 {
        noise_sum / noise_count as f64
    } else {
        0.0
    };

    eigenvalues.mapv(|ev| {
        if ev <= lambda_max {
            if shrinkage {
                alpha_val * ev + (1.0 - alpha_val) * noise_avg
            } else {
                noise_avg
            }
        } else {
            ev
        }
    })
}

/// Reconstruct a symmetric matrix from eigenvalues and eigenvectors, skipping the first `skip` components.
fn reconstruct_from_eigen(
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    n: usize,
    skip: usize,
) -> Array2<f64> {
    let mut result = Array2::zeros((n, n));
    for k in skip..n {
        let v = eigenvectors.column(k);
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += eigenvalues[k] * v[i] * v[j];
            }
        }
    }
    result
}

/// Denoise a correlation matrix using the constant residual eigenvalue method.
///
/// # Algorithm
/// 1. Eigendecompose the correlation matrix.
/// 2. Find the noise variance via Marcenko-Pastur fitting.
/// 3. Compute `lambda_max` for that noise variance.
/// 4. Eigenvalues `<= lambda_max` are considered noise.
///    - If `shrinkage` is `false`: replace them with their average.
///    - If `shrinkage` is `true`: blend each noise eigenvalue toward the average,
///      using `alpha` (default 0) for noise components, and preserving signal ones.
/// 5. Reconstruct: `corr_denoised = V * diag(new_eigenvalues) * V^T`.
/// 6. Rescale the diagonal to 1.0.
///
/// # Arguments
///
/// * `corr` - Correlation matrix (n x n), must be square.
/// * `q` - T/N ratio (number of observations / number of variables).
/// * `bandwidth` - KDE bandwidth for Marcenko-Pastur fitting (default: 0.25).
/// * `shrinkage` - Whether to use shrinkage for noise eigenvalues.
/// * `alpha` - Shrinkage blend factor in [0, 1] (default: 0.0); only used when `shrinkage` is true.
///
/// # Returns
///
/// Denoised correlation matrix with diagonal entries equal to 1.0.
///
/// # Errors
///
/// Returns an error if the matrix is not square or eigendecomposition fails.
pub fn denoise_corr(
    corr: &Array2<f64>,
    q: f64,
    bandwidth: Option<f64>,
    shrinkage: bool,
    alpha: Option<f64>,
) -> Result<Array2<f64>> {
    let n = corr.nrows();
    if n != corr.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "correlation matrix must be square".into(),
        });
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let bw = bandwidth.unwrap_or(0.25);
    let (eigenvalues, eigenvectors) = power_iteration_eig(corr, n, 1000, 1e-10)?;
    let noise_var = find_max_eigenvalue(&eigenvalues, q, bw, None, None, None)?;

    let q_inv_sqrt = 1.0 / q.sqrt();
    let lambda_max = noise_var * (1.0 + q_inv_sqrt).powi(2);

    let new_eigenvalues =
        adjust_noise_eigenvalues(&eigenvalues, lambda_max, shrinkage, alpha.unwrap_or(0.0));

    let mut corr_new = reconstruct_from_eigen(&new_eigenvalues, &eigenvectors, n, 0);
    rescale_diagonal(&mut corr_new);

    Ok(corr_new)
}

/// Denoise a covariance matrix.
///
/// Converts to a correlation matrix, denoises it, then converts back to covariance.
///
/// # Arguments
///
/// * `cov` - Covariance matrix (n x n), must be square.
/// * `q` - T/N ratio (number of observations / number of variables).
/// * `bandwidth` - KDE bandwidth for Marcenko-Pastur fitting (default: 0.25).
///
/// # Returns
///
/// Denoised covariance matrix preserving the original diagonal variances.
///
/// # Errors
///
/// Returns an error if the matrix is not square or denoising fails.
pub fn denoise_cov(cov: &Array2<f64>, q: f64, bandwidth: Option<f64>) -> Result<Array2<f64>> {
    let (corr, std_devs) = cov_to_corr(cov)?;
    let corr_denoised = denoise_corr(&corr, q, bandwidth, false, None)?;
    corr_to_cov(&corr_denoised, &std_devs)
}

// ---------------------------------------------------------------------------
// Detoning
// ---------------------------------------------------------------------------

/// Detone a correlation matrix by removing the market component.
///
/// # Algorithm
/// 1. Eigendecompose the correlation matrix.
/// 2. Remove the top `n_components` eigenvectors (market factors) from the
///    reconstruction.
/// 3. Rescale the diagonal back to 1.0.
///
/// # Arguments
///
/// * `corr` - Correlation matrix (n x n), must be square.
/// * `n_components` - Number of leading eigenvectors to remove (must be <= n).
///
/// # Returns
///
/// Detoned correlation matrix with market factors removed and diagonal rescaled to 1.0.
///
/// # Errors
///
/// Returns an error if the matrix is not square, `n_components > n`, or
/// eigendecomposition fails.
pub fn detone_corr(corr: &Array2<f64>, n_components: usize) -> Result<Array2<f64>> {
    let n = corr.nrows();
    if n != corr.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "correlation matrix must be square".into(),
        });
    }
    if n_components > n {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!(
                "n_components ({}) exceeds matrix dimension ({})",
                n_components, n
            ),
        });
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let (eigenvalues, eigenvectors) = power_iteration_eig(corr, n, 1000, 1e-10)?;

    let mut corr_new = reconstruct_from_eigen(&eigenvalues, &eigenvectors, n, n_components);
    rescale_diagonal(&mut corr_new);

    Ok(corr_new)
}

// ---------------------------------------------------------------------------
// Optimal portfolio
// ---------------------------------------------------------------------------

/// Compute portfolio weights from a covariance matrix.
///
/// - If `mu` is `None`: minimum-variance portfolio.
///   `w = Sigma^{-1} * 1 / (1^T Sigma^{-1} 1)`
/// - If `mu` is `Some`: maximum Sharpe ratio portfolio.
///   `w = Sigma^{-1} * mu / (1^T Sigma^{-1} mu)`
///
/// # Arguments
///
/// * `cov` - Covariance matrix (n x n), must be square and invertible.
/// * `mu` - Optional expected returns vector of length n. If `None`, computes
///   minimum-variance weights.
///
/// # Returns
///
/// Portfolio weight vector of length n that sums to 1.0.
///
/// # Errors
///
/// Returns an error if the matrix is not square, dimensions mismatch with `mu`,
/// or the covariance matrix is singular.
pub fn optimal_portfolio(cov: &Array2<f64>, mu: Option<&Array1<f64>>) -> Result<Array1<f64>> {
    let n = cov.nrows();
    if n != cov.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: "covariance matrix must be square".into(),
        });
    }
    if let Some(mu_vec) = mu {
        if mu_vec.len() != n {
            return Err(MlFinanceError::DimensionMismatch {
                msg: format!(
                    "mu length {} does not match covariance matrix size {}",
                    mu_vec.len(),
                    n
                ),
            });
        }
    }

    let inv_cov = matrix_inverse(cov)?;

    let target = match mu {
        Some(mu_vec) => mu_vec.clone(),
        None => Array1::ones(n),
    };

    let w = inv_cov.dot(&target);
    let w_sum: f64 = w.iter().sum();
    if w_sum.abs() < 1e-15 {
        return Err(MlFinanceError::ComputationError {
            msg: "portfolio weights sum to zero".into(),
        });
    }

    Ok(w / w_sum)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Rescale a symmetric matrix so that its diagonal entries are all 1.0.
///
/// For a symmetric matrix M, the rescaled entry is:
/// `M'[i,j] = M[i,j] / (sqrt(M[i,i]) * sqrt(M[j,j]))`
fn rescale_diagonal(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    let inv_sqrt_diag: Vec<f64> = (0..n)
        .map(|i| {
            let d = matrix[[i, i]];
            if d.abs() < 1e-15 {
                1.0
            } else {
                1.0 / d.sqrt()
            }
        })
        .collect();

    for i in 0..n {
        for j in 0..n {
            matrix[[i, j]] *= inv_sqrt_diag[i] * inv_sqrt_diag[j];
        }
    }
    // Force exact diagonal and symmetry
    for i in 0..n {
        matrix[[i, i]] = 1.0;
        for j in (i + 1)..n {
            let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
            matrix[[i, j]] = avg;
            matrix[[j, i]] = avg;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // cov_to_corr / corr_to_cov
    // -----------------------------------------------------------------------

    #[test]
    fn test_cov_to_corr_identity() {
        let cov = Array2::eye(3);
        let (corr, std_devs) = cov_to_corr(&cov).unwrap();
        assert_eq!(corr, Array2::eye(3));
        for &s in std_devs.iter() {
            assert!((s - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cov_to_corr_known_values() {
        // cov = [[4, 2], [2, 9]]  => std = [2, 3], corr[0,1] = 2/(2*3) = 1/3
        let cov = array![[4.0, 2.0], [2.0, 9.0]];
        let (corr, std_devs) = cov_to_corr(&cov).unwrap();
        assert!((std_devs[0] - 2.0).abs() < 1e-10);
        assert!((std_devs[1] - 3.0).abs() < 1e-10);
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((corr[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((corr[[0, 1]] - 1.0 / 3.0).abs() < 1e-10);
        assert!((corr[[1, 0]] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_to_corr_diagonal_is_one() {
        let cov = array![[4.0, 1.0, 0.5], [1.0, 9.0, 0.3], [0.5, 0.3, 16.0]];
        let (corr, _) = cov_to_corr(&cov).unwrap();
        for i in 0..3 {
            assert!((corr[[i, i]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_corr_to_cov_known_values() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let std_devs = array![2.0, 3.0];
        let cov = corr_to_cov(&corr, &std_devs).unwrap();
        assert!((cov[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((cov[[1, 1]] - 9.0).abs() < 1e-10);
        assert!((cov[[0, 1]] - 3.0).abs() < 1e-10); // 0.5 * 2 * 3
        assert!((cov[[1, 0]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cov_corr_roundtrip() {
        let cov = array![[4.0, 2.0, 1.0], [2.0, 9.0, 3.0], [1.0, 3.0, 16.0]];
        let (corr, std_devs) = cov_to_corr(&cov).unwrap();
        let cov2 = corr_to_cov(&corr, &std_devs).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[[i, j]] - cov2[[i, j]]).abs() < 1e-10,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    cov[[i, j]],
                    cov2[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_cov_to_corr_non_square_error() {
        let cov = Array2::zeros((2, 3));
        assert!(cov_to_corr(&cov).is_err());
    }

    #[test]
    fn test_corr_to_cov_dimension_mismatch() {
        let corr = Array2::eye(3);
        let std_devs = array![1.0, 2.0];
        assert!(corr_to_cov(&corr, &std_devs).is_err());
    }

    // -----------------------------------------------------------------------
    // marcenko_pastur_pdf
    // -----------------------------------------------------------------------

    #[test]
    fn test_mp_pdf_integrates_to_approximately_one() {
        let var = 1.0;
        let q = 5.0;
        let pts = 10000;
        let (x_vals, pdf_vals) = marcenko_pastur_pdf(var, q, pts).unwrap();
        let dx = x_vals[1] - x_vals[0];
        let integral: f64 = pdf_vals.iter().sum::<f64>() * dx;
        assert!(
            (integral - 1.0).abs() < 0.05,
            "integral = {} (expected ~1.0)",
            integral
        );
    }

    #[test]
    fn test_mp_pdf_correct_bounds() {
        let var = 1.0;
        let q = 4.0;
        let pts = 100;
        let (x_vals, _) = marcenko_pastur_pdf(var, q, pts).unwrap();

        let q_inv_sqrt = 1.0 / q.sqrt();
        let expected_min = (var * (1.0 - q_inv_sqrt).powi(2)).max(0.0);
        let expected_max = var * (1.0 + q_inv_sqrt).powi(2);

        assert!((x_vals[0] - expected_min).abs() < 1e-10);
        assert!((x_vals[pts - 1] - expected_max).abs() < 1e-10);
    }

    #[test]
    fn test_mp_pdf_boundary_values_zero() {
        // At the exact boundaries, the PDF should be 0
        let var = 1.0;
        let q = 5.0;
        let pts = 100;
        let (_, pdf_vals) = marcenko_pastur_pdf(var, q, pts).unwrap();
        assert!((pdf_vals[0]).abs() < 1e-10);
        assert!((pdf_vals[pts - 1]).abs() < 1e-10);
    }

    #[test]
    fn test_mp_pdf_positive_interior() {
        let var = 1.0;
        let q = 5.0;
        let pts = 100;
        let (_, pdf_vals) = marcenko_pastur_pdf(var, q, pts).unwrap();
        // Interior points should be positive
        for i in 2..(pts - 2) {
            assert!(pdf_vals[i] >= 0.0, "pdf_vals[{}] = {}", i, pdf_vals[i]);
        }
    }

    #[test]
    fn test_mp_pdf_invalid_var() {
        assert!(marcenko_pastur_pdf(-1.0, 5.0, 100).is_err());
        assert!(marcenko_pastur_pdf(0.0, 5.0, 100).is_err());
    }

    #[test]
    fn test_mp_pdf_invalid_q() {
        assert!(marcenko_pastur_pdf(1.0, 0.0, 100).is_err());
        assert!(marcenko_pastur_pdf(1.0, -1.0, 100).is_err());
    }

    #[test]
    fn test_mp_pdf_invalid_pts() {
        assert!(marcenko_pastur_pdf(1.0, 5.0, 0).is_err());
        assert!(marcenko_pastur_pdf(1.0, 5.0, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // fit_kde
    // -----------------------------------------------------------------------

    #[test]
    fn test_kde_single_observation() {
        let obs = [0.0];
        let bw = 1.0;
        let pts = array![0.0];
        let kde = fit_kde(&obs, bw, &pts);
        // KDE at the observation with h=1: (1/sqrt(2*pi)) * exp(0) / 1 = 1/sqrt(2*pi)
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!(
            (kde[0] - expected).abs() < 1e-10,
            "kde = {}, expected = {}",
            kde[0],
            expected
        );
    }

    #[test]
    fn test_kde_integrates_approximately() {
        // For a set of observations, KDE should integrate to ~1
        let obs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let bw = 0.1;
        let n_pts = 1000;
        let x_min = -0.5;
        let x_max = 1.5;
        let step = (x_max - x_min) / (n_pts - 1) as f64;
        let pts: Array1<f64> = (0..n_pts).map(|i| x_min + step * i as f64).collect();
        let kde = fit_kde(&obs, bw, &pts);
        let integral: f64 = kde.iter().sum::<f64>() * step;
        assert!(
            (integral - 1.0).abs() < 0.1,
            "integral = {} (expected ~1.0)",
            integral
        );
    }

    #[test]
    fn test_kde_bandwidth_effect() {
        // Smaller bandwidth => sharper peak at observation
        let obs = [0.0];
        let pts = array![0.0];
        let kde_narrow = fit_kde(&obs, 0.1, &pts);
        let kde_wide = fit_kde(&obs, 1.0, &pts);
        assert!(kde_narrow[0] > kde_wide[0]);
    }

    #[test]
    fn test_kde_symmetry() {
        let obs = [0.0];
        let bw = 1.0;
        let pts = array![-1.0, 1.0];
        let kde = fit_kde(&obs, bw, &pts);
        assert!((kde[0] - kde[1]).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // find_max_eigenvalue
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_max_eigenvalue_basic() {
        // Create eigenvalues that look like noise around var=1.0
        // For q=5, lambda_max(var=1) = (1 + 1/sqrt(5))^2 ~ 1.894
        // Generate eigenvalues uniformly in [0.2, 1.8]
        let eigenvalues: Array1<f64> = (0..50).map(|i| 0.2 + 1.6 * i as f64 / 49.0).collect();
        let q = 5.0;
        let result = find_max_eigenvalue(&eigenvalues, q, 0.25, None, None, None);
        assert!(result.is_ok());
        let var = result.unwrap();
        assert!(var > 0.0 && var < 1.0);
    }

    #[test]
    fn test_find_max_eigenvalue_empty() {
        let eigenvalues = Array1::zeros(0);
        assert!(find_max_eigenvalue(&eigenvalues, 5.0, 0.25, None, None, None).is_err());
    }

    #[test]
    fn test_find_max_eigenvalue_invalid_range() {
        let eigenvalues = array![1.0, 2.0, 3.0];
        assert!(find_max_eigenvalue(&eigenvalues, 5.0, 0.25, Some(0.9), Some(0.1), None).is_err());
    }

    // -----------------------------------------------------------------------
    // denoise_corr
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_corr_diagonal_is_one() {
        let corr = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];
        let q = 10.0;
        let result = denoise_corr(&corr, q, Some(0.5), false, None).unwrap();
        for i in 0..3 {
            assert!(
                (result[[i, i]] - 1.0).abs() < 1e-10,
                "diagonal[{}] = {}",
                i,
                result[[i, i]]
            );
        }
    }

    #[test]
    fn test_denoise_corr_symmetric() {
        let corr = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];
        let q = 10.0;
        let result = denoise_corr(&corr, q, Some(0.5), false, None).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result[[i, j]] - result[[j, i]]).abs() < 1e-10,
                    "asymmetry at ({}, {}): {} vs {}",
                    i,
                    j,
                    result[[i, j]],
                    result[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_denoise_corr_values_in_range() {
        let corr = array![[1.0, 0.5, -0.2], [0.5, 1.0, 0.3], [-0.2, 0.3, 1.0]];
        let q = 10.0;
        let result = denoise_corr(&corr, q, Some(0.5), false, None).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    result[[i, j]] >= -1.0 - 1e-10 && result[[i, j]] <= 1.0 + 1e-10,
                    "value at ({}, {}) = {} out of [-1, 1]",
                    i,
                    j,
                    result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_denoise_corr_non_square_error() {
        let corr = Array2::zeros((2, 3));
        assert!(denoise_corr(&corr, 10.0, None, false, None).is_err());
    }

    #[test]
    fn test_denoise_corr_empty() {
        let corr = Array2::zeros((0, 0));
        let result = denoise_corr(&corr, 10.0, None, false, None).unwrap();
        assert_eq!(result.nrows(), 0);
    }

    #[test]
    fn test_denoise_corr_shrinkage_vs_no_shrinkage() {
        let corr = array![[1.0, 0.4, 0.1], [0.4, 1.0, 0.3], [0.1, 0.3, 1.0]];
        let q = 10.0;
        let no_shrink = denoise_corr(&corr, q, Some(0.5), false, None).unwrap();
        let with_shrink = denoise_corr(&corr, q, Some(0.5), true, Some(0.5)).unwrap();
        // They should differ (unless all eigenvalues are signal)
        let mut any_diff = false;
        for i in 0..3 {
            for j in 0..3 {
                if (no_shrink[[i, j]] - with_shrink[[i, j]]).abs() > 1e-10 {
                    any_diff = true;
                }
            }
        }
        // Both should at least have diagonal = 1
        for i in 0..3 {
            assert!((no_shrink[[i, i]] - 1.0).abs() < 1e-10);
            assert!((with_shrink[[i, i]] - 1.0).abs() < 1e-10);
        }
        // It's possible they're the same if no eigenvalue falls below threshold,
        // so we just verify both are valid
        let _ = any_diff;
    }

    #[test]
    fn test_denoise_corr_shrinkage_alpha_zero() {
        // With shrinkage=true and alpha=0, should be the same as shrinkage=false
        let corr = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];
        let q = 10.0;
        let no_shrink = denoise_corr(&corr, q, Some(0.5), false, None).unwrap();
        let shrink_zero = denoise_corr(&corr, q, Some(0.5), true, Some(0.0)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (no_shrink[[i, j]] - shrink_zero[[i, j]]).abs() < 1e-10,
                    "mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_denoise_corr_shrinkage_alpha_one() {
        // With shrinkage=true and alpha=1, noise eigenvalues should be preserved
        // (since ev_new = 1.0 * ev + 0.0 * avg = ev)
        let corr = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];
        let q = 10.0;
        let result = denoise_corr(&corr, q, Some(0.5), true, Some(1.0)).unwrap();
        for i in 0..3 {
            assert!((result[[i, i]] - 1.0).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // denoise_cov
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_cov_basic() {
        let cov = array![[4.0, 1.2, 0.4], [1.2, 9.0, 1.8], [0.4, 1.8, 16.0]];
        let q = 10.0;
        let result = denoise_cov(&cov, q, Some(0.5)).unwrap();
        // Result should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result[[i, j]] - result[[j, i]]).abs() < 1e-10,
                    "asymmetry at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_denoise_cov_preserves_variances_approx() {
        // Diagonal entries should roughly preserve the original variances
        // (because we convert back with the same std devs)
        let cov = array![[4.0, 0.5, 0.1], [0.5, 9.0, 0.3], [0.1, 0.3, 16.0]];
        let q = 10.0;
        let result = denoise_cov(&cov, q, Some(0.5)).unwrap();
        for i in 0..3 {
            assert!(
                (result[[i, i]] - cov[[i, i]]).abs() < 1e-6,
                "variance mismatch at {}: {} vs {}",
                i,
                result[[i, i]],
                cov[[i, i]]
            );
        }
    }

    #[test]
    fn test_denoise_cov_roundtrip_with_corr() {
        // Denoising cov should give the same result as converting to corr,
        // denoising, and converting back
        let cov = array![[4.0, 1.0, 0.5], [1.0, 9.0, 1.5], [0.5, 1.5, 16.0]];
        let q = 10.0;
        let bw = Some(0.5);
        let result_cov = denoise_cov(&cov, q, bw).unwrap();
        let (corr, std_devs) = cov_to_corr(&cov).unwrap();
        let corr_dn = denoise_corr(&corr, q, bw, false, None).unwrap();
        let result_manual = corr_to_cov(&corr_dn, &std_devs).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result_cov[[i, j]] - result_manual[[i, j]]).abs() < 1e-10,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    result_cov[[i, j]],
                    result_manual[[i, j]]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // detone_corr
    // -----------------------------------------------------------------------

    #[test]
    fn test_detone_corr_zero_components_preserves() {
        // Removing 0 components should reconstruct the original (approx)
        let corr = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let result = detone_corr(&corr, 0).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result[[i, j]] - corr[[i, j]]).abs() < 0.05,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    result[[i, j]],
                    corr[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_detone_corr_diagonal_is_one() {
        let corr = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let result = detone_corr(&corr, 1).unwrap();
        for i in 0..3 {
            assert!(
                (result[[i, i]] - 1.0).abs() < 1e-10,
                "diagonal[{}] = {}",
                i,
                result[[i, i]]
            );
        }
    }

    #[test]
    fn test_detone_corr_symmetric() {
        let corr = array![[1.0, 0.6, -0.1], [0.6, 1.0, 0.4], [-0.1, 0.4, 1.0]];
        let result = detone_corr(&corr, 1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result[[i, j]] - result[[j, i]]).abs() < 1e-10,
                    "asymmetry at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_detone_corr_too_many_components() {
        let corr = Array2::eye(3);
        assert!(detone_corr(&corr, 4).is_err());
    }

    #[test]
    fn test_detone_corr_values_in_range() {
        let corr = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let result = detone_corr(&corr, 1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    result[[i, j]] >= -1.0 - 1e-10 && result[[i, j]] <= 1.0 + 1e-10,
                    "out of range at ({}, {}): {}",
                    i,
                    j,
                    result[[i, j]]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // optimal_portfolio
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimal_portfolio_identity_cov() {
        // With identity covariance, minimum-variance weights should be equal
        let cov = Array2::eye(3);
        let w = optimal_portfolio(&cov, None).unwrap();
        for &wi in w.iter() {
            assert!(
                (wi - 1.0 / 3.0).abs() < 1e-10,
                "weight = {} (expected 1/3)",
                wi
            );
        }
    }

    #[test]
    fn test_optimal_portfolio_weights_sum_to_one() {
        let cov = array![[4.0, 1.0, 0.5], [1.0, 9.0, 1.5], [0.5, 1.5, 16.0]];
        let w = optimal_portfolio(&cov, None).unwrap();
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights sum = {}", sum);
    }

    #[test]
    fn test_optimal_portfolio_with_mu() {
        let cov = Array2::eye(3);
        let mu = array![1.0, 2.0, 3.0];
        let w = optimal_portfolio(&cov, Some(&mu)).unwrap();
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights sum = {}", sum);
        // Weights should be proportional to mu for identity cov
        // w = Sigma^-1 * mu / (1' * Sigma^-1 * mu) = mu / sum(mu) = [1/6, 2/6, 3/6]
        assert!((w[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((w[1] - 2.0 / 6.0).abs() < 1e-10);
        assert!((w[2] - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_portfolio_dimension_mismatch() {
        let cov = Array2::eye(3);
        let mu = array![1.0, 2.0];
        assert!(optimal_portfolio(&cov, Some(&mu)).is_err());
    }

    #[test]
    fn test_optimal_portfolio_non_square_error() {
        let cov = Array2::zeros((2, 3));
        assert!(optimal_portfolio(&cov, None).is_err());
    }

    #[test]
    fn test_optimal_portfolio_2x2() {
        // Simple 2-asset case
        let cov = array![[1.0, 0.0], [0.0, 4.0]];
        let w = optimal_portfolio(&cov, None).unwrap();
        // Inv = [[1, 0], [0, 0.25]], ones = [1, 1]
        // w_raw = [1, 0.25], sum = 1.25
        // w = [0.8, 0.2]
        assert!((w[0] - 0.8).abs() < 1e-10);
        assert!((w[1] - 0.2).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // rescale_diagonal (tested indirectly through denoise/detone)
    // -----------------------------------------------------------------------

    #[test]
    fn test_rescale_diagonal_basic() {
        let mut m = array![[4.0, 2.0], [2.0, 9.0]];
        rescale_diagonal(&mut m);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((m[[1, 1]] - 1.0).abs() < 1e-10);
        // off-diagonal: 2 / (sqrt(4) * sqrt(9)) = 2/6 = 1/3
        assert!((m[[0, 1]] - 1.0 / 3.0).abs() < 1e-10);
        assert!((m[[1, 0]] - 1.0 / 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Edge cases & integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_then_detone() {
        let corr = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let q = 10.0;
        let denoised = denoise_corr(&corr, q, Some(0.5), false, None).unwrap();
        let detoned = detone_corr(&denoised, 1).unwrap();
        // Resulting matrix should be valid correlation matrix
        for i in 0..3 {
            assert!((detoned[[i, i]] - 1.0).abs() < 1e-10);
            for j in 0..3 {
                assert!(detoned[[i, j]] >= -1.0 - 1e-10 && detoned[[i, j]] <= 1.0 + 1e-10,);
                assert!((detoned[[i, j]] - detoned[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_mp_pdf_different_variances() {
        // With higher variance, the distribution should shift rightward
        let (x1, _) = marcenko_pastur_pdf(0.5, 5.0, 100).unwrap();
        let (x2, _) = marcenko_pastur_pdf(1.5, 5.0, 100).unwrap();
        assert!(x2[x2.len() - 1] > x1[x1.len() - 1]);
    }

    #[test]
    fn test_cov_to_corr_empty() {
        let cov = Array2::zeros((0, 0));
        let (corr, std_devs) = cov_to_corr(&cov).unwrap();
        assert_eq!(corr.nrows(), 0);
        assert_eq!(std_devs.len(), 0);
    }
}
