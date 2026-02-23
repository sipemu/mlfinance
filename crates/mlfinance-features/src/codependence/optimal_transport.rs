use mlfinance_core::error::{MlFinanceError, Result};

/// Optimal Transport dependence measure between two variables.
///
/// Approximation using the L2 distance between the empirical joint CDF
/// and the product of marginal CDFs (independence copula).
///
/// Steps:
/// 1. Convert to ranks (uniform marginals on \[0,1\])
/// 2. The independence copula is the product of uniform marginals
/// 3. Compute the L2 distance between the empirical joint CDF and the product of marginals
///
/// # Arguments
///
/// * `x` - First variable samples.
/// * `y` - Second variable samples (must have same length as `x`).
///
/// # Returns
///
/// A value in \[0, 1\] where 0 indicates independence.
///
/// # Errors
///
/// Returns an error if lengths differ or fewer than 2 samples.
pub fn optimal_transport_dependence(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!("x length {} != y length {}", n, y.len()),
        });
    }
    if n < 2 {
        return Err(MlFinanceError::InsufficientData {
            expected: 2,
            actual: n,
        });
    }

    let nf = n as f64;

    // Step 1: Convert to uniform ranks in (0, 1]
    let rank_x = compute_uniform_ranks(x);
    let rank_y = compute_uniform_ranks(y);

    // Step 2 & 3: Compute L2 distance between empirical copula and independence copula
    // Evaluate at each observation point (rank_x[i], rank_y[i])
    // The empirical copula at point (u, v) is the fraction of observations
    // where rank_x <= u AND rank_y <= v
    // The independence copula at (u, v) is u * v

    // We evaluate the squared difference at a grid of points defined by the ranks
    let mut l2_sq = 0.0;

    for i in 0..n {
        let u = rank_x[i];
        let v = rank_y[i];

        // Empirical copula: count of points where rank_x <= u AND rank_y <= v
        let mut count = 0;
        for j in 0..n {
            if rank_x[j] <= u + 1e-15 && rank_y[j] <= v + 1e-15 {
                count += 1;
            }
        }
        let c_empirical = count as f64 / nf;

        // Independence copula
        let c_independent = u * v;

        let diff = c_empirical - c_independent;
        l2_sq += diff * diff;
    }

    l2_sq /= nf;
    let l2 = l2_sq.sqrt();

    // Normalize to [0, 1] range
    // The maximum possible L2 distance for the copula is bounded.
    // We use a simple normalization: multiply by sqrt(12) to approximate [0,1] range
    // for the Cramer-von Mises style statistic, then clamp.
    let normalized = (l2 * 12.0_f64.sqrt()).clamp(0.0, 1.0);

    Ok(normalized)
}

/// Convert values to uniform ranks in (0, 1].
///
/// Each value gets its rank divided by n, producing values in (0, 1].
/// Ties receive the average rank.
fn compute_uniform_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let nf = n as f64;

    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank / nf;
        }
        i = j;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- compute_uniform_ranks tests ----

    #[test]
    fn test_uniform_ranks_simple() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = compute_uniform_ranks(&values);
        // Sorted: 1.0(idx 1)->rank 1, 2.0(idx 2)->rank 2, 3.0(idx 0)->rank 3
        // Uniform: rank / 3
        assert_abs_diff_eq!(ranks[0], 1.0, epsilon = 1e-10); // 3/3
        assert_abs_diff_eq!(ranks[1], 1.0 / 3.0, epsilon = 1e-10); // 1/3
        assert_abs_diff_eq!(ranks[2], 2.0 / 3.0, epsilon = 1e-10); // 2/3
    }

    #[test]
    fn test_uniform_ranks_ties() {
        let values = vec![1.0, 2.0, 2.0, 4.0];
        let ranks = compute_uniform_ranks(&values);
        // Rank of 1.0 = 1, rank of 2.0 = avg(2,3) = 2.5, rank of 4.0 = 4
        // Uniform: divide by 4
        assert_abs_diff_eq!(ranks[0], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[1], 0.625, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[2], 0.625, epsilon = 1e-10);
        assert_abs_diff_eq!(ranks[3], 1.0, epsilon = 1e-10);
    }

    // ---- optimal_transport_dependence tests ----

    #[test]
    fn test_ot_dependence_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let d = optimal_transport_dependence(&x, &x).unwrap();
        // Identical variables should show high dependence
        assert!(d > 0.0);
    }

    #[test]
    fn test_ot_dependence_perfect_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&v| v * 3.0 + 7.0).collect();
        let d = optimal_transport_dependence(&x, &y).unwrap();
        // Strong dependence
        assert!(d > 0.0);
    }

    #[test]
    fn test_ot_dependence_nonnegative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let d = optimal_transport_dependence(&x, &y).unwrap();
        assert!(d >= 0.0);
    }

    #[test]
    fn test_ot_dependence_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 6.0, 4.0];
        let d = optimal_transport_dependence(&x, &y).unwrap();
        assert!((0.0..=1.0).contains(&d));
    }

    #[test]
    fn test_ot_dependence_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(optimal_transport_dependence(&x, &y).is_err());
    }

    #[test]
    fn test_ot_dependence_too_short() {
        let x = vec![1.0];
        let y = vec![2.0];
        assert!(optimal_transport_dependence(&x, &y).is_err());
    }

    #[test]
    fn test_ot_dependence_negative_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&v| -v * 2.0 + 100.0).collect();
        let d = optimal_transport_dependence(&x, &y).unwrap();
        // Negative correlation should still show dependence
        assert!(d > 0.0);
    }
}
