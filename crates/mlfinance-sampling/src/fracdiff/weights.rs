/// Compute weights for fractional differentiation (Snippet 5.1).
///
/// Uses the recurrence relation:
///
/// ```text
/// w[0] = 1
/// w[k] = -w[k-1] * (d - k + 1) / k
/// ```
///
/// # Arguments
/// * `d` - fractional differentiation order (typically `0 < d < 1`)
/// * `size` - number of weights to compute
///
/// # Returns
/// A vector of `size` weights.
pub fn get_weights(d: f64, size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }

    let mut w = Vec::with_capacity(size);
    w.push(1.0);

    for k in 1..size {
        let prev = w[k - 1];
        let next = -prev * (d - k as f64 + 1.0) / k as f64;
        w.push(next);
    }

    w
}

/// Fixed-width window weights for FFD (Snippet 5.3).
///
/// Computes weights using the same recurrence as `get_weights`, but stops
/// when the absolute value of the weight drops below `threshold`.
///
/// # Arguments
/// * `d` - fractional differentiation order
/// * `threshold` - minimum absolute weight magnitude to include
///
/// # Returns
/// A vector of weights with `|w[k]| >= threshold` for all `k`.
pub fn get_weights_ffd(d: f64, threshold: f64) -> Vec<f64> {
    let mut w = vec![1.0];
    let mut k = 1usize;

    loop {
        let prev = w[k - 1];
        let next = -prev * (d - k as f64 + 1.0) / k as f64;
        if next.abs() < threshold {
            break;
        }
        w.push(next);
        k += 1;

        // Safety limit to prevent infinite loops
        if k > 1_000_000 {
            break;
        }
    }

    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_weights_empty() {
        let w = get_weights(0.5, 0);
        assert!(w.is_empty());
    }

    #[test]
    fn test_get_weights_single() {
        let w = get_weights(0.5, 1);
        assert_eq!(w.len(), 1);
        assert!((w[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_weights_first_few() {
        let d = 0.5;
        let w = get_weights(d, 4);
        assert_eq!(w.len(), 4);
        // w[0] = 1
        assert!((w[0] - 1.0).abs() < 1e-10);
        // w[1] = -1 * (0.5 - 0) / 1 = -0.5
        assert!((w[1] - (-0.5)).abs() < 1e-10);
        // w[2] = -(-0.5) * (0.5 - 1) / 2 = 0.5 * (-0.5) / 2 = -0.125
        assert!((w[2] - (-0.125)).abs() < 1e-10);
        // w[3] = -(-0.125) * (0.5 - 2) / 3 = 0.125 * (-1.5) / 3 = -0.0625
        assert!((w[3] - (-0.0625)).abs() < 1e-10);
    }

    #[test]
    fn test_get_weights_d_one() {
        // d=1 is ordinary first difference
        let w = get_weights(1.0, 3);
        // w[0] = 1, w[1] = -1*(1-0)/1 = -1, w[2] = -(-1)*(1-1)/2 = 0
        assert!((w[0] - 1.0).abs() < 1e-10);
        assert!((w[1] - (-1.0)).abs() < 1e-10);
        assert!((w[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_weights_d_zero() {
        // d=0 means no differentiation, only w[0]=1, rest are 0
        let w = get_weights(0.0, 5);
        assert!((w[0] - 1.0).abs() < 1e-10);
        for i in 1..5 {
            assert!(
                w[i].abs() < 1e-10,
                "w[{}] = {} should be 0 for d=0",
                i,
                w[i]
            );
        }
    }

    #[test]
    fn test_get_weights_ffd_threshold() {
        let w = get_weights_ffd(0.5, 0.1);
        // All weights should have |w| >= 0.1
        for &wi in &w {
            assert!(
                wi.abs() >= 0.1,
                "weight {} should have magnitude >= 0.1",
                wi
            );
        }
        assert!(!w.is_empty());
        assert!((w[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_weights_ffd_small_threshold() {
        let w_small = get_weights_ffd(0.5, 0.001);
        let w_large = get_weights_ffd(0.5, 0.1);
        // Smaller threshold should produce more weights
        assert!(w_small.len() >= w_large.len());
    }

    #[test]
    fn test_get_weights_ffd_d_one() {
        let w = get_weights_ffd(1.0, 0.01);
        // d=1: w[0]=1, w[1]=-1, w[2]=0 (which is < threshold)
        assert_eq!(w.len(), 2);
        assert!((w[0] - 1.0).abs() < 1e-10);
        assert!((w[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_weights_decrease_in_magnitude() {
        let w = get_weights(0.5, 10);
        for i in 1..w.len() {
            assert!(
                w[i].abs() <= w[i - 1].abs() + 1e-10,
                "|w[{}]|={} > |w[{}]|={}",
                i,
                w[i].abs(),
                i - 1,
                w[i - 1].abs()
            );
        }
    }
}
