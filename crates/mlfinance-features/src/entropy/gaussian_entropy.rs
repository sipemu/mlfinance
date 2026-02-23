use std::f64::consts::{E, PI};

/// Entropy of a Gaussian distribution with given variance.
///
/// H = 0.5 * log2(2 * pi * e * variance)
///
/// # Arguments
///
/// * `variance` - Variance of the Gaussian distribution (must be positive).
///
/// # Returns
///
/// Entropy in bits. Returns negative infinity if variance is zero or negative.
pub fn gaussian_entropy(variance: f64) -> f64 {
    if variance <= 0.0 {
        return f64::NEG_INFINITY;
    }
    0.5 * (2.0 * PI * E * variance).log2()
}

/// Entropy-implied volatility: the standard deviation such that Gaussian entropy matches observed.
///
/// Given entropy H, find sigma such that:
///   H = 0.5 * log2(2 * pi * e * sigma^2)
///   => sigma = sqrt(2^(2H) / (2 * pi * e))
///
/// # Arguments
///
/// * `entropy` - Observed entropy in bits.
///
/// # Returns
///
/// Implied volatility (standard deviation). Returns 0.0 if the implied
/// variance is non-positive.
pub fn entropy_implied_vol(entropy: f64) -> f64 {
    let var = (2.0_f64).powf(2.0 * entropy) / (2.0 * PI * E);
    if var <= 0.0 {
        return 0.0;
    }
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_entropy_unit_variance() {
        let h = gaussian_entropy(1.0);
        // H = 0.5 * log2(2 * pi * e) ~= 2.0471
        let expected = 0.5 * (2.0 * PI * E).log2();
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_entropy_increases_with_variance() {
        let h1 = gaussian_entropy(1.0);
        let h2 = gaussian_entropy(4.0);
        assert!(h2 > h1);
    }

    #[test]
    fn test_gaussian_entropy_zero_variance() {
        let h = gaussian_entropy(0.0);
        assert!(h.is_infinite() && h < 0.0);
    }

    #[test]
    fn test_gaussian_entropy_negative_variance() {
        let h = gaussian_entropy(-1.0);
        assert!(h.is_infinite() && h < 0.0);
    }

    #[test]
    fn test_entropy_implied_vol_roundtrip() {
        // Start with a known variance, compute entropy, then recover vol
        let original_var = 0.04; // sigma = 0.2
        let h = gaussian_entropy(original_var);
        let recovered_vol = entropy_implied_vol(h);
        let original_vol = original_var.sqrt();
        assert!((recovered_vol - original_vol).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_implied_vol_unit() {
        let h = gaussian_entropy(1.0);
        let vol = entropy_implied_vol(h);
        assert!((vol - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_implied_vol_high_entropy() {
        // Higher entropy -> higher vol
        let vol1 = entropy_implied_vol(1.0);
        let vol2 = entropy_implied_vol(2.0);
        assert!(vol2 > vol1);
    }
}
