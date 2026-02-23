/// Statistical features of the order size distribution.
pub struct OrderSizeFeatures {
    /// Mean order size.
    pub mean_size: f64,
    /// Standard deviation of order sizes.
    pub std_size: f64,
    /// Skewness of order sizes (adjusted for sample bias).
    pub skew_size: f64,
    /// Fraction of orders that are round lot multiples.
    pub round_lot_fraction: f64,
}

/// Compute order size distribution features.
///
/// # Arguments
///
/// * `volumes` - Trade volumes (order sizes).
///
/// # Returns
///
/// An [`OrderSizeFeatures`] struct with mean, standard deviation, skewness,
/// and round lot fraction. Returns all zeros for empty input.
pub fn order_size_distribution(volumes: &[f64]) -> OrderSizeFeatures {
    if volumes.is_empty() {
        return OrderSizeFeatures {
            mean_size: 0.0,
            std_size: 0.0,
            skew_size: 0.0,
            round_lot_fraction: 0.0,
        };
    }

    let n = volumes.len() as f64;
    let mean = volumes.iter().sum::<f64>() / n;

    let var = if volumes.len() > 1 {
        volumes.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let std = var.sqrt();

    let skew = if volumes.len() > 2 && std > 1e-15 {
        let nf = n;
        let sum: f64 = volumes.iter().map(|&v| ((v - mean) / std).powi(3)).sum();
        sum * nf / ((nf - 1.0) * (nf - 2.0))
    } else {
        0.0
    };

    let rlf = round_lot_fraction(volumes, 100.0);

    OrderSizeFeatures {
        mean_size: mean,
        std_size: std,
        skew_size: skew,
        round_lot_fraction: rlf,
    }
}

/// Detect round lot orders (multiples of `lot_size`).
///
/// Returns the fraction of volumes that are multiples of `lot_size`.
///
/// # Arguments
///
/// * `volumes` - Trade volumes (order sizes).
/// * `lot_size` - Round lot size (e.g. 100 for US equities).
///
/// # Returns
///
/// Fraction of round lot orders in [0, 1]. Returns 0.0 if volumes is empty
/// or `lot_size <= 0`.
pub fn round_lot_fraction(volumes: &[f64], lot_size: f64) -> f64 {
    if volumes.is_empty() || lot_size <= 0.0 {
        return 0.0;
    }

    let count = volumes
        .iter()
        .filter(|&&v| {
            if v <= 0.0 {
                return false;
            }
            let remainder = v % lot_size;
            // Allow small floating-point tolerance
            remainder < 1e-6 || (lot_size - remainder) < 1e-6
        })
        .count();

    count as f64 / volumes.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_size_distribution() {
        let volumes = vec![100.0, 200.0, 150.0, 300.0, 100.0];
        let features = order_size_distribution(&volumes);
        assert!((features.mean_size - 170.0).abs() < 1e-10);
        assert!(features.std_size > 0.0);
        assert!(features.skew_size.is_finite());
    }

    #[test]
    fn test_order_size_empty() {
        let features = order_size_distribution(&[]);
        assert_eq!(features.mean_size, 0.0);
        assert_eq!(features.std_size, 0.0);
        assert_eq!(features.skew_size, 0.0);
    }

    #[test]
    fn test_round_lot_fraction_all_round() {
        let volumes = vec![100.0, 200.0, 300.0, 400.0];
        let frac = round_lot_fraction(&volumes, 100.0);
        assert!((frac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_round_lot_fraction_none_round() {
        let volumes = vec![101.0, 203.0, 357.0];
        let frac = round_lot_fraction(&volumes, 100.0);
        assert!((frac).abs() < 1e-10);
    }

    #[test]
    fn test_round_lot_fraction_mixed() {
        let volumes = vec![100.0, 150.0, 200.0, 250.0];
        let frac = round_lot_fraction(&volumes, 100.0);
        assert!((frac - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_round_lot_fraction_empty() {
        let frac = round_lot_fraction(&[], 100.0);
        assert_eq!(frac, 0.0);
    }

    #[test]
    fn test_round_lot_fraction_zero_lot() {
        let frac = round_lot_fraction(&[100.0], 0.0);
        assert_eq!(frac, 0.0);
    }
}
