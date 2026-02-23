/// Aggregate volume into fixed-size buckets based on trade signs.
fn aggregate_into_buckets(
    volumes: &[f64],
    signs: &[f64],
    n: usize,
    bucket_size: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut buy_buckets = Vec::new();
    let mut sell_buckets = Vec::new();
    let mut current_buy = 0.0;
    let mut current_sell = 0.0;
    let mut current_volume = 0.0;

    for i in 0..n {
        if signs[i] > 0.0 {
            current_buy += volumes[i];
        } else {
            current_sell += volumes[i];
        }
        current_volume += volumes[i];
        if current_volume >= bucket_size {
            buy_buckets.push(current_buy);
            sell_buckets.push(current_sell);
            current_buy = 0.0;
            current_sell = 0.0;
            current_volume = 0.0;
        }
    }
    (buy_buckets, sell_buckets)
}

/// Compute VPIN over a rolling window of buckets.
fn compute_rolling_vpin(buy_buckets: &[f64], sell_buckets: &[f64], n_buckets: usize) -> Vec<f64> {
    let num = buy_buckets.len();
    (0..=(num - n_buckets))
        .map(|i| {
            let mut imbalance = 0.0;
            let mut volume = 0.0;
            for j in i..(i + n_buckets) {
                imbalance += (buy_buckets[j] - sell_buckets[j]).abs();
                volume += buy_buckets[j] + sell_buckets[j];
            }
            if volume > 0.0 {
                imbalance / volume
            } else {
                0.0
            }
        })
        .collect()
}

/// Volume-Synchronized Probability of Informed Trading (VPIN).
///
/// Classifies volume into buy/sell using the bulk volume classification
/// approach, then computes VPIN over rolling buckets.
///
/// # Arguments
///
/// * `volumes` - Trade volumes for each period.
/// * `prices` - Trade prices for each period.
/// * `bucket_size` - Total volume per bucket (must be positive).
/// * `n_buckets` - Number of buckets in the VPIN rolling window.
///
/// # Returns
///
/// Vector of VPIN values in [0, 1], where higher values indicate greater
/// probability of informed trading. Returns an empty vector if the input
/// is too short, `bucket_size <= 0`, or `n_buckets == 0`.
pub fn vpin(volumes: &[f64], prices: &[f64], bucket_size: f64, n_buckets: usize) -> Vec<f64> {
    let n = volumes.len().min(prices.len());
    if n < 2 || bucket_size <= 0.0 || n_buckets == 0 {
        return vec![];
    }

    // Step 1: Classify volume using tick rule approximation
    let trade_signs = classify_volume(prices);

    let (buy_buckets, sell_buckets) = aggregate_into_buckets(volumes, &trade_signs, n, bucket_size);

    let num_buckets = buy_buckets.len();
    if num_buckets < n_buckets {
        return vec![];
    }

    compute_rolling_vpin(&buy_buckets, &sell_buckets, n_buckets)
}

/// Simple volume classification using price changes.
fn classify_volume(prices: &[f64]) -> Vec<f64> {
    let mut signs = vec![0.0_f64; prices.len()];
    if prices.is_empty() {
        return signs;
    }
    let mut last_sign = 0.0;
    for i in 1..prices.len() {
        let diff = prices[i] - prices[i - 1];
        if diff > 0.0 {
            last_sign = 1.0;
        } else if diff < 0.0 {
            last_sign = -1.0;
        }
        signs[i] = last_sign;
    }
    signs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_basic() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let volumes: Vec<f64> = vec![100.0; 100];
        let result = vpin(&volumes, &prices, 500.0, 5);
        // Should produce some VPIN values
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_vpin_empty() {
        let result = vpin(&[], &[], 100.0, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_vpin_zero_bucket() {
        let result = vpin(&[100.0], &[100.0], 0.0, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_vpin_range() {
        // VPIN should be between 0 and 1
        let prices: Vec<f64> = (0..200)
            .map(|i| 50.0 + (i as f64 * 0.05).sin() * 5.0)
            .collect();
        let volumes = vec![50.0; 200];
        let result = vpin(&volumes, &prices, 200.0, 10);
        for &v in &result {
            assert!(v >= 0.0);
            assert!(v <= 1.0);
        }
    }

    #[test]
    fn test_classify_volume() {
        let prices = vec![100.0, 101.0, 100.5, 100.5, 101.5];
        let signs = classify_volume(&prices);
        assert_eq!(signs[0], 0.0);
        assert_eq!(signs[1], 1.0);
        assert_eq!(signs[2], -1.0);
        assert_eq!(signs[3], -1.0); // Carry forward
        assert_eq!(signs[4], 1.0);
    }
}
