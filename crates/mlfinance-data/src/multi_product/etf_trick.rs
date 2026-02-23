//! ETF trick: combine multiple product price series into a single continuous series.
//!
//! This is the technique from "Advances in Financial Machine Learning" (Chapter 3)
//! for constructing a single tradeable series from multiple products using
//! time-varying weights.

/// Combine multiple product price series into a single continuous ETF-like series.
///
/// Given `K` products over `T` time steps, this computes a single continuous
/// series by tracking the portfolio value as weights change over time.
///
/// # Arguments
/// - `prices`: slice of K vectors, each of length T (`prices[k][t]` = price of product k at time t)
/// - `weights`: slice of K vectors, each of length T (`weights[k][t]` = allocation to product k at time t)
///
/// # Returns
/// A `Vec<f64>` of length T representing the synthetic ETF price series.
///
/// # Panics
/// Panics if `prices` and `weights` have different lengths or if any inner
/// vectors have inconsistent lengths.
pub fn etf_trick(prices: &[Vec<f64>], weights: &[Vec<f64>]) -> Vec<f64> {
    assert!(!prices.is_empty(), "prices must not be empty");
    assert_eq!(
        prices.len(),
        weights.len(),
        "prices and weights must have the same number of products"
    );

    let n_products = prices.len();
    let n_periods = prices[0].len();
    assert!(n_periods > 0, "price series must have at least one element");

    for k in 0..n_products {
        assert_eq!(
            prices[k].len(),
            n_periods,
            "all price series must have the same length"
        );
        assert_eq!(
            weights[k].len(),
            n_periods,
            "all weight series must have the same length"
        );
    }

    let mut etf = vec![0.0; n_periods];

    // Compute initial portfolio value
    let mut portfolio_value: f64 = (0..n_products).map(|k| weights[k][0] * prices[k][0]).sum();
    etf[0] = portfolio_value;

    // Holdings: number of units of each product
    let mut holdings: Vec<f64> = (0..n_products)
        .map(|k| {
            if prices[k][0] != 0.0 {
                weights[k][0] * portfolio_value / prices[k][0]
            } else {
                0.0
            }
        })
        .collect();

    for t in 1..n_periods {
        // Mark-to-market with new prices
        portfolio_value = (0..n_products).map(|k| holdings[k] * prices[k][t]).sum();
        etf[t] = portfolio_value;

        // Rebalance: compute new holdings based on new weights
        holdings = (0..n_products)
            .map(|k| {
                if prices[k][t] != 0.0 {
                    weights[k][t] * portfolio_value / prices[k][t]
                } else {
                    0.0
                }
            })
            .collect();
    }

    etf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_etf_trick_single_product() {
        let prices = vec![vec![100.0, 110.0, 105.0, 115.0]];
        let weights = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let etf = etf_trick(&prices, &weights);
        assert_eq!(etf.len(), 4);
        assert_eq!(etf[0], 100.0);
        // With weight=1 and holdings = 100/100 = 1 unit, etf[1] = 1*110 = 110
        assert!((etf[1] - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_etf_trick_two_products() {
        let prices = vec![vec![100.0, 110.0, 120.0], vec![50.0, 45.0, 55.0]];
        let weights = vec![vec![0.6, 0.6, 0.6], vec![0.4, 0.4, 0.4]];
        let etf = etf_trick(&prices, &weights);
        assert_eq!(etf.len(), 3);
        // Initial: 0.6*100 + 0.4*50 = 80
        assert!((etf[0] - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_etf_trick_constant_prices() {
        let prices = vec![vec![100.0, 100.0, 100.0]];
        let weights = vec![vec![1.0, 1.0, 1.0]];
        let etf = etf_trick(&prices, &weights);
        for &v in &etf {
            assert!((v - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic]
    fn test_etf_trick_empty() {
        let prices: Vec<Vec<f64>> = vec![];
        let weights: Vec<Vec<f64>> = vec![];
        etf_trick(&prices, &weights);
    }
}
