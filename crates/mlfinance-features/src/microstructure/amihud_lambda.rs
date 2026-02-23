/// Amihud's lambda: illiquidity measure.
///
/// Computed as avg(|return| / dollar_volume).
/// Higher values indicate less liquid assets.
///
/// # Arguments
///
/// * `returns` - Asset returns for each period.
/// * `dollar_volumes` - Dollar trading volume for each period.
///
/// # Returns
///
/// Average price impact ratio. Returns 0.0 if both slices are empty or all
/// volumes are zero.
pub fn amihud_lambda(returns: &[f64], dollar_volumes: &[f64]) -> f64 {
    let n = returns.len().min(dollar_volumes.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        if dollar_volumes[i] > 0.0 {
            sum += returns[i].abs() / dollar_volumes[i];
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Rolling Amihud lambda.
///
/// Computes Amihud's lambda over a rolling window.
///
/// # Arguments
///
/// * `returns` - Asset returns for each period.
/// * `dollar_volumes` - Dollar trading volume for each period.
/// * `window` - Rolling window size.
///
/// # Returns
///
/// Vector of Amihud lambda values for each window position.
/// Returns an empty vector if the data is shorter than the window.
pub fn amihud_lambda_rolling(returns: &[f64], dollar_volumes: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len().min(dollar_volumes.len());
    if n < window || window == 0 {
        return vec![];
    }

    (0..=(n - window))
        .map(|i| amihud_lambda(&returns[i..i + window], &dollar_volumes[i..i + window]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amihud_basic() {
        let returns = vec![0.01, -0.02, 0.005];
        let volumes = vec![1_000_000.0, 2_000_000.0, 500_000.0];
        let lambda = amihud_lambda(&returns, &volumes);
        // avg(0.01/1e6, 0.02/2e6, 0.005/5e5)
        let expected = (0.01 / 1e6 + 0.02 / 2e6 + 0.005 / 5e5) / 3.0;
        assert!((lambda - expected).abs() < 1e-15);
    }

    #[test]
    fn test_amihud_zero_volume() {
        let returns = vec![0.01, -0.02, 0.005];
        let volumes = vec![0.0, 0.0, 0.0];
        let lambda = amihud_lambda(&returns, &volumes);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_amihud_empty() {
        let lambda = amihud_lambda(&[], &[]);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_amihud_rolling() {
        let returns = vec![0.01, -0.02, 0.005, 0.015, -0.01];
        let volumes = vec![1e6, 2e6, 1.5e6, 1e6, 2e6];
        let lambdas = amihud_lambda_rolling(&returns, &volumes, 3);
        assert_eq!(lambdas.len(), 3);
        // First window
        let expected_0 = amihud_lambda(&returns[0..3], &volumes[0..3]);
        assert!((lambdas[0] - expected_0).abs() < 1e-15);
    }

    #[test]
    fn test_amihud_rolling_short() {
        let lambdas = amihud_lambda_rolling(&[0.01], &[1e6], 3);
        assert!(lambdas.is_empty());
    }
}
