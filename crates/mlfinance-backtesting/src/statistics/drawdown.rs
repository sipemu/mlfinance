//! Drawdown statistics (Snippet 14.4).
//!
//! Computes drawdown-related metrics from a return series, including maximum
//! drawdown, drawdown duration, and time under water.

/// Result of drawdown analysis.
#[derive(Debug, Clone)]
pub struct DrawdownResult {
    /// Maximum drawdown (as a positive fraction, e.g., 0.15 = 15%).
    pub max_drawdown: f64,
    /// Duration of the maximum drawdown in number of periods.
    pub max_drawdown_duration: usize,
    /// Series of drawdown values at each time point (non-negative).
    pub drawdown_series: Vec<f64>,
    /// Time under water at each point: number of periods since the last high-water mark.
    pub time_under_water: Vec<usize>,
}

/// Compute drawdown statistics from a return series.
///
/// Constructs a cumulative wealth curve from the returns and computes the
/// drawdown at each point as the decline from the running maximum (high-water
/// mark).
///
/// # Arguments
/// * `returns` - Periodic return series (not log returns).
///
/// # Returns
/// A `DrawdownResult` with comprehensive drawdown metrics. Returns a default
/// result with empty vectors if the input is empty.
pub fn compute_drawdowns(returns: &[f64]) -> DrawdownResult {
    if returns.is_empty() {
        return DrawdownResult {
            max_drawdown: 0.0,
            max_drawdown_duration: 0,
            drawdown_series: Vec::new(),
            time_under_water: Vec::new(),
        };
    }

    let n = returns.len();

    // Build cumulative wealth curve: W_t = prod(1 + r_i)
    let mut wealth = Vec::with_capacity(n);
    let mut w = 1.0;
    for &r in returns {
        w *= 1.0 + r;
        wealth.push(w);
    }

    // Running maximum (high-water mark)
    let mut hwm = Vec::with_capacity(n);
    let mut current_max = f64::NEG_INFINITY;
    for &w in &wealth {
        current_max = current_max.max(w);
        hwm.push(current_max);
    }

    // Drawdown series: dd_t = (hwm_t - wealth_t) / hwm_t
    let drawdown_series: Vec<f64> = wealth
        .iter()
        .zip(hwm.iter())
        .map(
            |(&w, &h)| {
                if h > 0.0 {
                    ((h - w) / h).max(0.0)
                } else {
                    0.0
                }
            },
        )
        .collect();

    // Time under water
    let mut time_under_water = Vec::with_capacity(n);
    let mut underwater_count = 0usize;
    for &dd in &drawdown_series {
        if dd > 1e-15 {
            underwater_count += 1;
        } else {
            underwater_count = 0;
        }
        time_under_water.push(underwater_count);
    }

    // Maximum drawdown
    let max_drawdown = drawdown_series.iter().copied().fold(0.0_f64, f64::max);

    // Maximum drawdown duration
    let max_drawdown_duration = time_under_water.iter().copied().max().unwrap_or(0);

    DrawdownResult {
        max_drawdown,
        max_drawdown_duration,
        drawdown_series,
        time_under_water,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawdown_basic() {
        // Goes up then down
        let returns = vec![0.10, 0.05, -0.10, -0.05, 0.20];
        let dd = compute_drawdowns(&returns);
        assert_eq!(dd.drawdown_series.len(), 5);
        assert!(dd.max_drawdown > 0.0);
    }

    #[test]
    fn test_drawdown_all_positive() {
        let returns = vec![0.01, 0.02, 0.03, 0.01, 0.02];
        let dd = compute_drawdowns(&returns);
        assert!(
            dd.max_drawdown < 1e-10,
            "no drawdown with all positive returns, got {}",
            dd.max_drawdown
        );
        assert_eq!(dd.max_drawdown_duration, 0);
    }

    #[test]
    fn test_drawdown_all_negative() {
        let returns = vec![-0.01, -0.02, -0.03, -0.01, -0.02];
        let dd = compute_drawdowns(&returns);
        assert!(dd.max_drawdown > 0.0);
        // First bar sets the HWM, so no drawdown at index 0. Subsequent bars
        // are all below the HWM, so time_under_water = [0, 1, 2, 3, 4].
        assert_eq!(dd.max_drawdown_duration, 4);
    }

    #[test]
    fn test_drawdown_empty() {
        let dd = compute_drawdowns(&[]);
        assert_eq!(dd.max_drawdown, 0.0);
        assert_eq!(dd.max_drawdown_duration, 0);
        assert!(dd.drawdown_series.is_empty());
    }

    #[test]
    fn test_drawdown_single() {
        let dd = compute_drawdowns(&[0.05]);
        assert_eq!(dd.drawdown_series.len(), 1);
    }

    #[test]
    fn test_drawdown_recovery() {
        // Down then recover
        let returns = vec![-0.10, -0.05, 0.10, 0.10, 0.05];
        let dd = compute_drawdowns(&returns);
        // After recovery, drawdown should return to zero or near zero
        let last_dd = *dd.drawdown_series.last().unwrap();
        let max_dd = dd.max_drawdown;
        assert!(last_dd < max_dd, "should recover partially");
    }

    #[test]
    fn test_time_under_water() {
        let returns = vec![0.10, -0.05, -0.03, 0.10, 0.05];
        let dd = compute_drawdowns(&returns);
        // First return positive: dd=0, tuw=0
        assert_eq!(dd.time_under_water[0], 0);
        // After negative returns, should be underwater
        assert!(dd.time_under_water[1] > 0 || dd.time_under_water[2] > 0);
    }

    #[test]
    fn test_drawdown_values_non_negative() {
        let returns = vec![0.05, -0.10, 0.03, -0.02, 0.08, -0.05];
        let dd = compute_drawdowns(&returns);
        for &d in &dd.drawdown_series {
            assert!(d >= 0.0, "drawdown should be non-negative, got {}", d);
        }
    }
}
