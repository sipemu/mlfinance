//! Single future roll: compute roll gaps and adjust prices for a rolling
//! futures contract series.
//!
//! When futures contracts are rolled from one contract to the next, there is
//! typically a gap between the closing price of the old contract and the
//! opening price of the new contract. These functions handle the price
//! adjustment to create a continuous series.

/// Compute the cumulative roll gaps at each roll date.
///
/// At each roll date, the gap is `prices[roll_date] - prices[roll_date - 1]`.
/// The returned vector has the same length as `prices`, with the cumulative
/// gap adjustment for each index. Indices before the first roll have
/// the full cumulative gap; after the last roll, the gap is zero.
///
/// # Arguments
/// - `prices`: raw unadjusted price series
/// - `roll_dates`: indices where the contract rolls occur
///
/// # Returns
/// A vector of cumulative gap adjustments (same length as `prices`).
pub fn roll_gaps(prices: &[f64], roll_dates: &[usize]) -> Vec<f64> {
    let n = prices.len();
    if n == 0 {
        return vec![];
    }

    let mut gaps = vec![0.0; n];

    // Compute raw gaps at each roll date
    let mut raw_gaps: Vec<(usize, f64)> = Vec::new();
    for &rd in roll_dates {
        if rd > 0 && rd < n {
            let gap = prices[rd] - prices[rd - 1];
            raw_gaps.push((rd, gap));
        }
    }

    // Build cumulative adjustments backward:
    // Prices before a roll date need to be adjusted by the sum of all future gaps
    // to create a continuous series.
    let total_gap: f64 = raw_gaps.iter().map(|(_, g)| g).sum();
    let mut cumulative = total_gap;

    let mut gap_idx = 0;
    // Sort roll dates for correct processing
    let mut sorted_dates: Vec<(usize, f64)> = raw_gaps;
    sorted_dates.sort_by_key(|(idx, _)| *idx);

    for (i, gap) in gaps.iter_mut().enumerate() {
        if gap_idx < sorted_dates.len() && i >= sorted_dates[gap_idx].0 {
            cumulative -= sorted_dates[gap_idx].1;
            gap_idx += 1;
        }
        *gap = cumulative;
    }

    gaps
}

/// Compute a non-negative rolled price series.
///
/// Adjusts the raw prices by the cumulative roll gaps, then shifts the
/// entire series so that the minimum value is zero (ensuring no negative prices).
///
/// # Arguments
/// - `prices`: raw unadjusted price series
/// - `roll_dates`: indices where the contract rolls occur
///
/// # Returns
/// A vector of adjusted non-negative prices (same length as `prices`).
pub fn non_negative_rolled(prices: &[f64], roll_dates: &[usize]) -> Vec<f64> {
    let n = prices.len();
    if n == 0 {
        return vec![];
    }

    let gaps = roll_gaps(prices, roll_dates);
    let mut adjusted: Vec<f64> = prices.iter().zip(gaps.iter()).map(|(p, g)| p + g).collect();

    // Find minimum and shift so all values are non-negative
    let min_val = adjusted.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_val < 0.0 {
        for v in &mut adjusted {
            *v -= min_val;
        }
    }

    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_gaps_no_rolls() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let gaps = roll_gaps(&prices, &[]);
        assert_eq!(gaps, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_roll_gaps_single_roll() {
        // Contract 1: prices 100, 101
        // Contract 2 starts at index 2: price jumps from 101 to 105
        let prices = vec![100.0, 101.0, 105.0, 106.0];
        let roll_dates = vec![2];
        let gaps = roll_gaps(&prices, &roll_dates);
        // Gap at index 2 = 105 - 101 = 4
        // Indices 0,1 need adjustment of +4, indices 2,3 need 0
        assert_eq!(gaps[0], 4.0);
        assert_eq!(gaps[1], 4.0);
        assert_eq!(gaps[2], 0.0);
        assert_eq!(gaps[3], 0.0);
    }

    #[test]
    fn test_roll_gaps_two_rolls() {
        let prices = vec![100.0, 101.0, 105.0, 106.0, 110.0, 111.0];
        let roll_dates = vec![2, 4];
        let gaps = roll_gaps(&prices, &roll_dates);
        // Gap at 2: 105-101=4, gap at 4: 110-106=4
        // Total gap = 8
        // Indices 0,1: cum=8
        // Indices 2,3: cum=8-4=4
        // Indices 4,5: cum=4-4=0
        assert_eq!(gaps[0], 8.0);
        assert_eq!(gaps[1], 8.0);
        assert_eq!(gaps[2], 4.0);
        assert_eq!(gaps[3], 4.0);
        assert_eq!(gaps[4], 0.0);
        assert_eq!(gaps[5], 0.0);
    }

    #[test]
    fn test_non_negative_rolled_basic() {
        let prices = vec![100.0, 101.0, 105.0, 106.0];
        let roll_dates = vec![2];
        let adjusted = non_negative_rolled(&prices, &roll_dates);
        // Gaps: [4, 4, 0, 0]
        // Adjusted: [104, 105, 105, 106]
        assert_eq!(adjusted, vec![104.0, 105.0, 105.0, 106.0]);
    }

    #[test]
    fn test_non_negative_rolled_negative_shift() {
        // Downward roll gap (new contract cheaper)
        let prices = vec![100.0, 101.0, 95.0, 96.0];
        let roll_dates = vec![2];
        // Gap = 95 - 101 = -6
        // Adjusted: [100-6, 101-6, 95, 96] = [94, 95, 95, 96]
        let adjusted = non_negative_rolled(&prices, &roll_dates);
        assert!(adjusted.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_empty() {
        assert_eq!(roll_gaps(&[], &[]), Vec::<f64>::new());
        assert_eq!(non_negative_rolled(&[], &[]), Vec::<f64>::new());
    }
}
