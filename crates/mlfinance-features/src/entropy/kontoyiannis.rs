/// Kontoyiannis entropy estimator: based on longest match length.
///
/// The estimator uses the average of log2(longest_match_length) / log2(window)
/// over positions in the sequence. As the match lengths grow, the entropy
/// estimate converges to the true entropy rate.
///
/// Reference: Kontoyiannis, I. et al. (1998)
///
/// # Arguments
///
/// * `sequence` - Integer-encoded symbol sequence.
/// * `window` - Size of the lookback window for finding matches.
///
/// # Returns
///
/// Estimated entropy rate in bits. Returns 0.0 if the sequence is too short.
pub fn kontoyiannis_entropy(sequence: &[usize], window: usize) -> f64 {
    let n = sequence.len();
    if n <= window || window == 0 {
        return 0.0;
    }

    let mut sum_inv_match = 0.0;
    let mut count = 0;

    for i in window..n {
        let l = longest_match_length(sequence, i, window);
        // The entropy rate is estimated as the reciprocal of the average match length
        // H ~= 1 / (avg(L_i) / log2(window))
        // Equivalently: sum(log2(window) / L_i) / count
        if l > 0 {
            sum_inv_match += (l as f64).recip();
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    // H ~= (1/count) * sum(1/L_i) * log2(window)
    let log2_window = (window as f64).log2();
    (sum_inv_match / count as f64) * log2_window
}

/// Longest match length at position `i`.
///
/// Finds the length of the longest substring starting at position `i`
/// that also appears in the window `[i-window..i]`.
///
/// # Arguments
///
/// * `sequence` - Integer-encoded symbol sequence.
/// * `i` - Position in the sequence at which to start the match.
/// * `window` - Size of the lookback window for finding matches.
///
/// # Returns
///
/// Length of the longest matching substring plus 1 (Kontoyiannis convention).
pub fn longest_match_length(sequence: &[usize], i: usize, window: usize) -> usize {
    let n = sequence.len();
    if i >= n || i < 1 {
        return 0;
    }

    let search_start = i.saturating_sub(window);

    let mut max_match = 0;

    for start in search_start..i {
        let mut match_len = 0;
        while i + match_len < n
            && start + match_len < i
            && sequence[i + match_len] == sequence[start + match_len]
        {
            match_len += 1;
        }
        if match_len > max_match {
            max_match = match_len;
        }
    }

    // Add 1 for the next character (following Kontoyiannis convention)
    max_match + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest_match_constant() {
        // All same symbol: matches should be long
        let seq = vec![0; 20];
        let l = longest_match_length(&seq, 10, 10);
        // Should match up to the boundary
        assert!(l >= 2);
    }

    #[test]
    fn test_longest_match_no_match() {
        // Alternating, looking at position 1 with window 1
        let seq = vec![0, 1, 0, 1];
        let l = longest_match_length(&seq, 1, 1);
        // seq[1] = 1, search in seq[0..1] = [0]. No match, so length = 0 + 1 = 1
        assert_eq!(l, 1);
    }

    #[test]
    fn test_kontoyiannis_constant() {
        // Constant sequence: very low entropy
        let seq = vec![0; 100];
        let h = kontoyiannis_entropy(&seq, 20);
        assert!(h < 0.5);
    }

    #[test]
    fn test_kontoyiannis_high_entropy() {
        // "Random-like" sequence: higher entropy
        let seq: Vec<usize> = (0..200).map(|i| (i * 7 + 3) % 8).collect();
        let h = kontoyiannis_entropy(&seq, 30);
        assert!(h > 0.0);
    }

    #[test]
    fn test_kontoyiannis_empty() {
        let h = kontoyiannis_entropy(&[], 10);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_kontoyiannis_window_too_large() {
        let seq = vec![0, 1, 2, 3, 4];
        let h = kontoyiannis_entropy(&seq, 10);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_longest_match_boundary() {
        let l = longest_match_length(&[0, 1, 2], 0, 1);
        assert_eq!(l, 0);
    }
}
