/// Check if `target` appears as a substring within `source[..search_end]`.
fn substring_exists(source: &[bool], target: &[bool], search_end: usize) -> bool {
    let target_len = target.len();
    if target_len > search_end {
        return false;
    }
    (0..=(search_end - target_len)).any(|start| &source[start..start + target_len] == target)
}

/// Find the longest match of `binary_string[pos..]` that exists in `binary_string[..pos+match_len]`.
/// Returns `(match_length, reached_end)`.
fn find_longest_match(binary_string: &[bool], pos: usize) -> (usize, bool) {
    let n = binary_string.len();
    let mut match_len = 0;

    loop {
        if pos + match_len >= n {
            return (match_len, true);
        }

        let target = &binary_string[pos..=pos + match_len];
        let search_end = pos + match_len;

        if substring_exists(binary_string, target, search_end) {
            match_len += 1;
        } else {
            return (match_len, false);
        }
    }
}

/// Lempel-Ziv complexity of a binary string.
///
/// Counts the number of distinct substrings encountered when parsing
/// the string from left to right, extending the current match until
/// a new substring is found.
///
/// # Arguments
///
/// * `binary_string` - Binary sequence to analyze.
///
/// # Returns
///
/// Number of distinct phrases (complexity count). Returns 0 for empty input.
pub fn lempel_ziv_complexity(binary_string: &[bool]) -> usize {
    let n = binary_string.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 1; // First symbol is always new
    let mut i = 1; // Current position

    while i < n {
        let (match_len, reached_end) = find_longest_match(binary_string, i);
        if !reached_end {
            complexity += 1;
        }
        i += match_len + 1;
    }

    complexity
}

/// Normalized LZ complexity: LZ(s) / (n / log2(n)).
///
/// A normalized complexity close to 1 indicates a random sequence,
/// while values close to 0 indicate a highly structured sequence.
///
/// # Arguments
///
/// * `binary_string` - Binary sequence to analyze.
///
/// # Returns
///
/// Normalized complexity in approximately [0, 1]. Returns 0.0 for sequences
/// of length <= 1.
pub fn normalized_lz_complexity(binary_string: &[bool]) -> f64 {
    let n = binary_string.len();
    if n <= 1 {
        return 0.0;
    }

    let lz = lempel_ziv_complexity(binary_string) as f64;
    let normalizer = n as f64 / (n as f64).log2();

    lz / normalizer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz_constant() {
        // All same symbol: complexity should be 1
        let s = vec![false; 20];
        let c = lempel_ziv_complexity(&s);
        assert_eq!(c, 1);
    }

    #[test]
    fn test_lz_alternating() {
        let s = vec![true, false, true, false, true, false, true, false];
        let c = lempel_ziv_complexity(&s);
        // Should have moderate complexity
        assert!(c >= 2);
        assert!(c <= s.len());
    }

    #[test]
    fn test_lz_all_unique_patterns() {
        // Short diverse string
        let s = vec![true, false, false, true, true, true, false, false];
        let c = lempel_ziv_complexity(&s);
        assert!(c >= 1);
    }

    #[test]
    fn test_lz_empty() {
        let c = lempel_ziv_complexity(&[]);
        assert_eq!(c, 0);
    }

    #[test]
    fn test_lz_single() {
        let c = lempel_ziv_complexity(&[true]);
        assert_eq!(c, 1);
    }

    #[test]
    fn test_normalized_lz() {
        // Constant string: low normalized complexity
        let constant = vec![false; 50];
        let nc = normalized_lz_complexity(&constant);
        assert!(nc < 0.3);
    }

    #[test]
    fn test_normalized_lz_empty() {
        let nc = normalized_lz_complexity(&[]);
        assert_eq!(nc, 0.0);
    }

    #[test]
    fn test_normalized_lz_single() {
        let nc = normalized_lz_complexity(&[true]);
        assert_eq!(nc, 0.0);
    }
}
