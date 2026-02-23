/// Plug-in entropy estimator from a sequence of symbols.
///
/// Estimates Shannon entropy by counting symbol frequencies in the sequence.
/// H = -sum(p_i * log2(p_i))
/// where p_i = count(symbol_i) / total_count
///
/// # Arguments
///
/// * `sequence` - Integer-encoded symbol sequence, with symbols in `0..num_symbols`.
/// * `num_symbols` - Total number of possible symbols (alphabet size).
///
/// # Returns
///
/// Estimated entropy in bits. Returns 0.0 if the sequence is empty or `num_symbols` is 0.
pub fn plugin_entropy(sequence: &[usize], num_symbols: usize) -> f64 {
    if sequence.is_empty() || num_symbols == 0 {
        return 0.0;
    }

    // Count occurrences of each symbol
    let mut counts = vec![0usize; num_symbols];
    for &s in sequence {
        if s < num_symbols {
            counts[s] += 1;
        }
    }

    let n = sequence.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_entropy_uniform() {
        // Uniform distribution over 4 symbols
        let sequence: Vec<usize> = (0..4000).map(|i| i % 4).collect();
        let h = plugin_entropy(&sequence, 4);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_plugin_entropy_single_symbol() {
        let sequence = vec![0; 100];
        let h = plugin_entropy(&sequence, 3);
        assert!((h).abs() < 1e-10);
    }

    #[test]
    fn test_plugin_entropy_binary() {
        // Equal probability binary
        let mut sequence = vec![0; 500];
        sequence.extend(vec![1; 500]);
        let h = plugin_entropy(&sequence, 2);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_plugin_entropy_empty() {
        let h = plugin_entropy(&[], 4);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_plugin_entropy_zero_symbols() {
        let h = plugin_entropy(&[0, 1, 2], 0);
        assert_eq!(h, 0.0);
    }
}
