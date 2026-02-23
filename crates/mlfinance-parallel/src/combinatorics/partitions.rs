/// Number of ways to distribute `k` indistinguishable objects into `n` distinct slots.
///
/// Uses the stars-and-bars formula: C(n + k - 1, k).
/// Equivalent to Snippet 21.1.
///
/// # Arguments
///
/// * `n` - Number of slots.
/// * `k` - Number of objects to distribute.
///
/// # Returns
///
/// The count of multiset combinations. Returns 0 when `n` is zero.
pub fn partitions_count(n: usize, k: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    binomial(n + k - 1, k)
}

/// Enumerate all partitions of `k` objects into `n` slots.
///
/// Each partition is represented as a `Vec<usize>` of length `n` whose entries
/// sum to `k`.
///
/// # Arguments
///
/// * `n` - Number of slots.
/// * `k` - Number of objects to distribute.
///
/// # Returns
///
/// All distinct distributions, one per element of the returned vector.
pub fn generate_partitions(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0usize; n];
    generate_partitions_helper(n, k, 0, &mut current, &mut result);
    result
}

fn generate_partitions_helper(
    n: usize,
    remaining: usize,
    slot: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if slot == n - 1 {
        current[slot] = remaining;
        result.push(current.clone());
        current[slot] = 0;
        return;
    }
    for i in 0..=remaining {
        current[slot] = i;
        generate_partitions_helper(n, remaining - i, slot + 1, current, result);
    }
    current[slot] = 0;
}

/// Compute the binomial coefficient C(n, k).
///
/// # Arguments
///
/// * `n` - Total number of items.
/// * `k` - Number of items to choose.
///
/// # Returns
///
/// The number of ways to choose `k` items from `n`. Returns 0 when `k > n`.
pub fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(0, 0), 1);
    }

    #[test]
    fn test_partitions_count() {
        // 3 objects into 2 slots: C(4,3) = 4 partitions: (0,3),(1,2),(2,1),(3,0)
        assert_eq!(partitions_count(2, 3), 4);
    }

    #[test]
    fn test_generate_partitions() {
        let parts = generate_partitions(2, 3);
        assert_eq!(parts.len(), 4);
        for p in &parts {
            assert_eq!(p.iter().sum::<usize>(), 3);
        }
    }
}
