/// Quasi-diagonalization: reorder rows/columns based on linkage.
///
/// Recursive algorithm: for each cluster node, split into left and right
/// subclusters, recurse. Leaf nodes return a single index.
///
/// # Arguments
///
/// * `linkage` - Linkage matrix from hierarchical clustering, each row is
///   `[idx1, idx2, distance, cluster_size]`.
/// * `n` - Number of original items (leaf nodes).
///
/// # Returns
///
/// Permutation of indices `0..n` that quasi-diagonalizes the correlation matrix.
pub fn quasi_diag(linkage: &[[f64; 4]], n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }
    // The root of the dendrogram is the last merge, with label n + len(linkage) - 1
    let root = n + linkage.len() - 1;
    sort_cluster(linkage, root, n)
}

/// Recursively sort a cluster node.
/// If the node index < n, it is a leaf (original item).
/// Otherwise, it is a merged cluster at linkage[node - n].
fn sort_cluster(linkage: &[[f64; 4]], node: usize, n: usize) -> Vec<usize> {
    if node < n {
        // Leaf node
        return vec![node];
    }

    let row = &linkage[node - n];
    let left = row[0] as usize;
    let right = row[1] as usize;

    let left_sorted = sort_cluster(linkage, left, n);
    let right_sorted = sort_cluster(linkage, right, n);

    let mut result = Vec::with_capacity(left_sorted.len() + right_sorted.len());
    result.extend(left_sorted);
    result.extend(right_sorted);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quasi_diag_simple() {
        // 3 items: linkage merges 0,1 first, then {0,1} with 2
        // linkage[0] = [0, 1, 0.1, 2] -> cluster 3
        // linkage[1] = [3, 2, 0.8, 3] -> cluster 4
        let linkage = vec![[0.0, 1.0, 0.1, 2.0], [3.0, 2.0, 0.8, 3.0]];
        let sorted = quasi_diag(&linkage, 3);
        assert_eq!(sorted.len(), 3);
        // Should contain all indices
        let mut s = sorted.clone();
        s.sort();
        assert_eq!(s, vec![0, 1, 2]);
    }

    #[test]
    fn test_quasi_diag_four_items() {
        // 4 items:
        // linkage[0] = [0, 1, 0.1, 2] -> cluster 4
        // linkage[1] = [2, 3, 0.2, 2] -> cluster 5
        // linkage[2] = [4, 5, 0.5, 4] -> cluster 6
        let linkage = vec![
            [0.0, 1.0, 0.1, 2.0],
            [2.0, 3.0, 0.2, 2.0],
            [4.0, 5.0, 0.5, 4.0],
        ];
        let sorted = quasi_diag(&linkage, 4);
        assert_eq!(sorted.len(), 4);
        // left subtree: [0, 1], right subtree: [2, 3]
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_quasi_diag_single() {
        let sorted = quasi_diag(&[], 1);
        assert_eq!(sorted, vec![0]);
    }

    #[test]
    fn test_quasi_diag_empty() {
        let sorted = quasi_diag(&[], 0);
        assert!(sorted.is_empty());
    }
}
