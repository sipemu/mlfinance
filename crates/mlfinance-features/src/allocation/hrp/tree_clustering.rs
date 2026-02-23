use ndarray::Array2;

/// Compute correlation distance matrix: d\[i,j\] = sqrt(0.5 * (1 - corr\[i,j\])).
///
/// # Arguments
///
/// * `corr` - Correlation matrix (n x n) with values in [-1, 1].
///
/// # Returns
///
/// Distance matrix (n x n) with values in [0, 1].
pub fn correlation_distance(corr: &Array2<f64>) -> Array2<f64> {
    let n = corr.nrows();
    let mut dist = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let r = corr[[i, j]].clamp(-1.0, 1.0);
            dist[[i, j]] = (0.5 * (1.0 - r)).max(0.0).sqrt();
        }
    }
    dist
}

/// Collect indices of active clusters.
fn active_indices(active: &[bool]) -> Vec<usize> {
    active
        .iter()
        .enumerate()
        .filter(|(_, &a)| a)
        .map(|(i, _)| i)
        .collect()
}

/// Find the closest pair of active clusters.
fn find_closest_pair(d: &Array2<f64>, active: &[bool]) -> (usize, usize, f64) {
    let indices = active_indices(active);
    let mut min_dist = f64::INFINITY;
    let mut min_i = 0;
    let mut min_j = 0;
    for (ai, &i) in indices.iter().enumerate() {
        for &j in &indices[ai + 1..] {
            if d[[i, j]] < min_dist {
                min_dist = d[[i, j]];
                min_i = i;
                min_j = j;
            }
        }
    }
    (min_i, min_j, min_dist)
}

/// Update distances after merging clusters `ci` and `cj` (single linkage = min).
fn update_distances_single_linkage(d: &mut Array2<f64>, active: &[bool], ci: usize, cj: usize) {
    for k in 0..d.nrows() {
        if !active[k] || k == ci || k == cj {
            continue;
        }
        let new_dist = d[[ci, k]].min(d[[cj, k]]);
        d[[ci, k]] = new_dist;
        d[[k, ci]] = new_dist;
    }
}

/// Single-linkage hierarchical clustering.
///
/// Returns linkage matrix where each row is `[idx1, idx2, distance, cluster_size]`.
///
/// # Arguments
///
/// * `dist` - Symmetric distance matrix (n x n).
///
/// # Returns
///
/// Linkage matrix with (n-1) rows encoding the merge history.
pub fn single_linkage_clustering(dist: &Array2<f64>) -> Vec<[f64; 4]> {
    let n = dist.nrows();
    if n <= 1 {
        return vec![];
    }

    let mut d = dist.clone();
    let mut active: Vec<bool> = vec![true; n];
    let mut sizes: Vec<usize> = vec![1; n];
    let mut linkage: Vec<[f64; 4]> = Vec::with_capacity(n - 1);
    let mut labels: Vec<usize> = (0..n).collect();

    for step in 0..(n - 1) {
        let (min_i, min_j, min_dist) = find_closest_pair(&d, &active);
        let new_size = sizes[min_i] + sizes[min_j];

        linkage.push([
            labels[min_i] as f64,
            labels[min_j] as f64,
            min_dist,
            new_size as f64,
        ]);

        update_distances_single_linkage(&mut d, &active, min_i, min_j);

        active[min_j] = false;
        sizes[min_i] = new_size;
        labels[min_i] = n + step;
    }

    linkage
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_correlation_distance() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let dist = correlation_distance(&corr);
        assert!((dist[[0, 0]]).abs() < 1e-10);
        assert!((dist[[1, 1]]).abs() < 1e-10);
        // d = sqrt(0.5 * (1 - 0.5)) = sqrt(0.25) = 0.5
        assert!((dist[[0, 1]] - 0.5).abs() < 1e-10);
        assert!((dist[[1, 0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_distance_perfect() {
        let corr = array![[1.0, 1.0], [1.0, 1.0]];
        let dist = correlation_distance(&corr);
        assert!((dist[[0, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_distance_negative() {
        let corr = array![[1.0, -1.0], [-1.0, 1.0]];
        let dist = correlation_distance(&corr);
        // d = sqrt(0.5 * (1 - (-1))) = sqrt(1) = 1.0
        assert!((dist[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_linkage_basic() {
        // 3 items: 0-1 close, 2 far
        let dist = array![[0.0, 0.1, 0.9], [0.1, 0.0, 0.8], [0.9, 0.8, 0.0]];
        let linkage = single_linkage_clustering(&dist);
        assert_eq!(linkage.len(), 2);
        // First merge: 0 and 1 (distance 0.1)
        assert!((linkage[0][2] - 0.1).abs() < 1e-10);
        assert!((linkage[0][3] - 2.0).abs() < 1e-10);
        // Second merge: cluster(0,1) and 2 (distance min(0.9, 0.8) = 0.8)
        assert!((linkage[1][2] - 0.8).abs() < 1e-10);
        assert!((linkage[1][3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_linkage_single_item() {
        let dist = array![[0.0]];
        let linkage = single_linkage_clustering(&dist);
        assert!(linkage.is_empty());
    }
}
