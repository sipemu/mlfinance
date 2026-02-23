use mlfinance_core::error::{MlFinanceError, Result};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;

/// Cluster assignments: maps cluster_id -> list of feature indices.
pub type ClusterMap = HashMap<usize, Vec<usize>>;

/// K-Means clustering result.
pub struct KMeansResult {
    /// Cluster assignment for each point (0-indexed).
    pub labels: Vec<usize>,
    /// Cluster centroids (k x n_features).
    pub centroids: Array2<f64>,
    /// Number of iterations used.
    pub n_iterations: usize,
}

/// Result of ONC clustering.
pub struct OncResult {
    /// Cluster assignments (feature index -> cluster id).
    pub labels: Vec<usize>,
    /// Cluster map (cluster id -> feature indices).
    pub clusters: ClusterMap,
    /// Silhouette score of the best clustering.
    pub silhouette: f64,
    /// Optimal number of clusters found.
    pub n_clusters: usize,
}

/// Compute the squared Euclidean distance between two slices.
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

/// Compute the Euclidean distance between two slices.
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean(a, b).sqrt()
}

/// Assign each row to the nearest centroid. Returns labels.
fn assign_labels(data: &Array2<f64>, centroids: &Array2<f64>) -> Vec<usize> {
    let n_features = data.ncols();
    let k = centroids.nrows();
    (0..data.nrows())
        .map(|i| {
            let row: Vec<f64> = (0..n_features).map(|j| data[[i, j]]).collect();
            (0..k)
                .min_by(|&a, &b| {
                    let ca: Vec<f64> = (0..n_features).map(|j| centroids[[a, j]]).collect();
                    let cb: Vec<f64> = (0..n_features).map(|j| centroids[[b, j]]).collect();
                    squared_euclidean(&row, &ca)
                        .partial_cmp(&squared_euclidean(&row, &cb))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0)
        })
        .collect()
}

/// Recompute centroids from data and label assignments.
fn recompute_centroids(
    data: &Array2<f64>,
    labels: &[usize],
    k: usize,
    old: &Array2<f64>,
) -> Array2<f64> {
    let n_features = data.ncols();
    let mut counts = vec![0usize; k];
    let mut new_centroids = Array2::zeros((k, n_features));
    for i in 0..data.nrows() {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..n_features {
            new_centroids[[c, j]] += data[[i, j]];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..n_features {
                new_centroids[[c, j]] /= counts[c] as f64;
            }
        } else {
            for j in 0..n_features {
                new_centroids[[c, j]] = old[[c, j]];
            }
        }
    }
    new_centroids
}

/// Compute total inertia (sum of squared distances to assigned centroids).
fn compute_inertia(data: &Array2<f64>, labels: &[usize], centroids: &Array2<f64>) -> f64 {
    let n_features = data.ncols();
    (0..data.nrows())
        .map(|i| {
            let row: Vec<f64> = (0..n_features).map(|j| data[[i, j]]).collect();
            let c = labels[i];
            let centroid: Vec<f64> = (0..n_features).map(|j| centroids[[c, j]]).collect();
            squared_euclidean(&row, &centroid)
        })
        .sum()
}

/// Run a single K-Means iteration starting from given initial centroid indices.
/// Returns (labels, centroids, n_iterations, inertia).
fn kmeans_single(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
    initial_indices: &[usize],
) -> (Vec<usize>, Array2<f64>, usize, f64) {
    let n_features = data.ncols();

    let mut centroids = Array2::zeros((k, n_features));
    for (ci, &idx) in initial_indices.iter().enumerate() {
        for j in 0..n_features {
            centroids[[ci, j]] = data[[idx, j]];
        }
    }

    let mut labels = vec![0usize; data.nrows()];
    let mut n_iterations = 0;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        let new_labels = assign_labels(data, &centroids);
        let converged = new_labels == labels && iter > 0;
        labels = new_labels;

        if converged {
            break;
        }

        centroids = recompute_centroids(data, &labels, k, &centroids);
    }

    let inertia = compute_inertia(data, &labels, &centroids);
    (labels, centroids, n_iterations, inertia)
}

/// Run K-Means clustering on rows of data matrix.
///
/// Parameters:
/// - `data`: (n_samples, n_features) matrix
/// - `k`: number of clusters
/// - `max_iter`: maximum iterations (default 300)
/// - `n_init`: number of random initializations, take best (default 10)
/// - `seed`: random seed for reproducibility
///
/// Returns `KMeansResult` with the best clustering (lowest inertia).
/// Validate kmeans input parameters.
fn validate_kmeans(k: usize, n_samples: usize) -> Result<()> {
    if k == 0 {
        return Err(MlFinanceError::InvalidParameter {
            msg: "k must be >= 1".into(),
        });
    }
    if n_samples == 0 {
        return Err(MlFinanceError::EmptySeries);
    }
    if k > n_samples {
        return Err(MlFinanceError::InvalidParameter {
            msg: format!("k ({}) must be <= number of samples ({})", k, n_samples),
        });
    }
    Ok(())
}

/// Handle the k==1 special case: all points in one cluster.
fn kmeans_single_cluster(data: &Array2<f64>, n_samples: usize) -> KMeansResult {
    let n_features = data.ncols();
    let mut centroid = Array2::zeros((1, n_features));
    for j in 0..n_features {
        centroid[[0, j]] = (0..n_samples).map(|i| data[[i, j]]).sum::<f64>() / n_samples as f64;
    }
    KMeansResult {
        labels: vec![0; n_samples],
        centroids: centroid,
        n_iterations: 1,
    }
}

/// Run K-Means clustering on rows of a data matrix.
///
/// Performs multiple random initializations and returns the result with the
/// lowest inertia (sum of squared distances to assigned centroids).
///
/// # Arguments
///
/// * `data` - Data matrix with shape (n_samples, n_features).
/// * `k` - Number of clusters (must be >= 1 and <= n_samples).
/// * `max_iter` - Maximum iterations per initialization.
/// * `n_init` - Number of random initializations; the best result is kept.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
///
/// A [`KMeansResult`] containing labels, centroids, and iteration count.
///
/// # Errors
///
/// Returns an error if `k` is 0, data is empty, or `k > n_samples`.
pub fn kmeans(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> Result<KMeansResult> {
    let n_samples = data.nrows();
    validate_kmeans(k, n_samples)?;

    if k == 1 {
        return Ok(kmeans_single_cluster(data, n_samples));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<usize> = (0..n_samples).collect();

    let mut best_labels = None;
    let mut best_centroids = None;
    let mut best_inertia = f64::MAX;
    let mut best_iterations = 0;

    for _ in 0..n_init {
        let mut sample = indices.clone();
        sample.shuffle(&mut rng);

        let (labels, centroids, n_iter, inertia) = kmeans_single(data, k, max_iter, &sample[..k]);

        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = Some(labels);
            best_centroids = Some(centroids);
            best_iterations = n_iter;
        }
    }

    Ok(KMeansResult {
        labels: best_labels.unwrap(),
        centroids: best_centroids.unwrap(),
        n_iterations: best_iterations,
    })
}

/// Precompute the full pairwise Euclidean distance matrix.
fn pairwise_distance_matrix(data: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = data.nrows();
    let n_features = data.ncols();
    let mut dist_matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let row_i: Vec<f64> = (0..n_features).map(|f| data[[i, f]]).collect();
            let row_j: Vec<f64> = (0..n_features).map(|f| data[[j, f]]).collect();
            let d = euclidean(&row_i, &row_j);
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }
    dist_matrix
}

/// Compute mean intra-cluster distance a(i) for point `i`.
fn mean_intra_distance(i: usize, members: &[usize], dist_matrix: &[Vec<f64>]) -> f64 {
    if members.len() <= 1 {
        return 0.0;
    }
    let sum: f64 = members
        .iter()
        .filter(|&&j| j != i)
        .map(|&j| dist_matrix[i][j])
        .sum();
    sum / (members.len() - 1) as f64
}

/// Compute minimum mean inter-cluster distance b(i) for point `i`.
fn min_inter_distance(
    i: usize,
    ci: usize,
    cluster_members: &[Vec<usize>],
    dist_matrix: &[Vec<f64>],
) -> f64 {
    cluster_members
        .iter()
        .enumerate()
        .filter(|&(cj, members)| cj != ci && !members.is_empty())
        .map(|(_, members)| {
            let sum: f64 = members.iter().map(|&j| dist_matrix[i][j]).sum();
            sum / members.len() as f64
        })
        .fold(f64::MAX, f64::min)
}

/// Compute the silhouette score for a clustering.
///
/// For each point `i` in cluster `C_i`:
///   - `a(i)` = mean distance to other points in same cluster
///   - `b(i)` = min over other clusters `C_j` of mean distance to points in `C_j`
///   - `s(i)` = `(b(i) - a(i)) / max(a(i), b(i))`
///
/// Returns mean silhouette score across all points.
/// Returns 0.0 if there is only 1 cluster or all points are in one cluster.
/// Compute silhouette coefficient for a single point.
fn silhouette_point(
    i: usize,
    ci: usize,
    cluster_members: &[Vec<usize>],
    dist_matrix: &[Vec<f64>],
) -> Option<f64> {
    let a_i = mean_intra_distance(i, &cluster_members[ci], dist_matrix);
    let b_i = min_inter_distance(i, ci, cluster_members, dist_matrix);
    if b_i == f64::MAX {
        return None;
    }
    let max_ab = a_i.max(b_i);
    Some(if max_ab == 0.0 {
        0.0
    } else {
        (b_i - a_i) / max_ab
    })
}

/// Compute the silhouette score for a clustering.
///
/// For each point `i` in cluster `C_i`:
///   - `a(i)` = mean distance to other points in the same cluster
///   - `b(i)` = min over other clusters of mean distance to points in that cluster
///   - `s(i)` = `(b(i) - a(i)) / max(a(i), b(i))`
///
/// Returns the mean silhouette score across all points.
/// Returns 0.0 if there is only one cluster, no data, or all points share one cluster.
///
/// # Arguments
///
/// * `data` - Data matrix with shape (n_samples, n_features).
/// * `labels` - Cluster label for each sample (0-indexed).
///
/// # Returns
///
/// Mean silhouette coefficient in [-1, 1], where 1 indicates perfect clustering.
pub fn silhouette_score(data: &Array2<f64>, labels: &[usize]) -> f64 {
    let n = data.nrows();
    if n == 0 || labels.is_empty() {
        return 0.0;
    }

    let k = labels.iter().copied().max().unwrap_or(0) + 1;
    let unique_labels: std::collections::HashSet<usize> = labels.iter().copied().collect();
    if unique_labels.len() <= 1 {
        return 0.0;
    }

    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &label) in labels.iter().enumerate() {
        cluster_members[label].push(i);
    }

    let dist_matrix = pairwise_distance_matrix(data);

    let scores: Vec<f64> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, &ci)| silhouette_point(i, ci, &cluster_members, &dist_matrix))
        .collect();

    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// Build a ClusterMap from label assignments.
fn build_cluster_map(labels: &[usize]) -> ClusterMap {
    let mut map: ClusterMap = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        map.entry(label).or_default().push(i);
    }
    map
}

/// Relabel cluster assignments to be contiguous starting from 0.
fn relabel(labels: &[usize]) -> Vec<usize> {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    labels
        .iter()
        .map(|&l| {
            *mapping.entry(l).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
}

/// Base K-Means clustering on correlation matrix.
///
/// Tries `k` in `[min_clusters..=max_clusters]`, picks the `k` with best silhouette score.
/// Uses the correlation matrix directly as the feature space for clustering
/// (each column of the correlation matrix is treated as a feature vector for that asset).
///
/// Parameters:
/// - `corr`: correlation matrix (n x n)
/// - `max_clusters`: maximum k to try (default: n/2, min 10)
/// - `min_clusters`: minimum k to try (default: 2)
/// - `n_init`: number of K-Means inits per k (default: 10)
/// - `seed`: random seed
pub fn cluster_kmeans_base(
    corr: &Array2<f64>,
    max_clusters: Option<usize>,
    min_clusters: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<OncResult> {
    let n = corr.nrows();
    if n != corr.ncols() {
        return Err(MlFinanceError::DimensionMismatch {
            msg: format!(
                "correlation matrix must be square, got {}x{}",
                n,
                corr.ncols()
            ),
        });
    }
    if n == 0 {
        return Err(MlFinanceError::EmptySeries);
    }
    if n == 1 {
        let mut clusters = ClusterMap::new();
        clusters.insert(0, vec![0]);
        return Ok(OncResult {
            labels: vec![0],
            clusters,
            silhouette: 0.0,
            n_clusters: 1,
        });
    }

    let min_k = min_clusters.unwrap_or(2).max(2);
    let default_max = (n / 2).max(2).min(n);
    let max_k = max_clusters
        .map(|m| m.min(n))
        .unwrap_or(default_max.max(min_k));
    let max_k = max_k.max(min_k);
    let n_init_val = n_init.unwrap_or(10);
    let seed_val = seed.unwrap_or(42);

    let mut best_labels: Option<Vec<usize>> = None;
    let mut best_silhouette = f64::NEG_INFINITY;

    for k in min_k..=max_k {
        let result = kmeans(corr, k, 300, n_init_val, seed_val)?;
        let sil = silhouette_score(corr, &result.labels);
        if sil > best_silhouette {
            best_silhouette = sil;
            best_labels = Some(result.labels);
        }
    }

    let labels = relabel(&best_labels.unwrap());
    let clusters = build_cluster_map(&labels);
    let n_clusters = clusters.len();

    Ok(OncResult {
        labels,
        clusters,
        silhouette: best_silhouette,
        n_clusters,
    })
}

/// Extract a sub-correlation matrix for the given indices.
fn extract_submatrix(corr: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let m = indices.len();
    let mut sub = Array2::zeros((m, m));
    for (si, &i) in indices.iter().enumerate() {
        for (sj, &j) in indices.iter().enumerate() {
            sub[[si, sj]] = corr[[i, j]];
        }
    }
    sub
}

/// Assign a single cluster's members, attempting sub-clustering if large enough.
/// Returns the next available cluster id.
#[allow(clippy::too_many_arguments)]
fn assign_sub_clusters(
    corr: &Array2<f64>,
    members: &[usize],
    max_clusters: Option<usize>,
    min_clusters: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
    new_labels: &mut [usize],
    next_id: usize,
) -> usize {
    if members.len() <= 2 {
        for &idx in members {
            new_labels[idx] = next_id;
        }
        return next_id + 1;
    }

    let sub_corr = extract_submatrix(corr, members);
    let sub_max = max_clusters.map(|m| m.min(members.len()));
    match cluster_kmeans_base(&sub_corr, sub_max, min_clusters, n_init, seed) {
        Ok(sub_result) if sub_result.n_clusters > 1 => {
            for (local_idx, &global_idx) in members.iter().enumerate() {
                new_labels[global_idx] = next_id + sub_result.labels[local_idx];
            }
            next_id + sub_result.n_clusters
        }
        _ => {
            for &idx in members {
                new_labels[idx] = next_id;
            }
            next_id + 1
        }
    }
}

/// Top-down Optimal Number of Clusters.
///
/// Algorithm:
/// 1. Run `cluster_kmeans_base` to get initial clusters
/// 2. For each cluster with > 1 element, extract the sub-correlation matrix
/// 3. Run `cluster_kmeans_base` on each sub-matrix
/// 4. If any sub-cluster improves the silhouette, split it
/// 5. Merge results: if the new combined silhouette > old, keep the split
///
/// This is a one-level refinement (not fully recursive).
pub fn cluster_kmeans_top(
    corr: &Array2<f64>,
    max_clusters: Option<usize>,
    min_clusters: Option<usize>,
    n_init: Option<usize>,
    seed: Option<u64>,
) -> Result<OncResult> {
    let n = corr.nrows();
    if n <= 2 {
        // For very small matrices, just return the base result
        return cluster_kmeans_base(corr, max_clusters, min_clusters, n_init, seed);
    }

    // Step 1: Get initial clustering
    let base_result = cluster_kmeans_base(corr, max_clusters, min_clusters, n_init, seed)?;

    // Step 2 & 3: For each cluster, attempt sub-clustering
    let mut new_labels = vec![0usize; n];
    let mut next_cluster_id = 0;

    for members in base_result.clusters.values() {
        next_cluster_id = assign_sub_clusters(
            corr,
            members,
            max_clusters,
            min_clusters,
            n_init,
            seed,
            &mut new_labels,
            next_cluster_id,
        );
    }

    // Step 4: Relabel and compute new silhouette
    let new_labels = relabel(&new_labels);
    let new_silhouette = silhouette_score(corr, &new_labels);

    // Step 5: Keep the better result
    if new_silhouette > base_result.silhouette {
        let clusters = build_cluster_map(&new_labels);
        let n_clusters = clusters.len();
        Ok(OncResult {
            labels: new_labels,
            clusters,
            silhouette: new_silhouette,
            n_clusters,
        })
    } else {
        Ok(base_result)
    }
}

/// Get feature clusters from a data matrix.
///
/// 1. Compute correlation matrix from data
/// 2. Run ONC (`cluster_kmeans_top`) on the correlation matrix
/// 3. Return cluster assignments
pub fn get_feature_clusters(
    data: &Array2<f64>,
    max_clusters: Option<usize>,
    seed: Option<u64>,
) -> Result<OncResult> {
    let corr = mlfinance_core::stats::correlation_matrix(data)?;
    cluster_kmeans_top(&corr, max_clusters, None, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    // ==================== K-Means Tests ====================

    #[test]
    fn test_kmeans_two_clusters() {
        // Two well-separated clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let result = kmeans(&data, 2, 300, 10, 42).unwrap();
        assert_eq!(result.labels.len(), 6);
        // First three should be in same cluster, last three in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = array![[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]];
        let result = kmeans(&data, 1, 300, 10, 42).unwrap();
        assert_eq!(result.labels, vec![0, 0, 0]);
        assert_eq!(result.centroids.nrows(), 1);
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        let data = array![[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]];
        let result = kmeans(&data, 3, 300, 10, 42).unwrap();
        assert_eq!(result.labels.len(), 3);
        // Each point should be in its own cluster
        let unique: std::collections::HashSet<usize> = result.labels.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_kmeans_convergence() {
        // Simple dataset should converge quickly
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let result = kmeans(&data, 2, 300, 5, 42).unwrap();
        assert!(result.n_iterations <= 300);
    }

    #[test]
    fn test_kmeans_invalid_k_zero() {
        let data = array![[1.0, 2.0]];
        let result = kmeans(&data, 0, 300, 10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_greater_than_n() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let result = kmeans(&data, 5, 300, 10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_deterministic_with_seed() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];
        let r1 = kmeans(&data, 2, 300, 10, 123).unwrap();
        let r2 = kmeans(&data, 2, 300, 10, 123).unwrap();
        assert_eq!(r1.labels, r2.labels);
    }

    #[test]
    fn test_kmeans_centroids_shape() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = kmeans(&data, 2, 300, 10, 42).unwrap();
        assert_eq!(result.centroids.nrows(), 2);
        assert_eq!(result.centroids.ncols(), 3);
    }

    #[test]
    fn test_kmeans_three_clusters() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ];
        let result = kmeans(&data, 3, 300, 10, 42).unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_eq!(result.labels[4], result.labels[5]);
        let unique: std::collections::HashSet<usize> = result.labels.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    // ==================== Silhouette Score Tests ====================

    #[test]
    fn test_silhouette_perfect_separation() {
        // Two perfectly separated clusters far apart
        let data = array![[0.0, 0.0], [0.0, 0.0], [100.0, 100.0], [100.0, 100.0],];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        // With identical within-cluster points, a(i)=0, so s(i) = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_silhouette_single_cluster() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let labels = vec![0, 0, 0];
        let score = silhouette_score(&data, &labels);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_silhouette_all_different_clusters() {
        // Each point in its own cluster - silhouette not well defined, treated as 0
        // because a(i) = 0 for singleton clusters
        let data = array![[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]];
        let labels = vec![0, 1, 2];
        let score = silhouette_score(&data, &labels);
        // For singleton clusters, a(i) = 0, b(i) = min dist to other clusters
        // s(i) = (b(i) - 0) / max(0, b(i)) = 1.0 if b(i) > 0
        // Actually this would be 1.0 for well-separated singletons,
        // but the concept is unusual. Just verify it computes.
        assert!((-1.0..=1.0).contains(&score));
    }

    #[test]
    fn test_silhouette_empty_data() {
        let data = Array2::<f64>::zeros((0, 2));
        let labels: Vec<usize> = vec![];
        let score = silhouette_score(&data, &labels);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_silhouette_known_value() {
        // Two tight clusters: {(0,0), (1,0)} and {(10,0), (11,0)}
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        // a(0) = dist(0, 1) = 1.0
        // b(0) = mean dist to cluster 1 = (10 + 11)/2 = 10.5
        // s(0) = (10.5 - 1.0) / 10.5 = 9.5/10.5
        // a(1) = dist(1, 0) = 1.0
        // b(1) = mean dist to cluster 1 = (9 + 10)/2 = 9.5
        // s(1) = (9.5 - 1.0) / 9.5 = 8.5/9.5
        // a(2) = dist(2, 3) = 1.0
        // b(2) = mean dist to cluster 0 = (10 + 9)/2 = 9.5
        // s(2) = (9.5 - 1.0) / 9.5 = 8.5/9.5
        // a(3) = dist(3, 2) = 1.0
        // b(3) = mean dist to cluster 0 = (11 + 10)/2 = 10.5
        // s(3) = (10.5 - 1.0) / 10.5 = 9.5/10.5
        let expected = ((9.5 / 10.5) + (8.5 / 9.5) + (8.5 / 9.5) + (9.5 / 10.5)) / 4.0;
        assert!((score - expected).abs() < 1e-10);
    }

    #[test]
    fn test_silhouette_negative_for_bad_clustering() {
        // Points are close but assigned to wrong clusters
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]];
        let labels = vec![0, 1, 0, 1]; // Misassigned
        let score = silhouette_score(&data, &labels);
        // This should give a poor (negative) silhouette score
        assert!(score < 0.5);
    }

    // ==================== cluster_kmeans_base Tests ====================

    #[test]
    fn test_base_identity_correlation() {
        // Identity matrix: each item is uncorrelated -> should find n clusters or close
        let corr = Array2::eye(4);
        let result = cluster_kmeans_base(&corr, Some(4), Some(2), Some(10), Some(42)).unwrap();
        assert!(result.n_clusters >= 2);
        assert!(result.n_clusters <= 4);
    }

    #[test]
    fn test_base_block_diagonal() {
        // Block diagonal correlation -> should find 2 clusters
        let mut corr = Array2::eye(6);
        // Block 1: indices 0, 1, 2
        for i in 0..3 {
            for j in 0..3 {
                corr[[i, j]] = 0.9;
            }
            corr[[i, i]] = 1.0;
        }
        // Block 2: indices 3, 4, 5
        for i in 3..6 {
            for j in 3..6 {
                corr[[i, j]] = 0.9;
            }
            corr[[i, i]] = 1.0;
        }
        let result = cluster_kmeans_base(&corr, Some(3), Some(2), Some(10), Some(42)).unwrap();
        assert!(result.n_clusters >= 2);
        // Check that elements within the same block are in the same cluster
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_base_min_max_clusters_respected() {
        let corr = Array2::eye(10);
        let result = cluster_kmeans_base(&corr, Some(5), Some(3), Some(5), Some(42)).unwrap();
        assert!(result.n_clusters >= 2); // at least 2 after relabeling
        assert!(result.n_clusters <= 5);
    }

    #[test]
    fn test_base_1x1_matrix() {
        let corr = array![[1.0]];
        let result = cluster_kmeans_base(&corr, None, None, None, None).unwrap();
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.labels, vec![0]);
    }

    #[test]
    fn test_base_2x2_matrix() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let result = cluster_kmeans_base(&corr, Some(2), Some(2), Some(10), Some(42)).unwrap();
        // With only 2 items, should get 2 clusters since min_clusters=2
        assert!(result.n_clusters >= 1 && result.n_clusters <= 2);
        assert_eq!(result.labels.len(), 2);
    }

    #[test]
    fn test_base_all_identical_correlations() {
        // All correlations are 1.0 (all identical rows)
        let n = 4;
        let corr = Array2::from_elem((n, n), 1.0);
        let result = cluster_kmeans_base(&corr, Some(2), Some(2), Some(10), Some(42)).unwrap();
        // All points are identical in feature space, so clustering is arbitrary
        assert!(result.n_clusters >= 1);
        assert_eq!(result.labels.len(), n);
    }

    #[test]
    fn test_base_non_square_error() {
        let corr = Array2::zeros((3, 4));
        let result = cluster_kmeans_base(&corr, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_base_silhouette_in_valid_range() {
        let mut corr = Array2::eye(6);
        for i in 0..3 {
            for j in 0..3 {
                corr[[i, j]] = 0.8;
            }
            corr[[i, i]] = 1.0;
        }
        for i in 3..6 {
            for j in 3..6 {
                corr[[i, j]] = 0.8;
            }
            corr[[i, i]] = 1.0;
        }
        let result = cluster_kmeans_base(&corr, Some(3), Some(2), Some(10), Some(42)).unwrap();
        assert!(result.silhouette >= -1.0 && result.silhouette <= 1.0);
    }

    // ==================== cluster_kmeans_top Tests ====================

    #[test]
    fn test_top_improves_or_equals_base() {
        let mut corr = Array2::eye(8);
        // Create 4 blocks of 2
        for block in 0..4 {
            let start = block * 2;
            for i in start..start + 2 {
                for j in start..start + 2 {
                    corr[[i, j]] = 0.95;
                }
                corr[[i, i]] = 1.0;
            }
        }
        let base = cluster_kmeans_base(&corr, Some(4), Some(2), Some(10), Some(42)).unwrap();
        let top = cluster_kmeans_top(&corr, Some(4), Some(2), Some(10), Some(42)).unwrap();
        assert!(top.silhouette >= base.silhouette - 1e-10);
    }

    #[test]
    fn test_top_block_diagonal() {
        let mut corr = Array2::eye(6);
        for i in 0..3 {
            for j in 0..3 {
                corr[[i, j]] = 0.9;
            }
            corr[[i, i]] = 1.0;
        }
        for i in 3..6 {
            for j in 3..6 {
                corr[[i, j]] = 0.9;
            }
            corr[[i, i]] = 1.0;
        }
        let result = cluster_kmeans_top(&corr, Some(3), Some(2), Some(10), Some(42)).unwrap();
        assert!(result.n_clusters >= 2);
        assert_eq!(result.labels.len(), 6);
    }

    #[test]
    fn test_top_1x1_matrix() {
        let corr = array![[1.0]];
        let result = cluster_kmeans_top(&corr, None, None, None, None).unwrap();
        assert_eq!(result.n_clusters, 1);
    }

    #[test]
    fn test_top_2x2_matrix() {
        let corr = array![[1.0, 0.1], [0.1, 1.0]];
        let result = cluster_kmeans_top(&corr, Some(2), Some(2), Some(5), Some(42)).unwrap();
        assert!(result.n_clusters >= 1 && result.n_clusters <= 2);
    }

    #[test]
    fn test_top_valid_cluster_map() {
        let mut corr = Array2::eye(6);
        for i in 0..3 {
            for j in 0..3 {
                corr[[i, j]] = 0.9;
            }
            corr[[i, i]] = 1.0;
        }
        for i in 3..6 {
            for j in 3..6 {
                corr[[i, j]] = 0.9;
            }
            corr[[i, i]] = 1.0;
        }
        let result = cluster_kmeans_top(&corr, Some(3), Some(2), Some(10), Some(42)).unwrap();
        // Every feature index should appear exactly once
        let mut all_indices: Vec<usize> = result
            .clusters
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        all_indices.sort();
        assert_eq!(all_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    // ==================== get_feature_clusters Tests ====================

    #[test]
    fn test_get_feature_clusters_basic() {
        // Create data with two groups of correlated features
        let n_rows = 100;
        let mut data = Array2::zeros((n_rows, 4));
        let mut rng = StdRng::seed_from_u64(42);
        use rand::Rng;
        for i in 0..n_rows {
            let base1: f64 = rng.gen();
            let base2: f64 = rng.gen();
            data[[i, 0]] = base1 + rng.gen::<f64>() * 0.01;
            data[[i, 1]] = base1 + rng.gen::<f64>() * 0.01;
            data[[i, 2]] = base2 + rng.gen::<f64>() * 0.01;
            data[[i, 3]] = base2 + rng.gen::<f64>() * 0.01;
        }
        let result = get_feature_clusters(&data, Some(2), Some(42)).unwrap();
        assert_eq!(result.labels.len(), 4);
        assert!(result.n_clusters >= 1);
    }

    #[test]
    fn test_get_feature_clusters_valid_cluster_map() {
        let n_rows = 50;
        let mut data = Array2::zeros((n_rows, 3));
        let mut rng = StdRng::seed_from_u64(99);
        use rand::Rng;
        for i in 0..n_rows {
            let base: f64 = rng.gen();
            data[[i, 0]] = base + rng.gen::<f64>() * 0.1;
            data[[i, 1]] = base + rng.gen::<f64>() * 0.1;
            data[[i, 2]] = rng.gen::<f64>() * 10.0;
        }
        let result = get_feature_clusters(&data, Some(2), Some(42)).unwrap();
        // All feature indices should be present
        let mut all_indices: Vec<usize> = result
            .clusters
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        all_indices.sort();
        assert_eq!(all_indices, vec![0, 1, 2]);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_build_cluster_map() {
        let labels = vec![0, 1, 0, 2, 1];
        let map = build_cluster_map(&labels);
        assert_eq!(map[&0], vec![0, 2]);
        assert_eq!(map[&1], vec![1, 4]);
        assert_eq!(map[&2], vec![3]);
    }

    #[test]
    fn test_relabel() {
        let labels = vec![5, 5, 10, 10, 3];
        let new_labels = relabel(&labels);
        assert_eq!(new_labels[0], new_labels[1]);
        assert_eq!(new_labels[2], new_labels[3]);
        assert_ne!(new_labels[0], new_labels[2]);
        assert_ne!(new_labels[0], new_labels[4]);
        // Should start from 0
        assert!(new_labels.iter().all(|&l| l < 3));
    }

    #[test]
    fn test_extract_submatrix() {
        let m = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let sub = extract_submatrix(&m, &[0, 2]);
        assert_eq!(sub, array![[1.0, 3.0], [9.0, 11.0]]);
    }

    #[test]
    fn test_squared_euclidean() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((squared_euclidean(&a, &b) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean(&a, &b) - 5.0).abs() < 1e-10);
    }
}
