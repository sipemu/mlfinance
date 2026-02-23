//! Clustering algorithms for feature grouping.
//!
//! Implements the Optimal Number of Clusters (ONC) algorithm using K-Means
//! with silhouette scoring, as described in AFML Chapter 4.

/// Optimal Number of Clusters (ONC) using K-Means with silhouette scoring.
pub mod onc;

pub use onc::{
    cluster_kmeans_base, cluster_kmeans_top, get_feature_clusters, kmeans, silhouette_score,
    ClusterMap, KMeansResult, OncResult,
};
