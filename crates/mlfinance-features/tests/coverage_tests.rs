use mlfinance_features::allocation::cla::{cla_max_sharpe, cla_min_variance};
use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use mlfinance_features::allocation::hrp::quasi_diag::quasi_diag;
use mlfinance_features::allocation::hrp::recursive_bisection::recursive_bisection;
use mlfinance_features::allocation::hrp::tree_clustering::{
    correlation_distance, single_linkage_clustering,
};
use mlfinance_features::allocation::ivp::inverse_variance_weights;
use mlfinance_features::allocation::monte_carlo::compare_allocations;
use mlfinance_features::clustering::{
    cluster_kmeans_base, cluster_kmeans_top, kmeans, silhouette_score,
};
use mlfinance_features::codependence::codependence_matrix::{
    dependence_matrix, distance_matrix, DependenceMethod, DistanceMetric,
};
use mlfinance_features::codependence::correlation::{
    absolute_angular_distance, angular_distance, distance_correlation, kullback_leibler_distance,
    squared_angular_distance,
};
use mlfinance_features::codependence::gnpr_distance::{gnpr_distance, gpr_distance, spearmans_rho};
use mlfinance_features::codependence::information::{
    mutual_information, optimal_number_of_bins, variation_of_information,
};
use mlfinance_features::codependence::optimal_transport::optimal_transport_dependence;
use mlfinance_features::denoising::{
    corr_to_cov, cov_to_corr, denoise_corr, denoise_cov, detone_corr, fit_kde, marcenko_pastur_pdf,
    optimal_portfolio,
};
use mlfinance_features::entropy::encoding::{binary_encode, quantile_encode, sigma_encode};
use mlfinance_features::entropy::gaussian_entropy::{entropy_implied_vol, gaussian_entropy};
use mlfinance_features::entropy::kontoyiannis::kontoyiannis_entropy;
use mlfinance_features::entropy::lempel_ziv::{lempel_ziv_complexity, normalized_lz_complexity};
use mlfinance_features::entropy::plugin::plugin_entropy;
use mlfinance_features::entropy::shannon::{redundancy, shannon_entropy};
use mlfinance_features::microstructure::amihud_lambda::{amihud_lambda, amihud_lambda_rolling};
use mlfinance_features::microstructure::corwin_schultz::corwin_schultz_spread;
use mlfinance_features::microstructure::hasbrouck_lambda::hasbrouck_lambda;
use mlfinance_features::microstructure::kyle_lambda::kyle_lambda;
use mlfinance_features::microstructure::order_features::{
    order_size_distribution, round_lot_fraction,
};
use mlfinance_features::microstructure::roll_model::{roll_spread, roll_spread_rolling};
use mlfinance_features::microstructure::tick_rule::tick_rule_classify;
use mlfinance_features::microstructure::vpin::vpin;
use ndarray::{array, Array1};

// ═══════════════════════════════════════════════════════════
// Encoding
// ═══════════════════════════════════════════════════════════

#[test]
fn test_binary_encode() {
    let values = vec![-1.0, 0.0, 0.5, 1.0, -0.5];
    let encoded = binary_encode(&values);
    assert_eq!(encoded.len(), 5);
    // binary_encode uses v > 0.0
    assert!(!encoded[0]); // -1.0 <= 0
    assert!(!encoded[1]); // 0.0 <= 0
    assert!(encoded[2]); // 0.5 > 0
    assert!(encoded[3]); // 1.0 > 0
    assert!(!encoded[4]); // -0.5 <= 0
}

#[test]
fn test_quantile_encode() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let encoded = quantile_encode(&values, 4);
    assert_eq!(encoded.len(), 10);
    for &e in &encoded {
        assert!(e < 4, "bin index should be < num_bins");
    }
}

#[test]
fn test_sigma_encode() {
    let values = vec![0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0];
    let encoded = sigma_encode(&values, 3);
    assert_eq!(encoded.len(), 10);
    for &e in &encoded {
        assert!(e < 3, "band index should be < num_bands");
    }
}

// ═══════════════════════════════════════════════════════════
// Microstructure
// ═══════════════════════════════════════════════════════════

#[test]
fn test_amihud_lambda() {
    let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03];
    let dollar_volumes = vec![1e6, 1.2e6, 0.9e6, 1.1e6, 1.3e6];
    let lambda = amihud_lambda(&returns, &dollar_volumes);
    assert!(lambda.is_finite());
    assert!(lambda >= 0.0);
}

#[test]
fn test_amihud_lambda_rolling() {
    let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03, 0.01, -0.01];
    let dollar_volumes = vec![1e6, 1.2e6, 0.9e6, 1.1e6, 1.3e6, 1.0e6, 1.1e6];
    let rolling = amihud_lambda_rolling(&returns, &dollar_volumes, 3);
    // Rolling window: n - window + 1 outputs
    assert_eq!(rolling.len(), returns.len() - 3 + 1);
}

#[test]
fn test_kyle_lambda() {
    let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03];
    let signed_volume = vec![100.0, -150.0, 80.0, -50.0, 200.0];
    let lambda = kyle_lambda(&returns, &signed_volume);
    assert!(lambda.is_finite());
}

#[test]
fn test_roll_spread() {
    let prices = vec![
        100.0, 100.5, 99.8, 100.3, 99.9, 100.1, 100.4, 99.7, 100.2, 100.0,
    ];
    let spread = roll_spread(&prices);
    assert!(spread.is_finite());
}

#[test]
fn test_roll_spread_rolling() {
    let prices = vec![100.0, 100.5, 99.8, 100.3, 99.9, 100.1, 100.4, 99.7];
    let rolling = roll_spread_rolling(&prices, 4);
    // Rolling window: n - window + 1 outputs
    assert_eq!(rolling.len(), prices.len() - 4 + 1);
}

#[test]
fn test_vpin() {
    let volumes: Vec<f64> = vec![
        100.0, 150.0, 120.0, 200.0, 80.0, 160.0, 140.0, 110.0, 90.0, 180.0,
    ];
    let prices: Vec<f64> = vec![
        100.0, 100.5, 99.8, 100.3, 99.9, 100.1, 100.4, 99.7, 100.2, 100.0,
    ];
    let result = vpin(&volumes, &prices, 200.0, 3);
    assert!(!result.is_empty());
    for &v in &result {
        assert!((0.0..=1.0 + 1e-10).contains(&v));
    }
}

#[test]
fn test_corwin_schultz_spread() {
    let highs = vec![101.0, 102.0, 101.5, 103.0, 102.5, 101.8, 103.5, 102.0];
    let lows = vec![99.0, 100.0, 99.5, 101.0, 100.5, 99.8, 101.5, 100.0];
    let spreads = corwin_schultz_spread(&highs, &lows);
    // Uses pairwise comparison: n-1 outputs
    assert_eq!(spreads.len(), highs.len() - 1);
}

#[test]
fn test_hasbrouck_lambda() {
    let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03, 0.01, -0.01, 0.02];
    let trade_signs = vec![1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let lambda = hasbrouck_lambda(&returns, &trade_signs, 100, 42);
    assert!(lambda.is_finite());
}

#[test]
fn test_tick_rule_classify() {
    let prices = vec![100.0, 101.0, 101.0, 100.5, 100.5, 101.0];
    let classified = tick_rule_classify(&prices);
    assert_eq!(classified.len(), prices.len());
    for &c in &classified {
        assert!(c == 1.0 || c == -1.0 || c == 0.0);
    }
}

#[test]
fn test_order_size_distribution() {
    let volumes = vec![100.0, 200.0, 150.0, 300.0, 250.0, 180.0, 120.0, 400.0];
    let features = order_size_distribution(&volumes);
    assert!(features.mean_size.is_finite());
    assert!(features.std_size >= 0.0);
}

#[test]
fn test_round_lot_fraction() {
    let volumes = vec![100.0, 200.0, 150.0, 300.0, 50.0];
    let fraction = round_lot_fraction(&volumes, 100.0);
    assert!((0.0..=1.0).contains(&fraction));
}

// ═══════════════════════════════════════════════════════════
// Codependence
// ═══════════════════════════════════════════════════════════

#[test]
fn test_angular_distance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let d = angular_distance(&x, &y).unwrap();
    // Perfect positive correlation => distance near 0
    assert!(d < 0.1);
}

#[test]
fn test_absolute_angular_distance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
    let d = absolute_angular_distance(&x, &y).unwrap();
    // Perfect negative correlation => |corr| = 1 => absolute angular distance near 0
    assert!(d < 0.1);
}

#[test]
fn test_squared_angular_distance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let d = squared_angular_distance(&x, &y).unwrap();
    assert!(d < 0.1);
}

#[test]
fn test_distance_correlation() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2 (nonlinear)
    let dcor = distance_correlation(&x, &y).unwrap();
    // Distance correlation should capture nonlinear dependence
    assert!(dcor > 0.5);
}

#[test]
fn test_kullback_leibler_distance() {
    let corr_a = array![[1.0, 0.5], [0.5, 1.0]];
    let corr_b = array![[1.0, 0.3], [0.3, 1.0]];
    let kl = kullback_leibler_distance(&corr_a, &corr_b).unwrap();
    assert!(kl >= 0.0);
}

#[test]
fn test_spearmans_rho() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let rho = spearmans_rho(&x, &y).unwrap();
    assert!((rho - 1.0).abs() < 1e-10);
}

#[test]
fn test_gpr_distance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let d = gpr_distance(&x, &y, 0.5).unwrap();
    assert!(d >= 0.0);
}

#[test]
fn test_gnpr_distance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let d = gnpr_distance(&x, &y, 0.5, None).unwrap();
    assert!(d >= 0.0);
}

#[test]
fn test_optimal_transport_dependence() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let ot = optimal_transport_dependence(&x, &y).unwrap();
    assert!(ot >= 0.0);
}

#[test]
fn test_optimal_number_of_bins() {
    let n = optimal_number_of_bins(100, Some(0.5));
    assert!(n >= 2);
    let n_no_corr = optimal_number_of_bins(100, None);
    assert!(n_no_corr >= 2);
}

#[test]
fn test_mutual_information() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mi = mutual_information(&x, &y, None, true).unwrap();
    assert!(mi >= 0.0);
}

#[test]
fn test_variation_of_information() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vi = variation_of_information(&x, &y, None, true).unwrap();
    assert!(vi >= 0.0);
}

#[test]
fn test_dependence_matrix_pearson() {
    let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let dm = dependence_matrix(&data, DependenceMethod::Pearson).unwrap();
    assert_eq!(dm.shape(), &[2, 2]);
    // Diagonal should be 1
    assert!((dm[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((dm[[1, 1]] - 1.0).abs() < 1e-10);
}

#[test]
fn test_dependence_matrix_spearman() {
    let data = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0]
    ];
    let dm = dependence_matrix(&data, DependenceMethod::Spearman).unwrap();
    assert_eq!(dm.shape(), &[2, 2]);
}

#[test]
fn test_distance_matrix_angular() {
    let corr = array![[1.0, 0.8], [0.8, 1.0]];
    let dm = distance_matrix(&corr, DistanceMetric::Angular).unwrap();
    assert_eq!(dm.shape(), &[2, 2]);
    // Diagonal should be 0
    assert!(dm[[0, 0]].abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════
// Clustering
// ═══════════════════════════════════════════════════════════

#[test]
fn test_kmeans() {
    // Two clear clusters
    let data = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.1],
        [10.2, 10.0]
    ];
    let result = kmeans(&data, 2, 100, 3, 42).unwrap();
    assert_eq!(result.labels.len(), 6);
    // Points in the same cluster should have the same label
    assert_eq!(result.labels[0], result.labels[1]);
    assert_eq!(result.labels[0], result.labels[2]);
    assert_eq!(result.labels[3], result.labels[4]);
    assert_eq!(result.labels[3], result.labels[5]);
    assert_ne!(result.labels[0], result.labels[3]);
}

#[test]
fn test_silhouette_score() {
    let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]];
    let labels = vec![0, 0, 1, 1];
    let score = silhouette_score(&data, &labels);
    assert!(
        score > 0.5,
        "well-separated clusters should have high silhouette score"
    );
}

#[test]
fn test_cluster_kmeans_base() {
    // Generate a simple correlation matrix
    let corr = array![
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0]
    ];
    let result = cluster_kmeans_base(&corr, Some(3), Some(2), Some(5), Some(42)).unwrap();
    assert_eq!(result.labels.len(), 4);
}

#[test]
fn test_cluster_kmeans_top() {
    let corr = array![
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0]
    ];
    let result = cluster_kmeans_top(&corr, Some(3), Some(2), Some(5), Some(42)).unwrap();
    assert_eq!(result.labels.len(), 4);
}

// ═══════════════════════════════════════════════════════════
// Denoising (RMT)
// ═══════════════════════════════════════════════════════════

#[test]
fn test_cov_to_corr_and_back() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let (corr, std_devs) = cov_to_corr(&cov).unwrap();
    assert_eq!(corr.shape(), &[2, 2]);
    assert!((corr[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((corr[[1, 1]] - 1.0).abs() < 1e-10);
    assert!((std_devs[0] - 0.2).abs() < 1e-10);
    assert!((std_devs[1] - 0.3).abs() < 1e-10);

    let cov_back = corr_to_cov(&corr, &std_devs).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (cov_back[[i, j]] - cov[[i, j]]).abs() < 1e-10,
                "round-trip failed at [{i},{j}]"
            );
        }
    }
}

#[test]
fn test_marcenko_pastur_pdf() {
    let (x, pdf) = marcenko_pastur_pdf(1.0, 2.0, 100).unwrap();
    assert_eq!(x.len(), 100);
    assert_eq!(pdf.len(), 100);
    for &v in pdf.iter() {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_fit_kde() {
    let observations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let eval_points = Array1::linspace(0.0, 6.0, 50);
    let kde = fit_kde(&observations, 0.5, &eval_points);
    assert_eq!(kde.len(), 50);
    for &v in kde.iter() {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_denoise_corr() {
    let corr = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];
    let denoised = denoise_corr(&corr, 3.0, None, false, None).unwrap();
    assert_eq!(denoised.shape(), &[3, 3]);
    // Diagonal should still be 1
    for i in 0..3 {
        assert!((denoised[[i, i]] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_denoise_corr_with_shrinkage() {
    let corr = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]];
    let denoised = denoise_corr(&corr, 3.0, None, true, Some(0.5)).unwrap();
    assert_eq!(denoised.shape(), &[3, 3]);
}

#[test]
fn test_denoise_cov() {
    let cov = array![[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.16]];
    let denoised = denoise_cov(&cov, 3.0, None).unwrap();
    assert_eq!(denoised.shape(), &[3, 3]);
}

#[test]
fn test_detone_corr() {
    let corr = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]];
    let detoned = detone_corr(&corr, 1).unwrap();
    assert_eq!(detoned.shape(), &[3, 3]);
}

#[test]
fn test_optimal_portfolio() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let weights = optimal_portfolio(&cov, None).unwrap();
    assert_eq!(weights.len(), 2);
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_optimal_portfolio_with_mu() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let mu = Array1::from_vec(vec![0.05, 0.10]);
    let weights = optimal_portfolio(&cov, Some(&mu)).unwrap();
    assert_eq!(weights.len(), 2);
}

// ═══════════════════════════════════════════════════════════
// Entropy
// ═══════════════════════════════════════════════════════════

#[test]
fn test_shannon_entropy() {
    let probs = vec![0.5, 0.5];
    let h = shannon_entropy(&probs);
    assert!((h - 1.0).abs() < 1e-10); // log2(2) = 1 bit
}

#[test]
fn test_shannon_entropy_certain() {
    let probs = vec![1.0, 0.0];
    let h = shannon_entropy(&probs);
    assert!(h.abs() < 1e-10); // 0 entropy for certain outcome
}

#[test]
fn test_redundancy() {
    let probs = vec![0.5, 0.5];
    let r = redundancy(&probs);
    assert!(r.is_finite());
}

#[test]
fn test_plugin_entropy() {
    let sequence = vec![0, 1, 0, 1, 0, 1, 2, 2];
    let h = plugin_entropy(&sequence, 3);
    assert!(h >= 0.0);
}

#[test]
fn test_lempel_ziv_complexity() {
    let binary = vec![true, false, true, false, true, true, false, false, true];
    let c = lempel_ziv_complexity(&binary);
    assert!(c >= 1);
}

#[test]
fn test_normalized_lz_complexity() {
    let binary = vec![true, false, true, false, true, true, false, false, true];
    let nc = normalized_lz_complexity(&binary);
    assert!(nc > 0.0);
}

#[test]
fn test_kontoyiannis_entropy() {
    let sequence = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let h = kontoyiannis_entropy(&sequence, 5);
    assert!(h >= 0.0);
}

#[test]
fn test_gaussian_entropy() {
    let h = gaussian_entropy(1.0);
    assert!(h > 0.0);
}

#[test]
fn test_entropy_implied_vol() {
    let h = gaussian_entropy(1.0);
    let vol = entropy_implied_vol(h);
    assert!((vol - 1.0).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════
// Allocation
// ═══════════════════════════════════════════════════════════

#[test]
fn test_hrp_weights() {
    let returns = array![
        [0.01, 0.02, 0.005],
        [-0.005, 0.01, -0.01],
        [0.02, -0.01, 0.015],
        [0.005, 0.005, 0.01],
        [-0.01, 0.015, -0.005],
        [0.015, -0.005, 0.02],
        [-0.002, 0.008, 0.003],
        [0.01, -0.01, 0.01]
    ];
    let weights = hrp_weights(&returns).unwrap();
    assert_eq!(weights.len(), 3);
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    for &w in weights.iter() {
        assert!(w >= 0.0);
    }
}

#[test]
fn test_correlation_distance() {
    let corr = array![[1.0, 0.5], [0.5, 1.0]];
    let dist = correlation_distance(&corr);
    assert_eq!(dist.shape(), &[2, 2]);
    assert!(dist[[0, 0]].abs() < 1e-10);
    assert!(dist[[0, 1]] > 0.0);
}

#[test]
fn test_single_linkage_clustering() {
    let dist = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]];
    let linkage = single_linkage_clustering(&dist);
    assert_eq!(linkage.len(), 2); // n-1 merges
    for merge in &linkage {
        assert_eq!(merge.len(), 4); // [idx1, idx2, distance, count]
    }
}

#[test]
fn test_quasi_diag() {
    let linkage = vec![[0.0, 1.0, 0.5, 2.0], [2.0, 3.0, 1.0, 3.0]];
    let order = quasi_diag(&linkage, 3);
    assert_eq!(order.len(), 3);
    // Should be a permutation of [0, 1, 2]
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn test_recursive_bisection() {
    let cov = array![[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.16]];
    let sorted_indices = vec![0, 1, 2];
    let weights = recursive_bisection(&cov, &sorted_indices);
    assert_eq!(weights.len(), 3);
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_inverse_variance_weights() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let weights = inverse_variance_weights(&cov);
    assert_eq!(weights.len(), 2);
    // Lower variance should get higher weight
    assert!(weights[0] > weights[1]);
}

#[test]
fn test_cla_min_variance() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let weights = cla_min_variance(&cov).unwrap();
    assert_eq!(weights.len(), 2);
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_cla_max_sharpe() {
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let mu = Array1::from_vec(vec![0.05, 0.10]);
    let weights = cla_max_sharpe(&mu, &cov).unwrap();
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_compare_allocations() {
    let returns = array![
        [0.01, 0.02, 0.005],
        [-0.005, 0.01, -0.01],
        [0.02, -0.01, 0.015],
        [0.005, 0.005, 0.01],
        [-0.01, 0.015, -0.005],
        [0.015, -0.005, 0.02],
        [-0.002, 0.008, 0.003],
        [0.01, -0.01, 0.01],
        [0.005, 0.012, -0.003],
        [-0.008, 0.003, 0.007],
        [0.012, -0.002, 0.009],
        [0.003, 0.006, -0.004]
    ];
    let result = compare_allocations(&returns, 10, 42).unwrap();
    assert!(result.hrp_sharpe.is_finite());
    assert!(result.cla_sharpe.is_finite());
    assert!(result.ivp_sharpe.is_finite());
}
