use mlfinance_core::traits::Classifier;
use mlfinance_modeling::ensemble::bagging::bagging_accuracy;
use mlfinance_modeling::ensemble::boosting::adaboost_update;
use mlfinance_modeling::ensemble::random_forest::{
    rf_setup_1, rf_setup_2, rf_setup_3, MaxFeatures,
};
use mlfinance_modeling::feature_importance::kendall_tau::weighted_kendall_tau;
use mlfinance_modeling::feature_importance::mda::mean_decrease_accuracy;
use mlfinance_modeling::feature_importance::mdi::mean_decrease_impurity;
use mlfinance_modeling::feature_importance::orthogonal::orthogonal_features;
use mlfinance_modeling::feature_importance::sfi::single_feature_importance;
use mlfinance_modeling::feature_importance::synthetic_data::make_classification;
use mlfinance_modeling::hyperparams::grid_search::{grid_search, ParamGrid};
use mlfinance_modeling::hyperparams::log_uniform::log_uniform_sample;
use mlfinance_modeling::hyperparams::random_search::random_search;
use ndarray::{Array1, Array2};

/// Simple mock classifier for testing.
struct MockClassifier {
    coefs: Option<Array1<f64>>,
}

impl MockClassifier {
    fn new() -> Self {
        Self { coefs: None }
    }
}

impl Classifier for MockClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, _sample_weight: Option<&Array1<f64>>) {
        // Simple: store mean of each feature for positive vs negative class
        let n_features = x.ncols();
        let mut c = Array1::zeros(n_features);
        let mut count = 0.0;
        for (i, &label) in y.iter().enumerate() {
            if label > 0.0 {
                for j in 0..n_features {
                    c[j] += x[[i, j]];
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            c /= count;
        }
        self.coefs = Some(c);
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let coefs = self.coefs.as_ref().unwrap();
        let mut preds = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            let score: f64 = x.row(i).iter().zip(coefs.iter()).map(|(a, b)| a * b).sum();
            preds[i] = if score > 0.0 { 1.0 } else { -1.0 };
        }
        preds
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut proba = Array2::zeros((x.nrows(), 2));
        let preds = self.predict(x);
        for i in 0..x.nrows() {
            if preds[i] > 0.0 {
                proba[[i, 0]] = 0.3;
                proba[[i, 1]] = 0.7;
            } else {
                proba[[i, 0]] = 0.7;
                proba[[i, 1]] = 0.3;
            }
        }
        proba
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        self.coefs.as_ref().map(|c| {
            let sum: f64 = c.iter().map(|v| v.abs()).sum();
            if sum > 0.0 {
                c.mapv(|v| v.abs() / sum)
            } else {
                Array1::ones(c.len()) / c.len() as f64
            }
        })
    }
}

// ── adaboost_update ──

#[test]
fn test_adaboost_update_perfect() {
    let predictions = vec![1.0, -1.0, 1.0];
    let labels = vec![1.0, -1.0, 1.0]; // Perfect match
    let weights = vec![1.0 / 3.0; 3];
    let (new_weights, alpha) = adaboost_update(&predictions, &labels, &weights);
    assert_eq!(new_weights.len(), 3);
    assert!(alpha > 0.0); // Good classifier gets positive alpha
    let sum: f64 = new_weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10); // Weights normalized
}

#[test]
fn test_adaboost_update_random() {
    let predictions = vec![1.0, -1.0, 1.0, -1.0];
    let labels = vec![1.0, 1.0, -1.0, -1.0]; // 50% accuracy
    let weights = vec![0.25; 4];
    let (new_weights, _alpha) = adaboost_update(&predictions, &labels, &weights);
    assert_eq!(new_weights.len(), 4);
    let sum: f64 = new_weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

// ── bagging_accuracy ──

#[test]
fn test_bagging_accuracy() {
    // Perfect base classifier
    let acc = bagging_accuracy(100, 1.0);
    assert!((acc - 1.0).abs() < 1e-10);

    // Random base classifier
    let acc = bagging_accuracy(100, 0.5);
    assert!((acc - 0.5).abs() < 0.1);
}

#[test]
fn test_bagging_accuracy_improves_with_n() {
    let acc_10 = bagging_accuracy(10, 0.6);
    let acc_100 = bagging_accuracy(100, 0.6);
    // More base classifiers should not decrease accuracy
    assert!(acc_100 >= acc_10 - 1e-10);
}

// ── rf_setup configs ──

#[test]
fn test_rf_setup_1() {
    let config = rf_setup_1(10);
    assert_eq!(config.n_estimators, 1000);
    assert!(matches!(config.max_features, MaxFeatures::Sqrt));
}

#[test]
fn test_rf_setup_2() {
    let config = rf_setup_2(10);
    assert_eq!(config.n_estimators, 5000);
    assert!(matches!(config.max_features, MaxFeatures::Log2));
    assert!(config.min_weight_fraction_leaf > 0.0);
}

#[test]
fn test_rf_setup_3() {
    let config = rf_setup_3(10);
    assert_eq!(config.n_estimators, 1000);
    assert!(matches!(config.max_features, MaxFeatures::Fixed(10)));
}

// ── feature importance ──

#[test]
fn test_mean_decrease_impurity() {
    // 3 trees, 4 features each
    let importances = vec![
        vec![0.4, 0.3, 0.2, 0.1],
        vec![0.35, 0.25, 0.25, 0.15],
        vec![0.5, 0.2, 0.2, 0.1],
    ];
    let mdi = mean_decrease_impurity(&importances);
    assert_eq!(mdi.len(), 4);
    // Feature 0 should be most important
    assert!(mdi[0] > mdi[3]);
}

#[test]
fn test_mean_decrease_accuracy() {
    let data = make_classification(50, 2, 1, 1, 42);
    let mut clf = MockClassifier::new();
    clf.fit(&data.x, &data.y, None); // Must be pre-fitted
    let scoring = |y_true: &Array1<f64>, y_pred: &Array1<f64>| -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-10)
            .count() as f64
            / y_true.len() as f64
    };
    let mda = mean_decrease_accuracy(&clf, &data.x, &data.y, &scoring, 42);
    assert_eq!(mda.len(), data.x.ncols());
}

#[test]
fn test_single_feature_importance() {
    let data = make_classification(60, 2, 0, 1, 42);
    let mut clf = MockClassifier::new();
    let events: Vec<(usize, usize)> = (0..60).map(|i| (i, i)).collect();
    let scoring = |y_true: &Array1<f64>, y_pred: &Array1<f64>| -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-10)
            .count() as f64
            / y_true.len() as f64
    };
    let sfi = single_feature_importance(&mut clf, &data.x, &data.y, &events, 3, &scoring);
    assert_eq!(sfi.len(), data.x.ncols());
}

// ── weighted_kendall_tau ──

#[test]
fn test_weighted_kendall_tau_perfect() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tau = weighted_kendall_tau(&x, &y, None);
    assert!((tau - 1.0).abs() < 1e-10);
}

#[test]
fn test_weighted_kendall_tau_inverse() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let tau = weighted_kendall_tau(&x, &y, None);
    assert!((tau - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_weighted_kendall_tau_with_weights() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 3.0, 2.0, 4.0];
    let weights = vec![1.0, 2.0, 2.0, 1.0];
    let tau = weighted_kendall_tau(&x, &y, Some(&weights));
    assert!(tau > -1.0 && tau < 1.0);
}

// ── make_classification ──

#[test]
fn test_make_classification() {
    let data = make_classification(100, 3, 2, 1, 42);
    assert_eq!(data.x.nrows(), 100);
    assert_eq!(data.x.ncols(), 6); // 3 + 2 + 1
    assert_eq!(data.y.len(), 100);
    assert_eq!(data.n_informative, 3);
    assert_eq!(data.n_redundant, 2);
    assert_eq!(data.n_noise, 1);
    // Labels should be -1 or 1
    for &l in data.y.iter() {
        assert!(l == -1.0 || l == 1.0);
    }
}

// ── orthogonal_features ──

#[test]
fn test_orthogonal_features() {
    let data = make_classification(50, 3, 0, 0, 42);
    let (x_pca, explained) = orthogonal_features(&data.x, 2).unwrap();
    assert_eq!(x_pca.nrows(), 50);
    assert_eq!(x_pca.ncols(), 2);
    assert_eq!(explained.len(), 2);
    // Explained variance should be positive
    for &e in explained.iter() {
        assert!(e >= 0.0);
    }
}

// ── grid_search ──

#[test]
fn test_grid_search() {
    let grids = vec![
        ParamGrid {
            name: "alpha".to_string(),
            values: vec![0.1, 0.5, 1.0],
        },
        ParamGrid {
            name: "beta".to_string(),
            values: vec![0.01, 0.1],
        },
    ];
    let result = grid_search(&grids, &|params| {
        // Score: prefer alpha close to 0.5
        let alpha = params.iter().find(|(n, _)| n == "alpha").unwrap().1;
        -(alpha - 0.5).powi(2)
    });
    assert_eq!(result.all_scores.len(), 6); // 3 * 2
    let best_alpha = result
        .best_params
        .iter()
        .find(|(n, _)| n == "alpha")
        .unwrap()
        .1;
    assert!((best_alpha - 0.5).abs() < 1e-10);
}

// ── random_search ──

#[test]
fn test_random_search() {
    let distributions = vec![
        ("alpha".to_string(), 0.0, 1.0),
        ("beta".to_string(), 0.001, 0.1),
    ];
    let result = random_search(&distributions, 20, &|_params| 0.5, 42);
    assert_eq!(result.all_scores.len(), 20);
    assert!((result.best_score - 0.5).abs() < 1e-10);
}

// ── log_uniform_sample ──

#[test]
fn test_log_uniform_sample() {
    let samples = log_uniform_sample(0.001, 1000.0, 100, 42);
    assert_eq!(samples.len(), 100);
    for &s in &samples {
        assert!(s >= 0.001);
        assert!(s <= 1000.0);
    }
}

#[test]
fn test_log_uniform_sample_deterministic() {
    let s1 = log_uniform_sample(0.01, 10.0, 50, 42);
    let s2 = log_uniform_sample(0.01, 10.0, 50, 42);
    assert_eq!(s1, s2);
}
