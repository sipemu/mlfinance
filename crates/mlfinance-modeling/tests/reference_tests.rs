use mlfinance_modeling::cross_validation::purged_kfold::PurgedKFold;
use mlfinance_modeling::hyperparams::scoring::{accuracy_score, f1_score, neg_log_loss};
use serde_json::Value;
use std::fs;

fn parse_f64_array(val: &Value) -> Vec<f64> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn load_fixture(name: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(fixture_path(name)).unwrap()).unwrap()
}

fn fixture_path(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../tests/fixtures/{}", manifest, name)
}

#[test]
fn test_purged_kfold_all_samples_in_exactly_one_test_set() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("purged_kfold_splits.json")).unwrap(),
    )
    .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let events: Vec<(usize, usize)> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let arr = v.as_array().unwrap();
                (
                    arr[0].as_u64().unwrap() as usize,
                    arr[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let n_samples = case["n_samples"].as_u64().unwrap() as usize;
        let n_splits = case["n_splits"].as_u64().unwrap() as usize;
        let embargo_pct = case["embargo_pct"].as_f64().unwrap();

        let kfold = PurgedKFold::new(n_splits, embargo_pct);
        let folds = kfold.split(&events, n_samples);

        assert_eq!(folds.len(), n_splits);

        // Every sample must appear in exactly one test set
        let mut test_counts = vec![0usize; n_samples];
        for fold in &folds {
            for &idx in &fold.test {
                test_counts[idx] += 1;
            }
        }
        for (i, &count) in test_counts.iter().enumerate() {
            assert_eq!(
                count, 1,
                "Sample {} appears in {} test sets (n_splits={}, embargo={})",
                i, count, n_splits, embargo_pct
            );
        }
    }
}

#[test]
fn test_purged_kfold_no_train_test_overlap() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("purged_kfold_splits.json")).unwrap(),
    )
    .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let events: Vec<(usize, usize)> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let arr = v.as_array().unwrap();
                (
                    arr[0].as_u64().unwrap() as usize,
                    arr[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let n_samples = case["n_samples"].as_u64().unwrap() as usize;
        let n_splits = case["n_splits"].as_u64().unwrap() as usize;
        let embargo_pct = case["embargo_pct"].as_f64().unwrap();

        let kfold = PurgedKFold::new(n_splits, embargo_pct);
        let folds = kfold.split(&events, n_samples);

        for (fold_idx, fold) in folds.iter().enumerate() {
            for &test_idx in &fold.test {
                assert!(
                    !fold.train.contains(&test_idx),
                    "Fold {}: test sample {} found in training (n_splits={}, embargo={})",
                    fold_idx,
                    test_idx,
                    n_splits,
                    embargo_pct
                );
            }
        }
    }
}

#[test]
fn test_purged_kfold_fold_count() {
    let data: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path("purged_kfold_splits.json")).unwrap(),
    )
    .unwrap();

    for case in data["cases"].as_array().unwrap() {
        let events: Vec<(usize, usize)> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let arr = v.as_array().unwrap();
                (
                    arr[0].as_u64().unwrap() as usize,
                    arr[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let n_samples = case["n_samples"].as_u64().unwrap() as usize;
        let n_splits = case["n_splits"].as_u64().unwrap() as usize;
        let embargo_pct = case["embargo_pct"].as_f64().unwrap();

        let kfold = PurgedKFold::new(n_splits, embargo_pct);
        let folds = kfold.split(&events, n_samples);

        assert_eq!(
            folds.len(),
            n_splits,
            "Fold count mismatch for n_splits={}, embargo={}",
            n_splits,
            embargo_pct
        );
    }
}

// =====================================================================
// P1 — Scoring reference tests
// =====================================================================

#[test]
fn test_f1_score_match_python() {
    let data = load_fixture("scoring.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "f1_score" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let y_true = parse_f64_array(&case["y_true"]);
        let y_pred = parse_f64_array(&case["y_pred"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = f1_score(&y_true, &y_pred);
        assert!(
            (actual - expected).abs() < 1e-10,
            "F1 score mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_accuracy_score_match_python() {
    let data = load_fixture("scoring.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "accuracy_score" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let y_true = parse_f64_array(&case["y_true"]);
        let y_pred = parse_f64_array(&case["y_pred"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = accuracy_score(&y_true, &y_pred);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Accuracy score mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}

#[test]
fn test_neg_log_loss_match_python() {
    let data = load_fixture("scoring.json");

    for case in data["cases"].as_array().unwrap() {
        if case["type"].as_str().unwrap() != "neg_log_loss" {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let y_true = parse_f64_array(&case["y_true"]);
        let y_proba = parse_f64_array(&case["y_proba"]);
        let expected = case["result"].as_f64().unwrap();

        let actual = neg_log_loss(&y_true, &y_proba);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Neg log loss mismatch for '{}': got={}, expected={}",
            label,
            actual,
            expected
        );
    }
}
