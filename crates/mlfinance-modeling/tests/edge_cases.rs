use mlfinance_modeling::cross_validation::purged_kfold::PurgedKFold;

#[test]
#[should_panic(expected = "n_splits must be at least 2")]
fn purged_kfold_n_splits_one() {
    PurgedKFold::new(1, 0.0);
}

#[test]
#[should_panic(expected = "embargo_pct must be in [0, 1]")]
fn purged_kfold_embargo_pct_negative() {
    PurgedKFold::new(3, -0.1);
}

#[test]
#[should_panic(expected = "embargo_pct must be in [0, 1]")]
fn purged_kfold_embargo_pct_above_one() {
    PurgedKFold::new(3, 1.5);
}

#[test]
fn purged_kfold_embargo_pct_one() {
    // Should succeed - extreme but valid
    let events: Vec<(usize, usize)> = (0..10).map(|i| (i, i)).collect();
    let kfold = PurgedKFold::new(2, 1.0);
    let folds = kfold.split(&events, 10);
    assert_eq!(folds.len(), 2);
}

#[test]
fn purged_kfold_n_splits_equals_n_samples() {
    let events: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
    let kfold = PurgedKFold::new(5, 0.0);
    let folds = kfold.split(&events, 5);
    assert_eq!(folds.len(), 5);
    // Each fold should have exactly 1 test sample
    for fold in &folds {
        assert_eq!(fold.test.len(), 1);
    }
}

#[test]
fn purged_kfold_overlapping_events_reduces_training() {
    // All events overlap: every sample spans the entire range
    let events: Vec<(usize, usize)> = (0..6).map(|_| (0, 100)).collect();
    let kfold = PurgedKFold::new(3, 0.0);
    let folds = kfold.split(&events, 6);

    // Because all events overlap, training sets should be empty after purging
    for fold in &folds {
        assert!(
            fold.train.is_empty(),
            "All-overlapping events should result in empty training set, got {:?}",
            fold.train
        );
    }
}

#[test]
fn purged_kfold_no_train_test_overlap_with_embargo() {
    let events: Vec<(usize, usize)> = (0..20).map(|i| (i, i + 2)).collect();
    let kfold = PurgedKFold::new(4, 0.15);
    let folds = kfold.split(&events, 20);

    for (fold_idx, fold) in folds.iter().enumerate() {
        for &test_idx in &fold.test {
            assert!(
                !fold.train.contains(&test_idx),
                "Fold {}: test sample {} in training",
                fold_idx,
                test_idx
            );
        }
    }
}
