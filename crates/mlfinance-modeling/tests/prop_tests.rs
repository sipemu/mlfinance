use mlfinance_modeling::cross_validation::purged_kfold::PurgedKFold;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn all_samples_in_exactly_one_test_set(
        n_samples in 6usize..30,
        n_splits in 2usize..6
    ) {
        let n_splits = n_splits.min(n_samples);
        let events: Vec<(usize, usize)> = (0..n_samples).map(|i| (i, i)).collect();
        let kfold = PurgedKFold::new(n_splits, 0.0);
        let folds = kfold.split(&events, n_samples);

        let mut test_counts = vec![0usize; n_samples];
        for fold in &folds {
            for &idx in &fold.test {
                test_counts[idx] += 1;
            }
        }
        for (i, &count) in test_counts.iter().enumerate() {
            prop_assert_eq!(count, 1, "Sample {} in {} test sets", i, count);
        }
    }

    #[test]
    fn no_train_test_overlap(
        n_samples in 6usize..30,
        n_splits in 2usize..6,
        embargo_pct_int in 0u32..20
    ) {
        let n_splits = n_splits.min(n_samples);
        let embargo_pct = embargo_pct_int as f64 / 100.0;
        let events: Vec<(usize, usize)> = (0..n_samples).map(|i| (i, i)).collect();
        let kfold = PurgedKFold::new(n_splits, embargo_pct);
        let folds = kfold.split(&events, n_samples);

        for fold in &folds {
            for &test_idx in &fold.test {
                prop_assert!(
                    !fold.train.contains(&test_idx),
                    "Test sample {} in training set",
                    test_idx
                );
            }
        }
    }

    #[test]
    fn fold_count_equals_n_splits(
        n_samples in 6usize..30,
        n_splits in 2usize..6
    ) {
        let n_splits = n_splits.min(n_samples);
        let events: Vec<(usize, usize)> = (0..n_samples).map(|i| (i, i)).collect();
        let kfold = PurgedKFold::new(n_splits, 0.0);
        let folds = kfold.split(&events, n_samples);
        prop_assert_eq!(folds.len(), n_splits);
    }
}
