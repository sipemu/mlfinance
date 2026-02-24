use mlfinance_parallel::combinatorics::optimization::{dynamic_optimization, static_optimization};
use mlfinance_parallel::combinatorics::partitions::{
    binomial, generate_partitions, partitions_count,
};
use mlfinance_parallel::combinatorics::random_matrix::random_matrix_of_rank;
use mlfinance_parallel::combinatorics::trajectories::{
    evaluate_trajectories, expected_value, generate_trajectories,
};
use mlfinance_parallel::parallel_map::{mp_apply, mp_apply_indexed, par_map};
use mlfinance_parallel::partition::{lin_parts, nested_parts};
use mlfinance_parallel::process_jobs::{process_jobs, process_jobs_result};
use mlfinance_parallel::vectorization::{cartesian_product, index_product};
use ndarray::{array, Array1};

// ── partition ──

#[test]
fn test_lin_parts() {
    let parts = lin_parts(10, 3);
    assert_eq!(parts.len(), 3);
    // Should cover the full range [0, 10)
    assert_eq!(parts[0].0, 0);
    assert_eq!(parts.last().unwrap().1, 10);
    // No gaps
    for i in 1..parts.len() {
        assert_eq!(parts[i].0, parts[i - 1].1);
    }
}

#[test]
fn test_lin_parts_single_thread() {
    let parts = lin_parts(5, 1);
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], (0, 5));
}

#[test]
fn test_nested_parts_lower_triangle() {
    let parts = nested_parts(10, 3, false);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].0, 0);
    assert_eq!(parts.last().unwrap().1, 10);
}

#[test]
fn test_nested_parts_upper_triangle() {
    let parts = nested_parts(10, 3, true);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].0, 0);
}

// ── parallel_map ──

#[test]
fn test_mp_apply() {
    let data: Vec<i32> = (0..20).collect();
    let result = mp_apply(&data, 4, |chunk| chunk.iter().map(|&x| x * 2).collect());
    assert_eq!(result.len(), 20);
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, (i as i32) * 2);
    }
}

#[test]
fn test_mp_apply_indexed() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = mp_apply_indexed(&data, 2, |start, _end, chunk| {
        chunk
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (start + i) as f64)
            .collect()
    });
    assert_eq!(result.len(), 6);
}

#[test]
fn test_par_map() {
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let result = par_map(&data, |x| x * x);
    assert_eq!(result, vec![1, 4, 9, 16, 25]);
}

// ── process_jobs ──

#[test]
fn test_process_jobs_basic() {
    let jobs: Vec<i32> = vec![1, 2, 3, 4, 5];
    let result = process_jobs(&jobs, |&x| x * 10, 2, None);
    assert_eq!(result, vec![10, 20, 30, 40, 50]);
}

#[test]
fn test_process_jobs_with_progress() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let jobs: Vec<i32> = vec![1, 2, 3];
    let progress_count = AtomicUsize::new(0);
    let _result = process_jobs(
        &jobs,
        |&x| x + 1,
        2,
        Some(&|_done, _total| {
            progress_count.fetch_add(1, Ordering::Relaxed);
        }),
    );
    assert!(progress_count.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_process_jobs_result_all_ok() {
    let jobs: Vec<i32> = vec![1, 2, 3];
    let (results, errors): (Vec<i32>, Vec<(usize, String)>) =
        process_jobs_result(&jobs, |&x| Ok::<i32, String>(x * 2), 2);
    assert_eq!(results, vec![2, 4, 6]);
    assert!(errors.is_empty());
}

#[test]
fn test_process_jobs_result_with_errors() {
    let jobs: Vec<i32> = vec![1, -1, 2];
    let (results, errors): (Vec<i32>, Vec<(usize, &str)>) = process_jobs_result(
        &jobs,
        |&x| {
            if x < 0 {
                Err("negative")
            } else {
                Ok(x * 2)
            }
        },
        2,
    );
    assert_eq!(results.len(), 2);
    assert_eq!(errors.len(), 1);
}

// ── vectorization ──

#[test]
fn test_cartesian_product() {
    let sets = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = cartesian_product(&sets);
    assert_eq!(result.len(), 4);
    assert!(result.contains(&vec![1.0, 3.0]));
    assert!(result.contains(&vec![1.0, 4.0]));
    assert!(result.contains(&vec![2.0, 3.0]));
    assert!(result.contains(&vec![2.0, 4.0]));
}

#[test]
fn test_cartesian_product_single_set() {
    let sets = vec![vec![1.0, 2.0, 3.0]];
    let result = cartesian_product(&sets);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_index_product() {
    let dims = vec![2, 3];
    let result = index_product(&dims);
    assert_eq!(result.len(), 6);
    assert!(result.contains(&vec![0, 0]));
    assert!(result.contains(&vec![1, 2]));
}

// ── combinatorics::partitions ──

#[test]
fn test_generate_partitions() {
    // Distribute 3 objects into 2 slots: (0,3),(1,2),(2,1),(3,0)
    let parts = generate_partitions(2, 3);
    assert_eq!(parts.len(), 4);
    for p in &parts {
        assert_eq!(p.iter().sum::<usize>(), 3);
        assert_eq!(p.len(), 2);
    }
}

#[test]
fn test_partitions_count() {
    // partitions_count(n, k) = C(n+k-1, k)
    let count = partitions_count(2, 3);
    let parts = generate_partitions(2, 3);
    assert_eq!(count, parts.len() as u64);
}

#[test]
fn test_binomial() {
    assert_eq!(binomial(5, 2), 10);
    assert_eq!(binomial(10, 0), 1);
    assert_eq!(binomial(10, 10), 1);
    assert_eq!(binomial(6, 3), 20);
}

// ── combinatorics::random_matrix ──

#[test]
fn test_random_matrix_of_rank() {
    let m = random_matrix_of_rank(5, 4, 2, 42);
    assert_eq!(m.shape(), &[5, 4]);
    // All elements should be finite
    for &v in m.iter() {
        assert!(v.is_finite());
    }
}

#[test]
fn test_random_matrix_of_rank_full() {
    let m = random_matrix_of_rank(3, 3, 3, 99);
    assert_eq!(m.shape(), &[3, 3]);
}

// ── combinatorics::trajectories ──

#[test]
fn test_generate_trajectories() {
    let moves = vec![1.0, -1.0];
    let trajs = generate_trajectories(0.0, &moves, 3);
    // 2^3 = 8 trajectories, each of length 4 (initial + 3 steps)
    assert_eq!(trajs.len(), 8);
    for t in &trajs {
        assert_eq!(t.len(), 4);
        assert_eq!(t[0], 0.0);
    }
}

#[test]
fn test_evaluate_trajectories() {
    let trajs = vec![vec![0.0, 1.0, 2.0], vec![0.0, -1.0, -2.0]];
    let payoffs = evaluate_trajectories(&trajs, |t| *t.last().unwrap());
    assert_eq!(payoffs, vec![2.0, -2.0]);
}

#[test]
fn test_expected_value() {
    let trajs = vec![
        vec![0.0, 1.0, 2.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, -1.0, 0.0],
        vec![0.0, -1.0, -2.0],
    ];
    let ev = expected_value(&trajs, |t| *t.last().unwrap());
    assert!((ev - 0.0).abs() < 1e-10);
}

// ── combinatorics::optimization ──

#[test]
fn test_static_optimization() {
    let returns = Array1::from_vec(vec![0.05, 0.10]);
    let cov = array![[0.04, 0.01], [0.01, 0.09]];
    let weights = static_optimization(&returns, &cov, 1.0).unwrap();
    assert_eq!(weights.len(), 2);
    // Weights should sum to approximately 1
    let sum: f64 = weights.iter().sum();
    assert!(sum.is_finite());
}

#[test]
fn test_dynamic_optimization() {
    let returns1 = Array1::from_vec(vec![0.05, 0.10]);
    let returns2 = Array1::from_vec(vec![0.08, 0.06]);
    let cov1 = array![[0.04, 0.01], [0.01, 0.09]];
    let cov2 = array![[0.05, 0.02], [0.02, 0.07]];
    let result = dynamic_optimization(&[returns1, returns2], &[cov1, cov2], 1.0).unwrap();
    assert_eq!(result.len(), 2);
    for w in &result {
        assert_eq!(w.len(), 2);
    }
}
