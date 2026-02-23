use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Process a list of jobs in parallel with an optional progress callback.
///
/// Spawns a dedicated rayon thread pool and applies `func` to every job in
/// parallel. Results are returned in the same order as the input.
/// Equivalent to Snippets 20.9-20.10 (`processJobs`).
///
/// # Arguments
///
/// * `jobs` - Slice of work items to process.
/// * `func` - Function applied to each job to produce a result.
/// * `num_threads` - Number of threads in the pool (at least 1).
/// * `on_progress` - Optional callback invoked after each job completes.
///   Receives `(completed_count, total_count)`.
///
/// # Returns
///
/// A vector of results, one per job, preserving the input order.
pub fn process_jobs<T, R, F>(
    jobs: &[T],
    func: F,
    num_threads: usize,
    on_progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let total = jobs.len();
    let completed = Arc::new(AtomicUsize::new(0));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.max(1))
        .build()
        .unwrap();

    pool.install(|| {
        jobs.par_iter()
            .map(|job| {
                let result = func(job);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(cb) = on_progress {
                    cb(done, total);
                }
                result
            })
            .collect()
    })
}

/// Process jobs that return `Result` values, separating successes from errors.
///
/// Like [`process_jobs`], but `func` returns a `Result`. Successful results and
/// errors are collected into separate vectors.
///
/// # Arguments
///
/// * `jobs` - Slice of work items to process.
/// * `func` - Function applied to each job, returning `Ok(R)` or `Err(E)`.
/// * `num_threads` - Number of threads in the pool (at least 1).
///
/// # Returns
///
/// A tuple `(successes, errors)` where `errors` contains `(index, error)` pairs
/// indicating which job failed and the associated error.
pub fn process_jobs_result<T, R, E, F>(
    jobs: &[T],
    func: F,
    num_threads: usize,
) -> (Vec<R>, Vec<(usize, E)>)
where
    T: Sync,
    R: Send,
    E: Send,
    F: Fn(&T) -> Result<R, E> + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.max(1))
        .build()
        .unwrap();

    let results: Vec<(usize, Result<R, E>)> = pool.install(|| {
        jobs.par_iter()
            .enumerate()
            .map(|(i, job)| (i, func(job)))
            .collect()
    });

    let mut successes = Vec::new();
    let mut errors = Vec::new();
    for (i, result) in results {
        match result {
            Ok(v) => successes.push(v),
            Err(e) => errors.push((i, e)),
        }
    }
    (successes, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_jobs() {
        let jobs: Vec<i32> = (0..20).collect();
        let results = process_jobs(&jobs, |&x| x * x, 4, None);
        assert_eq!(results.len(), 20);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, (i as i32) * (i as i32));
        }
    }
}
