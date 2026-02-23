use rayon::prelude::*;

use crate::partition::lin_parts;

/// Apply a function to contiguous partitions of a slice in parallel.
///
/// Splits `data` into `num_threads` partitions using [`lin_parts`],
/// applies `func` to each partition, and flattens the results.
/// Equivalent to Snippet 20.7 (`mpPandasObj`).
///
/// # Arguments
///
/// * `data` - Slice of elements to process.
/// * `num_threads` - Number of partitions and parallel workers.
/// * `func` - Function that processes a sub-slice and returns a vector of results.
///
/// # Returns
///
/// A flattened vector of all results across partitions, in partition order.
pub fn mp_apply<T, R, F>(data: &[T], num_threads: usize, func: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&[T]) -> Vec<R> + Sync,
{
    if data.is_empty() {
        return vec![];
    }
    let parts = lin_parts(data.len(), num_threads.max(1));
    let results: Vec<Vec<R>> = parts
        .par_iter()
        .map(|&(start, end)| func(&data[start..end]))
        .collect();
    results.into_iter().flatten().collect()
}

/// Apply a function to partitions in parallel, providing each partition's index and offset.
///
/// Like [`mp_apply`], but `func` additionally receives the partition index and
/// the start offset of the partition within the original slice.
///
/// # Arguments
///
/// * `data` - Slice of elements to process.
/// * `num_threads` - Number of partitions and parallel workers.
/// * `func` - Function `(partition_index, start_offset, sub_slice) -> Vec<R>`.
///
/// # Returns
///
/// A flattened vector of all results across partitions, in partition order.
pub fn mp_apply_indexed<T, R, F>(data: &[T], num_threads: usize, func: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, usize, &[T]) -> Vec<R> + Sync,
{
    if data.is_empty() {
        return vec![];
    }
    let parts = lin_parts(data.len(), num_threads.max(1));
    let results: Vec<Vec<R>> = parts
        .par_iter()
        .enumerate()
        .map(|(idx, &(start, end))| func(idx, start, &data[start..end]))
        .collect();
    results.into_iter().flatten().collect()
}

/// Apply a function to each element in parallel using rayon's `par_iter`.
///
/// # Arguments
///
/// * `data` - Slice of elements to process.
/// * `func` - Function applied to each element.
///
/// # Returns
///
/// A vector of results, one per element, preserving the input order.
pub fn par_map<T, R, F>(data: &[T], func: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    data.par_iter().map(func).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp_apply() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = mp_apply(&data, 4, |chunk| vec![chunk.iter().sum::<f64>()]);
        let total: f64 = result.iter().sum();
        assert!((total - 4950.0).abs() < 1e-10);
    }

    #[test]
    fn test_par_map() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = par_map(&data, |x| x * 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }
}
