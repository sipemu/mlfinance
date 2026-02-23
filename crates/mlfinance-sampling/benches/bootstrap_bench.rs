use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_sampling::bootstrap::sequential::seq_bootstrap;
use mlfinance_sampling::concurrency::indicator_matrix::get_indicator_matrix;

fn bench_get_indicator_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_indicator_matrix");
    for (rows, cols) in [(50, 10), (200, 50), (1000, 100)] {
        let events: Vec<(usize, usize)> = (0..cols)
            .map(|i| {
                let start = i * rows / cols;
                let end = (start + rows / 5).min(rows - 1);
                (start, end)
            })
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{cols}")),
            &(events.clone(), rows),
            |b, (events, num_bars)| {
                b.iter(|| get_indicator_matrix(black_box(events), black_box(*num_bars)));
            },
        );
    }
    group.finish();
}

fn bench_seq_bootstrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_bootstrap");
    for (rows, cols) in [(50, 10), (200, 50)] {
        let events: Vec<(usize, usize)> = (0..cols)
            .map(|i| {
                let start = i * rows / cols;
                let end = (start + rows / 5).min(rows - 1);
                (start, end)
            })
            .collect();
        let ind_matrix = get_indicator_matrix(&events, rows);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{cols}")),
            &ind_matrix,
            |b, matrix| {
                b.iter(|| seq_bootstrap(black_box(matrix), black_box(cols), black_box(42)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_get_indicator_matrix, bench_seq_bootstrap);
criterion_main!(benches);
