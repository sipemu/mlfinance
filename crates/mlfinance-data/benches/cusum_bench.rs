use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_data::sampling::cusum_filter::cusum_filter;

fn bench_cusum_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("cusum_filter");
    for n in [1_000, 10_000, 100_000] {
        let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin() * 0.02).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, values| {
            b.iter(|| cusum_filter(black_box(values), black_box(0.05)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cusum_filter);
criterion_main!(benches);
