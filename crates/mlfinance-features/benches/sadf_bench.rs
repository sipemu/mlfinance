use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_features::structural_breaks::sadf::sadf;

fn bench_sadf(c: &mut Criterion) {
    let mut group = c.benchmark_group("sadf");
    group.sample_size(10);
    for n in [100, 200, 500] {
        let series: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.05).sin() * 10.0 + i as f64 * 0.01)
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &series, |b, series| {
            b.iter(|| sadf(black_box(series), black_box(20), black_box(5)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sadf);
criterion_main!(benches);
