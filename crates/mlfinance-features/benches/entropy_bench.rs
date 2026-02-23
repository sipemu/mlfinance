use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity;
use mlfinance_features::entropy::plugin::plugin_entropy;

fn bench_plugin_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("plugin_entropy");
    for n in [100, 1_000, 10_000] {
        let sequence: Vec<usize> = (0..n).map(|i| i % 10).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &sequence, |b, seq| {
            b.iter(|| plugin_entropy(black_box(seq), black_box(10)));
        });
    }
    group.finish();
}

fn bench_lempel_ziv(c: &mut Criterion) {
    let mut group = c.benchmark_group("lempel_ziv_complexity");
    for n in [100, 1_000, 10_000] {
        let binary: Vec<bool> = (0..n).map(|i| (i * 7 + 3) % 5 < 3).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &binary, |b, bin| {
            b.iter(|| lempel_ziv_complexity(black_box(bin)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_plugin_entropy, bench_lempel_ziv);
criterion_main!(benches);
