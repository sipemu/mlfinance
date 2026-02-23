use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_sampling::fracdiff::ffd::frac_diff_ffd;
use mlfinance_sampling::fracdiff::weights::get_weights_ffd;

fn bench_get_weights_ffd(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_weights_ffd");
    for threshold in [1e-2, 1e-4, 1e-6] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("t={threshold}")),
            &threshold,
            |b, &t| {
                b.iter(|| get_weights_ffd(black_box(0.5), black_box(t)));
            },
        );
    }
    group.finish();
}

fn bench_frac_diff_ffd(c: &mut Criterion) {
    let mut group = c.benchmark_group("frac_diff_ffd");
    for n in [100, 1_000, 10_000] {
        let series: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.01).sin() * 5.0)
            .collect();
        for threshold in [1e-2, 1e-4, 1e-6] {
            group.bench_with_input(
                BenchmarkId::new(format!("n={n}"), format!("t={threshold}")),
                &(&series, threshold),
                |b, &(s, t)| {
                    b.iter(|| frac_diff_ffd(black_box(s), black_box(0.5), black_box(t)));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_get_weights_ffd, bench_frac_diff_ffd);
criterion_main!(benches);
