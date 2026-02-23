use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_backtesting::statistics::drawdown::compute_drawdowns;
use mlfinance_backtesting::statistics::sharpe::sharpe_ratio;

fn bench_sharpe_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharpe_ratio");
    for n in [100, 1_000, 10_000] {
        let returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &returns, |b, returns| {
            b.iter(|| sharpe_ratio(black_box(returns), black_box(0.0), black_box(252.0)));
        });
    }
    group.finish();
}

fn bench_drawdowns(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_drawdowns");
    for n in [100, 1_000, 10_000] {
        let returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &returns, |b, returns| {
            b.iter(|| compute_drawdowns(black_box(returns)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sharpe_ratio, bench_drawdowns);
criterion_main!(benches);
