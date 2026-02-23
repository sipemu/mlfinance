use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_backtesting::bet_sizing::probability_to_size::{power_bet_size, sigmoid_bet_size};

fn bench_sigmoid_bet_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid_bet_size");
    for n in [1_000, 10_000] {
        let probs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &probs, |b, probs| {
            b.iter(|| {
                for &p in probs {
                    black_box(sigmoid_bet_size(black_box(p), black_box(2)));
                }
            });
        });
    }
    group.finish();
}

fn bench_power_bet_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_bet_size");
    for n in [1_000, 10_000] {
        let probs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &probs, |b, probs| {
            b.iter(|| {
                for &p in probs {
                    black_box(power_bet_size(black_box(p), black_box(2), black_box(2.0)));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sigmoid_bet_size, bench_power_bet_size);
criterion_main!(benches);
