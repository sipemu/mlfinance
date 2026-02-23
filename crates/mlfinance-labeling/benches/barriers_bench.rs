use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_labeling::barriers::{find_first_touch, TripleBarrierConfig};
use mlfinance_labeling::events::get_events;
use mlfinance_labeling::volatility::daily_volatility;

fn generate_prices(n: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(n);
    let mut price = 100.0;
    for i in 0..n {
        price += (i as f64 * 0.1).sin() * 0.5;
        prices.push(price);
    }
    prices
}

fn bench_daily_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("daily_volatility");
    for n in [1_000, 10_000, 50_000] {
        let prices = generate_prices(n);
        let timestamps: Vec<_> = (0..n)
            .map(|i| {
                chrono::DateTime::from_timestamp(1_700_000_000 + (i as i64) * 86400, 0).unwrap()
            })
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(&prices, &timestamps),
            |b, &(prices, timestamps)| {
                b.iter(|| {
                    daily_volatility(black_box(prices), black_box(timestamps), black_box(20))
                });
            },
        );
    }
    group.finish();
}

fn bench_find_first_touch(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_first_touch");
    for n in [1_000, 10_000, 50_000] {
        let prices = generate_prices(n);
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0),
            max_holding_period: Some(100),
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &prices, |b, prices| {
            b.iter(|| {
                find_first_touch(
                    black_box(prices),
                    black_box(0),
                    black_box(&config),
                    black_box(0.02),
                )
            });
        });
    }
    group.finish();
}

fn bench_get_events(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_events");
    for n in [1_000, 10_000] {
        let prices = generate_prices(n);
        let entry_indices: Vec<usize> = (0..n).step_by(50).collect();
        let config = TripleBarrierConfig {
            upper_barrier: Some(2.0),
            lower_barrier: Some(2.0),
            max_holding_period: Some(100),
        };
        let daily_vols = vec![0.02; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                get_events(
                    black_box(&prices),
                    black_box(&entry_indices),
                    black_box(&config),
                    black_box(&daily_vols),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_daily_volatility,
    bench_find_first_touch,
    bench_get_events
);
criterion_main!(benches);
