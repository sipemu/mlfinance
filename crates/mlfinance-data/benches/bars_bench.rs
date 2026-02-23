use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_core::traits::BarAggregator;
use mlfinance_core::types::TickData;
use mlfinance_data::bars::dollar_bars::DollarBarAggregator;
use mlfinance_data::bars::tick_bars::TickBarAggregator;
use mlfinance_data::bars::volume_bars::VolumeBarAggregator;

fn generate_ticks(n: usize) -> Vec<TickData> {
    (0..n)
        .map(|i| {
            let price = 100.0 + (i as f64 * 0.01).sin() * 5.0;
            let volume = 100.0 + (i as f64 * 0.03).cos() * 50.0;
            TickData {
                timestamp: chrono::DateTime::from_timestamp(1_700_000_000 + i as i64, 0).unwrap(),
                price,
                volume,
            }
        })
        .collect()
}

fn bench_tick_bars(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_bars");
    for n in [1_000, 10_000, 100_000] {
        let ticks = generate_ticks(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ticks, |b, ticks| {
            b.iter(|| {
                let mut agg = TickBarAggregator::new(50);
                agg.process_ticks(black_box(ticks))
            });
        });
    }
    group.finish();
}

fn bench_volume_bars(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_bars");
    for n in [1_000, 10_000, 100_000] {
        let ticks = generate_ticks(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ticks, |b, ticks| {
            b.iter(|| {
                let mut agg = VolumeBarAggregator::new(5000.0);
                agg.process_ticks(black_box(ticks))
            });
        });
    }
    group.finish();
}

fn bench_dollar_bars(c: &mut Criterion) {
    let mut group = c.benchmark_group("dollar_bars");
    for n in [1_000, 10_000, 100_000] {
        let ticks = generate_ticks(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ticks, |b, ticks| {
            b.iter(|| {
                let mut agg = DollarBarAggregator::new(500_000.0);
                agg.process_ticks(black_box(ticks))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tick_bars,
    bench_volume_bars,
    bench_dollar_bars
);
criterion_main!(benches);
