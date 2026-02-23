use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlfinance_features::allocation::hrp::hrp::hrp_weights;
use ndarray::Array2;

fn generate_returns(n_obs: usize, n_assets: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_obs, n_assets), |(i, j)| {
        ((i * 7 + j * 13) as f64 * 0.7).sin() * 0.01
    })
}

fn bench_hrp_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrp_weights");
    for n_assets in [5, 10, 20, 50] {
        let returns = generate_returns(252, n_assets);
        group.bench_with_input(
            BenchmarkId::from_parameter(n_assets),
            &returns,
            |b, returns| {
                b.iter(|| hrp_weights(black_box(returns)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_hrp_weights);
criterion_main!(benches);
