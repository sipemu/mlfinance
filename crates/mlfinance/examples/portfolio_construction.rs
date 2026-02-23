//! Portfolio construction: Correlated Returns → HRP vs Equal-Weight → Sharpe Comparison
//!
//! Demonstrates Chapter 16 of "Advances in Financial Machine Learning":
//! - Hierarchical Risk Parity (HRP) portfolio allocation
//! - Comparison with equal-weight benchmark
//!
//! Run with: `cargo run -p mlfinance --example portfolio_construction`

use mlfinance::backtesting::statistics::sharpe::sharpe_ratio;
use mlfinance::core::stats::correlation_matrix;
use mlfinance::features::allocation::hrp::hrp::hrp_weights;
use ndarray::Array2;

fn main() {
    println!("=== Portfolio Construction Example ===\n");

    let n_obs = 500;
    let n_assets = 8;

    // Generate correlated returns with cluster structure
    let returns = generate_clustered_returns(n_obs, n_assets);
    println!("Generated {} observations for {} assets", n_obs, n_assets);

    // Show correlation matrix
    let corr = correlation_matrix(&returns).unwrap();
    println!("\nCorrelation matrix (first 4x4):");
    for i in 0..4.min(n_assets) {
        print!("  ");
        for j in 0..4.min(n_assets) {
            print!("{:6.3} ", corr[[i, j]]);
        }
        println!();
    }

    // HRP allocation
    let hrp_w = hrp_weights(&returns).unwrap();
    println!("\nHRP weights:");
    for (i, &w) in hrp_w.iter().enumerate() {
        println!("  Asset {}: {:.4}", i, w);
    }
    let hrp_sum: f64 = hrp_w.iter().sum();
    println!("  Sum: {:.6}", hrp_sum);

    // Equal-weight allocation
    let eq_w = 1.0 / n_assets as f64;
    println!("\nEqual weight: {:.4} per asset", eq_w);

    // Compute portfolio returns
    let hrp_returns: Vec<f64> = (0..n_obs)
        .map(|t| {
            (0..n_assets)
                .map(|j| returns[[t, j]] * hrp_w[j])
                .sum::<f64>()
        })
        .collect();

    let eq_returns: Vec<f64> = (0..n_obs)
        .map(|t| (0..n_assets).map(|j| returns[[t, j]] * eq_w).sum::<f64>())
        .collect();

    // Compute Sharpe ratios
    let hrp_sr = sharpe_ratio(&hrp_returns, 0.0, 252.0);
    let eq_sr = sharpe_ratio(&eq_returns, 0.0, 252.0);

    println!("\nPerformance comparison:");
    println!(
        "  HRP portfolio:   Sharpe = {:.4}, Vol = {:.6}",
        hrp_sr,
        portfolio_vol(&hrp_returns)
    );
    println!(
        "  Equal-weight:    Sharpe = {:.4}, Vol = {:.6}",
        eq_sr,
        portfolio_vol(&eq_returns)
    );

    if hrp_sr > eq_sr {
        println!(
            "\n  HRP outperforms equal-weight by {:.4} SR units",
            hrp_sr - eq_sr
        );
    } else {
        println!(
            "\n  Equal-weight outperforms HRP by {:.4} SR units",
            eq_sr - hrp_sr
        );
    }

    println!("\nPortfolio construction complete.");
}

fn generate_clustered_returns(n_obs: usize, n_assets: usize) -> Array2<f64> {
    // Create returns with two clusters:
    // Cluster 1 (assets 0-3): driven by factor 1
    // Cluster 2 (assets 4-7): driven by factor 2
    Array2::from_shape_fn((n_obs, n_assets), |(i, j)| {
        let factor1 = ((i * 7 + 1) as f64 * 0.3).sin() * 0.01;
        let factor2 = ((i * 11 + 3) as f64 * 0.5).sin() * 0.01;
        let idio = ((i * 13 + j * 31 + 1) as f64 * 0.7).sin() * 0.005;

        if j < n_assets / 2 {
            factor1 * 0.7 + idio
        } else {
            factor2 * 0.7 + idio
        }
    })
}

fn portfolio_vol(returns: &[f64]) -> f64 {
    let n = returns.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}
