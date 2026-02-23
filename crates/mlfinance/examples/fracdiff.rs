//! Fractional Differentiation & Stationarity
//!
//! Demonstrates Chapter 5 of "Advances in Financial Machine Learning":
//! - Fixed-width window fractional differentiation (FFD)
//! - Expanding-window fractional differentiation
//! - Finding the minimum d for stationarity
//! - Inspecting FFD weight vectors
//!
//! Run with: `cargo run -p mlfinance --example fracdiff`

use mlfinance::sampling::fracdiff::{
    expanding::frac_diff_expanding, ffd::frac_diff_ffd, min_d::find_min_d, weights::get_weights_ffd,
};

fn main() {
    println!("=== Fractional Differentiation Example ===\n");

    // Step 1: Generate a trending (non-stationary) price series
    let prices = generate_trending_series(500);
    println!("Generated {} prices (trending series)", prices.len());
    println!(
        "  First: {:.2}, Last: {:.2}, Range: {:.2}",
        prices[0],
        prices[prices.len() - 1],
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - prices.iter().cloned().fold(f64::INFINITY, f64::min)
    );

    // Step 2: Inspect FFD weights for various d values
    println!("\n--- FFD Weights ---");
    for &d in &[0.3, 0.5, 0.7, 1.0] {
        let w = get_weights_ffd(d, 1e-4);
        println!(
            "  d={:.1}: {} weights, first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            d,
            w.len(),
            w[0],
            w.get(1).unwrap_or(&0.0),
            w.get(2).unwrap_or(&0.0),
            w.get(3).unwrap_or(&0.0),
            w.get(4).unwrap_or(&0.0),
        );
    }

    // Step 3: Find minimum d for stationarity
    println!("\n--- Finding Minimum d ---");
    let min_d = find_min_d(&prices, 1.0, 0.1, 1e-4);
    println!("  Minimum d for stationarity: {:.1}", min_d);

    // Step 4: Apply FFD with found d
    println!("\n--- FFD Transform (d={:.1}) ---", min_d);
    let ffd = frac_diff_ffd(&prices, min_d, 1e-4);
    let valid: Vec<f64> = ffd.iter().copied().filter(|x| !x.is_nan()).collect();
    println!(
        "  Output: {} values ({} valid, {} NaN lead-in)",
        ffd.len(),
        valid.len(),
        ffd.len() - valid.len()
    );
    if valid.len() >= 3 {
        println!(
            "  First 3 valid: [{:.4}, {:.4}, {:.4}]",
            valid[0], valid[1], valid[2]
        );
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        let var = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
        println!("  Mean: {:.4}, Std: {:.4}", mean, var.sqrt());
    }

    // Step 5: Compare with expanding-window method
    println!("\n--- Expanding-Window Transform (d={:.1}) ---", min_d);
    let expanding = frac_diff_expanding(&prices, min_d, 1e-4);
    let valid_exp: Vec<f64> = expanding.iter().copied().filter(|x| !x.is_nan()).collect();
    println!(
        "  Output: {} values ({} valid)",
        expanding.len(),
        valid_exp.len()
    );
    if valid_exp.len() >= 3 {
        let mean = valid_exp.iter().sum::<f64>() / valid_exp.len() as f64;
        let var =
            valid_exp.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid_exp.len() as f64;
        println!("  Mean: {:.4}, Std: {:.4}", mean, var.sqrt());
    }

    // Step 6: Compare d=0 (original) vs d=1 (full diff) vs optimal d
    println!("\n--- Differentiation Comparison ---");
    let full_diff = frac_diff_ffd(&prices, 1.0, 1e-4);
    let full_valid: Vec<f64> = full_diff.iter().copied().filter(|x| !x.is_nan()).collect();
    println!("  d=0.0 (original):  memory=full, stationarity=none");
    println!(
        "  d={:.1} (optimal):  memory=partial, stationarity=achieved",
        min_d
    );
    if !full_valid.is_empty() {
        let mean = full_valid.iter().sum::<f64>() / full_valid.len() as f64;
        println!("  d=1.0 (full diff): memory=none, mean={:.4}", mean);
    }

    println!("\nFractional differentiation complete.");
}

fn generate_trending_series(n: usize) -> Vec<f64> {
    let mut price = 100.0;
    (0..n)
        .map(|i| {
            // Random walk with drift (trending and non-stationary)
            let drift = 0.05;
            let noise = (i as f64 * 0.17).sin() * 1.5
                + ((i * 3) as f64 * 0.23).cos() * 0.8
                + ((i * 7) as f64 * 0.07).sin() * 0.4;
            price += drift + noise;
            price = price.max(1.0);
            price
        })
        .collect()
}
