//! Structural Breaks & Bubble Detection
//!
//! Demonstrates Chapter 17 of "Advances in Financial Machine Learning":
//! - Augmented Dickey-Fuller test for stationarity
//! - SADF (Supremum ADF) for bubble detection
//! - GSADF (Generalized SADF) for multi-bubble detection
//! - Brown-Durbin-Evans CUSUM test
//! - Chu-Stinchcombe-White CUSUM test
//!
//! Run with: `cargo run -p mlfinance --example structural_breaks`

use mlfinance::features::structural_breaks::{
    adf::adf_test,
    cusum_tests::{brown_durbin_evans, chu_stinchcombe_white},
    gsadf::gsadf,
    sadf::sadf,
};

fn main() {
    println!("=== Structural Breaks & Bubble Detection ===\n");

    // Step 1: Generate a series with an embedded bubble
    let series = generate_bubble_series(200);
    println!(
        "Generated {} observations with embedded bubble",
        series.len()
    );
    println!(
        "  Start: {:.2}, Peak: {:.2}, End: {:.2}",
        series[0],
        series.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        series[series.len() - 1]
    );

    // Step 2: ADF test on original series
    println!("\n--- ADF Test ---");
    let (adf_stat, betas) = adf_test(&series, 4);
    println!("  ADF statistic: {:.4}", adf_stat);
    println!("  Regression coefficients: {} betas", betas.len());
    println!(
        "  Interpretation: {} (5% critical ~-2.87)",
        if adf_stat < -2.87 {
            "stationary"
        } else {
            "non-stationary"
        }
    );

    // Step 3: Log prices for SADF/GSADF
    let log_prices: Vec<f64> = series.iter().map(|p| p.ln()).collect();

    // Step 4: SADF test
    println!("\n--- SADF Test ---");
    let min_window = 20;
    let sadf_stats = sadf(&log_prices, min_window, 4);
    if !sadf_stats.is_empty() {
        let sadf_max = sadf_stats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("  {} SADF statistics computed", sadf_stats.len());
        println!("  SADF (supremum): {:.4}", sadf_max);
        println!(
            "  Bubble detected: {} (critical ~1.0)",
            if sadf_max > 1.0 { "YES" } else { "no" }
        );
    }

    // Step 5: GSADF test
    println!("\n--- GSADF Test ---");
    let gsadf_stats = gsadf(&log_prices, min_window, 4);
    if !gsadf_stats.is_empty() {
        let gsadf_max = gsadf_stats
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("  {} GSADF statistics computed", gsadf_stats.len());
        println!("  GSADF (supremum): {:.4}", gsadf_max);
        println!(
            "  Bubble detected: {} (critical ~2.0)",
            if gsadf_max > 2.0 { "YES" } else { "no" }
        );
    }

    // Step 6: Brown-Durbin-Evans CUSUM test
    println!("\n--- Brown-Durbin-Evans CUSUM ---");
    // Compute residuals (first differences as a simple proxy)
    let residuals: Vec<f64> = log_prices.windows(2).map(|w| w[1] - w[0]).collect();
    let (cusum_values, critical_boundary) = brown_durbin_evans(&residuals);
    let max_cusum = cusum_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  Critical boundary: {:.4}", critical_boundary);
    println!("  Max CUSUM value:   {:.4}", max_cusum);
    println!(
        "  Break detected:    {}",
        if max_cusum > critical_boundary {
            "YES"
        } else {
            "no"
        }
    );

    // Step 7: Chu-Stinchcombe-White CUSUM test
    println!("\n--- Chu-Stinchcombe-White CUSUM ---");
    let csw_stats = chu_stinchcombe_white(&log_prices, 1.358);
    let csw_max = csw_stats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  {} CSW statistics computed", csw_stats.len());
    println!("  Max CSW statistic: {:.4}", csw_max);
    println!(
        "  Break detected:    {}",
        if csw_max > 1.0 { "YES" } else { "no" }
    );

    println!("\nStructural breaks analysis complete.");
}

fn generate_bubble_series(n: usize) -> Vec<f64> {
    let mut price = 100.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            if t < 0.3 {
                // Normal growth
                price += 0.1 + (i as f64 * 0.13).sin() * 0.5;
            } else if t < 0.6 {
                // Bubble: exponential growth
                price *= 1.015 + (i as f64 * 0.07).sin().abs() * 0.005;
            } else if t < 0.7 {
                // Crash
                price *= 0.97 - (i as f64 * 0.11).sin().abs() * 0.01;
            } else {
                // Recovery
                price += 0.05 + (i as f64 * 0.19).sin() * 0.3;
            }
            price = price.max(10.0);
            price
        })
        .collect()
}
