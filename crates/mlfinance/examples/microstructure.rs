//! Market Microstructure Features
//!
//! Demonstrates Chapter 19 of "Advances in Financial Machine Learning":
//! - VPIN (Volume-Synchronized Probability of Informed Trading)
//! - Amihud illiquidity (price impact per dollar volume)
//! - Kyle's lambda (price impact coefficient)
//! - Roll spread (bid-ask spread from price autocovariance)
//! - Corwin-Schultz spread (high-low based spread estimator)
//!
//! Run with: `cargo run -p mlfinance --example microstructure`

use mlfinance::features::microstructure::{
    amihud_lambda::amihud_lambda, corwin_schultz::corwin_schultz_spread, kyle_lambda::kyle_lambda,
    roll_model::roll_spread, vpin::vpin,
};

fn main() {
    println!("=== Market Microstructure Features ===\n");

    // Step 1: Generate synthetic trade data
    let n = 500;
    let (prices, volumes, highs, lows) = generate_trade_data(n);
    println!("Generated {} observations of trade data", n);
    println!(
        "  Price range: {:.2} to {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Step 2: VPIN
    println!("\n--- VPIN ---");
    let bucket_size = volumes.iter().sum::<f64>() / 50.0; // ~50 buckets
    let n_buckets = 10;
    let vpin_values = vpin(&volumes, &prices, bucket_size, n_buckets);
    if !vpin_values.is_empty() {
        let mean_vpin = vpin_values.iter().sum::<f64>() / vpin_values.len() as f64;
        let max_vpin = vpin_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("  {} VPIN values computed", vpin_values.len());
        println!("  Mean VPIN: {:.4}", mean_vpin);
        println!(
            "  Max VPIN:  {:.4} (higher = more informed trading)",
            max_vpin
        );
    } else {
        println!("  No VPIN values (series too short for bucket params)");
    }

    // Step 3: Amihud lambda (illiquidity)
    println!("\n--- Amihud Lambda ---");
    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let dollar_volumes: Vec<f64> = prices
        .iter()
        .zip(volumes.iter())
        .skip(1)
        .map(|(p, v)| p * v)
        .collect();
    let amihud = amihud_lambda(&returns, &dollar_volumes);
    println!("  Amihud lambda: {:.8} (higher = less liquid)", amihud);

    // Step 4: Kyle's lambda
    println!("\n--- Kyle's Lambda ---");
    let signed_volumes: Vec<f64> = returns
        .iter()
        .zip(volumes.iter().skip(1))
        .map(|(r, v)| v * r.signum())
        .collect();
    let kyle = kyle_lambda(&returns, &signed_volumes);
    println!("  Kyle's lambda: {:.8} (price impact per unit flow)", kyle);

    // Step 5: Roll spread
    println!("\n--- Roll Spread ---");
    let roll = roll_spread(&prices);
    println!("  Effective spread: {:.6}", roll);
    println!(
        "  As % of price:   {:.4}%",
        100.0 * roll / prices.iter().sum::<f64>() * prices.len() as f64
    );

    // Step 6: Corwin-Schultz spread
    println!("\n--- Corwin-Schultz Spread ---");
    let cs_spreads = corwin_schultz_spread(&highs, &lows);
    if !cs_spreads.is_empty() {
        let valid: Vec<f64> = cs_spreads
            .iter()
            .copied()
            .filter(|x| !x.is_nan() && x.is_finite())
            .collect();
        let mean_cs = if !valid.is_empty() {
            valid.iter().sum::<f64>() / valid.len() as f64
        } else {
            0.0
        };
        let max_cs = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {} spread estimates ({} valid)",
            cs_spreads.len(),
            valid.len()
        );
        println!("  Mean spread: {:.6}", mean_cs);
        println!("  Max spread:  {:.6}", max_cs);
    }

    // Step 7: Summary
    println!("\n--- Microstructure Summary ---");
    println!("  Metric              | Value");
    println!("  --------------------|--------");
    if !vpin_values.is_empty() {
        println!(
            "  VPIN (mean)         | {:.4}",
            vpin_values.iter().sum::<f64>() / vpin_values.len() as f64
        );
    }
    println!("  Amihud lambda       | {:.8}", amihud);
    println!("  Kyle's lambda       | {:.8}", kyle);
    println!("  Roll spread         | {:.6}", roll);

    println!("\nMicrostructure analysis complete.");
}

fn generate_trade_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut price = 50.0;
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);

    for i in 0..n {
        let noise = (i as f64 * 0.17).sin() * 0.3
            + ((i * 3) as f64 * 0.11).cos() * 0.15
            + ((i * 7) as f64 * 0.03).sin() * 0.1;
        price = (price + noise).max(10.0);

        let spread = 0.05 + (i as f64 * 0.07).sin().abs() * 0.1;
        let high = price + spread;
        let low = price - spread;

        prices.push(price);
        volumes.push(1000.0 + (i as f64 * 0.05).sin() * 500.0 + 500.0);
        highs.push(high);
        lows.push(low.max(1.0));
    }

    (prices, volumes, highs, lows)
}
