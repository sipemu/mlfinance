//! Volatility Estimation: Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang
//!
//! Demonstrates volatility estimators from Chapter 3:
//! - Daily (close-to-close) EWMA volatility
//! - Parkinson high-low range estimator
//! - Garman-Klass OHLC estimator
//! - Yang-Zhang estimator (most efficient)
//!
//! Run with: `cargo run -p mlfinance --example volatility_estimators`

use mlfinance::core::types::{OhlcvBar, Timestamp};
use mlfinance::labeling::volatility::{
    daily_volatility, garman_klass_volatility, parkinson_volatility, yang_zhang_volatility,
};

fn main() {
    println!("=== Volatility Estimators Comparison ===\n");

    // Step 1: Generate synthetic OHLC data
    let bars = generate_ohlc_bars(200);
    println!("Generated {} OHLC bars", bars.len());
    if let Some(b) = bars.first() {
        println!(
            "  First bar: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            b.open, b.high, b.low, b.close, b.volume
        );
    }

    let prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let timestamps: Vec<Timestamp> = bars.iter().map(|b| b.timestamp).collect();

    // Step 2: Daily (close-to-close) volatility
    println!("\n--- Daily Volatility (EWMA, span=20) ---");
    let daily_vol = daily_volatility(&prices, &timestamps, 20);
    print_vol_stats("Daily", &daily_vol);

    // Step 3: Parkinson volatility
    println!("\n--- Parkinson Volatility (window=20) ---");
    let park_vol = parkinson_volatility(&bars, 20).expect("Parkinson failed");
    print_vol_stats("Parkinson", &park_vol);

    // Step 4: Garman-Klass volatility
    println!("\n--- Garman-Klass Volatility (window=20) ---");
    let gk_vol = garman_klass_volatility(&bars, 20).expect("Garman-Klass failed");
    print_vol_stats("Garman-Klass", &gk_vol);

    // Step 5: Yang-Zhang volatility
    println!("\n--- Yang-Zhang Volatility (window=20) ---");
    let yz_vol = yang_zhang_volatility(&bars, 20).expect("Yang-Zhang failed");
    print_vol_stats("Yang-Zhang", &yz_vol);

    // Step 6: Side-by-side comparison at the last bar
    println!("\n--- Comparison at Last Bar ---");
    let n = bars.len() - 1;
    println!("  Daily (EWMA):  {:.6}", daily_vol[n]);
    println!("  Parkinson:     {:.6}", park_vol[n]);
    println!("  Garman-Klass:  {:.6}", gk_vol[n]);
    println!("  Yang-Zhang:    {:.6}", yz_vol[n]);

    println!("\nVolatility estimation complete.");
}

fn print_vol_stats(name: &str, vols: &[f64]) {
    let valid: Vec<f64> = vols
        .iter()
        .copied()
        .filter(|x| !x.is_nan() && *x > 0.0)
        .collect();
    if valid.is_empty() {
        println!("  {}: no valid values", name);
        return;
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  {} valid values, mean={:.6}, min={:.6}, max={:.6}",
        valid.len(),
        mean,
        min,
        max
    );
}

fn generate_ohlc_bars(n: usize) -> Vec<OhlcvBar> {
    let mut close = 100.0;
    (0..n)
        .map(|i| {
            let prev_close = close;
            let noise = (i as f64 * 0.13).sin() * 1.2
                + ((i * 3) as f64 * 0.07).cos() * 0.6
                + ((i * 7) as f64 * 0.31).sin() * 0.3;
            close = (prev_close + noise).max(10.0);

            // Realistic OHLC from close movement
            let open = prev_close + noise * 0.3;
            let high = open.max(close) + (0.5 + (i as f64 * 0.2).sin().abs() * 0.8);
            let low = open.min(close) - (0.5 + (i as f64 * 0.15).cos().abs() * 0.8);

            OhlcvBar {
                timestamp: chrono::DateTime::from_timestamp(1_700_000_000 + (i as i64) * 86400, 0)
                    .unwrap(),
                open,
                high,
                low: low.max(1.0),
                close,
                volume: 1000.0 + (i as f64 * 0.1).sin() * 300.0,
                vwap: (open + high + low + close) / 4.0,
            }
        })
        .collect()
}
