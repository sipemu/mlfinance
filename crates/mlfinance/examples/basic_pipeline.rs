//! Basic AFML pipeline: Ticks → Bars → CUSUM → Volatility → Triple Barrier → Labels
//!
//! Demonstrates Chapters 2-3 of "Advances in Financial Machine Learning":
//! - Tick bar aggregation (Ch. 2)
//! - CUSUM filter for event detection (Ch. 2)
//! - Triple barrier labeling (Ch. 3)
//!
//! Run with: `cargo run -p mlfinance --example basic_pipeline`

use mlfinance::core::traits::BarAggregator;
use mlfinance::core::types::TickData;
use mlfinance::data::bars::tick_bars::TickBarAggregator;
use mlfinance::data::sampling::cusum_filter::cusum_filter;
use mlfinance::labeling::barriers::{BarrierTouchType, TripleBarrierConfig};
use mlfinance::labeling::events::get_events;
use mlfinance::labeling::labels::get_bins;

fn main() {
    println!("=== AFML Basic Pipeline Example ===\n");

    // Step 1: Generate synthetic tick data
    let ticks = generate_ticks(3000);
    println!("Generated {} synthetic ticks", ticks.len());

    // Step 2: Aggregate into tick bars (20 ticks per bar)
    let mut agg = TickBarAggregator::new(20);
    let bars = agg.process_ticks(&ticks);
    println!("Aggregated into {} tick bars", bars.len());
    if let Some(bar) = bars.first() {
        println!(
            "  First bar: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            bar.open, bar.high, bar.low, bar.close, bar.volume
        );
    }

    // Step 3: Extract close prices and apply CUSUM filter
    let prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let cusum_events = cusum_filter(&prices, 1.5);
    println!(
        "\nCUSUM filter (threshold=1.5): detected {} events",
        cusum_events.len()
    );
    if cusum_events.len() > 3 {
        println!("  First 3 event indices: {:?}", &cusum_events[..3]);
    }

    // Step 4: Set up daily volatility estimate
    let daily_vols = vec![0.02; prices.len()]; // 2% daily vol

    // Step 5: Apply triple barrier labeling
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),     // 2x daily vol profit-taking
        lower_barrier: Some(2.0),     // 2x daily vol stop-loss
        max_holding_period: Some(15), // 15-bar holding period
    };

    let events = get_events(&prices, &cusum_events, &config, &daily_vols);
    println!("\nTriple barrier: {} events labeled", events.len());

    // Step 6: Generate labels
    let labels = get_bins(&events);
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = labels.iter().filter(|&&l| l == -1).count();
    let n_zero = labels.iter().filter(|&&l| l == 0).count();

    println!("\nLabel distribution:");
    println!("  +1 (profit): {}", n_pos);
    println!("  -1 (loss):   {}", n_neg);
    println!("   0 (flat):   {}", n_zero);

    // Show first few events
    println!("\nFirst 5 events:");
    for (i, event) in events.iter().take(5).enumerate() {
        let touch = match event.touch_type {
            BarrierTouchType::Upper => "Upper",
            BarrierTouchType::Lower => "Lower",
            BarrierTouchType::Vertical => "Vertical",
        };
        println!(
            "  [{}] entry={} exit={} touch={} return={:.4}",
            i, event.entry_idx, event.exit_idx, touch, event.return_value
        );
    }

    println!("\nPipeline complete.");
}

fn generate_ticks(n: usize) -> Vec<TickData> {
    let mut price = 100.0;
    (0..n)
        .map(|i| {
            price += (i as f64 * 0.03).sin() * 0.5 + ((i * 7) as f64 * 0.1).sin() * 0.2;
            price = price.max(1.0);
            TickData {
                timestamp: chrono::DateTime::from_timestamp(1_700_000_000 + i as i64, 0).unwrap(),
                price,
                volume: (100.0 + (i as f64 * 0.07).cos() * 50.0).abs(),
            }
        })
        .collect()
}
