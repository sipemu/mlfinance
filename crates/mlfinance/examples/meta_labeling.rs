//! Meta-Labeling & Trend Scanning
//!
//! Demonstrates Chapter 3 and Chapter 5 of "Advances in Financial Machine Learning":
//! - Trend scanning for label generation
//! - Meta-labeling: learning when the primary model is correct
//! - Bet sizing from meta-label probabilities
//!
//! Run with: `cargo run -p mlfinance --example meta_labeling`

use mlfinance::labeling::{
    barriers::TripleBarrierConfig,
    events::get_events,
    labels::{get_bins, get_meta_bins},
    meta_labeling::MetaLabeler,
    trend_scanning::trend_scanning_label_series,
};

fn main() {
    println!("=== Meta-Labeling Example ===\n");

    // Step 1: Generate synthetic price series
    let prices = generate_prices(300);
    println!("Generated {} prices", prices.len());

    // Step 2: Trend scanning labels
    println!("\n--- Trend Scanning ---");
    let max_window = 20;
    let trend_labels =
        trend_scanning_label_series(&prices, max_window).expect("Trend scanning failed");
    let n_up = trend_labels.iter().filter(|&&l| l == 1).count();
    let n_down = trend_labels.iter().filter(|&&l| l == -1).count();
    let n_flat = trend_labels.iter().filter(|&&l| l == 0).count();
    println!("  Trend labels (window={}):", max_window);
    println!("    +1 (uptrend):   {}", n_up);
    println!("    -1 (downtrend): {}", n_down);
    println!("     0 (flat):      {}", n_flat);

    // Step 3: Primary model predictions (use trend scanning as primary model)
    println!("\n--- Primary Model ---");
    // Simulate a primary model that predicts direction
    let primary_predictions = simulate_primary_model(&prices);
    let n_pred_pos = primary_predictions.iter().filter(|&&p| p == 1).count();
    println!(
        "  Primary model: {} buy, {} sell signals",
        n_pred_pos,
        primary_predictions.len() - n_pred_pos
    );

    // Step 4: Generate events via triple-barrier labeling
    println!("\n--- Triple Barrier Events ---");
    let daily_vols = vec![0.02; prices.len()];
    let event_indices: Vec<usize> = (10..prices.len() - 20).step_by(5).collect();
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),
        lower_barrier: Some(2.0),
        max_holding_period: Some(15),
    };
    let events = get_events(&prices, &event_indices, &config, &daily_vols);
    println!(
        "  {} events from {} candidates",
        events.len(),
        event_indices.len()
    );

    let labels = get_bins(&events);
    let n_profit = labels.iter().filter(|&&l| l == 1).count();
    let n_loss = labels.iter().filter(|&&l| l == -1).count();
    println!(
        "  Labels: {} profit, {} loss, {} flat",
        n_profit,
        n_loss,
        labels.len() - n_profit - n_loss
    );

    // Step 5: Meta-labeling
    println!("\n--- Meta-Labeling ---");
    // Align primary predictions with events
    let aligned_predictions: Vec<i32> = events
        .iter()
        .map(|e| primary_predictions[e.entry_idx.min(primary_predictions.len() - 1)])
        .collect();

    let meta_bins = get_meta_bins(&events, &aligned_predictions);
    let n_correct = meta_bins.iter().filter(|&&m| m == 1).count();
    let n_incorrect = meta_bins.iter().filter(|&&m| m == 0).count();
    println!(
        "  Meta-labels: {} correct, {} incorrect (accuracy: {:.1}%)",
        n_correct,
        n_incorrect,
        100.0 * n_correct as f64 / meta_bins.len().max(1) as f64
    );

    // Step 6: MetaLabeler with bet sizing
    println!("\n--- Bet Sizing ---");
    let meta_labeler = MetaLabeler::new(0.5);
    let meta_labels = meta_labeler.generate_labels(&events, &aligned_predictions);
    println!(
        "  MetaLabeler (min_prob=0.5): {} correct, {} incorrect",
        meta_labels.iter().filter(|&&l| l == 1).count(),
        meta_labels.iter().filter(|&&l| l == 0).count(),
    );

    // Demonstrate bet sizing for various confidence levels
    println!("\n  Probability → Bet Size:");
    for &prob in &[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        let size = meta_labeler.bet_size(prob);
        let bar = "#".repeat((size * 20.0) as usize);
        println!("    p={:.1}: size={:.2} {}", prob, size, bar);
    }

    // Step 7: Summary
    println!("\n--- Pipeline Summary ---");
    println!("  1. Price series:        {} observations", prices.len());
    println!(
        "  2. Primary model:       {} predictions",
        primary_predictions.len()
    );
    println!("  3. Triple barrier:      {} events", events.len());
    println!("  4. Meta-labels:         {} labels", meta_bins.len());
    println!(
        "  5. Primary accuracy:    {:.1}%",
        100.0 * n_correct as f64 / meta_bins.len().max(1) as f64
    );

    println!("\nMeta-labeling complete.");
}

fn generate_prices(n: usize) -> Vec<f64> {
    let mut price = 100.0;
    (0..n)
        .map(|i| {
            let trend = (i as f64 / 50.0).sin() * 2.0;
            let noise = (i as f64 * 0.17).sin() * 0.5 + ((i * 3) as f64 * 0.11).cos() * 0.3;
            price += trend * 0.1 + noise;
            price = price.max(50.0);
            price
        })
        .collect()
}

fn simulate_primary_model(prices: &[f64]) -> Vec<i32> {
    // Simple momentum model: predict based on recent trend
    let lookback = 5;
    prices
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i < lookback {
                1
            } else {
                let momentum = prices[i] - prices[i - lookback];
                if momentum > 0.0 {
                    1
                } else {
                    -1
                }
            }
        })
        .collect()
}
