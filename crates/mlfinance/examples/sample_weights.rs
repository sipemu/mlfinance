//! Sample Weighting & Sequential Bootstrap
//!
//! Demonstrates Chapter 4 of "Advances in Financial Machine Learning":
//! - Event concurrency via indicator matrices
//! - Counting co-events and average uniqueness
//! - Sequential bootstrap (vs standard random sampling)
//! - Time-decay sample weights
//!
//! Run with: `cargo run -p mlfinance --example sample_weights`

use mlfinance::sampling::{
    bootstrap::sequential::seq_bootstrap,
    concurrency::{
        average_uniqueness::average_uniqueness, indicator_matrix::get_indicator_matrix,
        num_co_events::num_co_events,
    },
    weights::time_decay::time_decay,
};

fn main() {
    println!("=== Sample Weights & Bootstrap Example ===\n");

    // Step 1: Define overlapping events (start_idx, end_idx)
    // Simulates triple-barrier events that overlap in time
    let events: Vec<(usize, usize)> = vec![
        (0, 4),
        (1, 6),
        (3, 8),
        (5, 9),
        (7, 12),
        (8, 14),
        (10, 15),
        (12, 17),
        (14, 19),
        (16, 20),
        (18, 22),
        (20, 24),
    ];
    let num_bars = 25;

    println!("{} events spanning {} bars", events.len(), num_bars);
    println!("  First 3 events: {:?}", &events[..3]);

    // Step 2: Build the indicator matrix
    println!("\n--- Indicator Matrix ---");
    let ind_matrix = get_indicator_matrix(&events, num_bars);
    println!(
        "  Shape: {} events x {} bars",
        ind_matrix.nrows(),
        ind_matrix.ncols()
    );

    // Step 3: Count co-events per bar
    println!("\n--- Co-Events per Bar ---");
    let co_events = num_co_events(&events, num_bars);
    println!("  Co-event counts: {:?}", co_events);
    let max_concurrent = co_events.iter().max().unwrap_or(&0);
    println!("  Max concurrency: {}", max_concurrent);

    // Step 4: Compute average uniqueness per event
    println!("\n--- Average Uniqueness ---");
    let uniqueness = average_uniqueness(&events, num_bars);
    println!("  Uniqueness per event:");
    for (i, u) in uniqueness.iter().enumerate() {
        println!(
            "    Event {} (bars {}..{}): {:.4}",
            i, events[i].0, events[i].1, u
        );
    }
    let mean_u = uniqueness.iter().sum::<f64>() / uniqueness.len() as f64;
    println!("  Mean uniqueness: {:.4}", mean_u);

    // Step 5: Sequential bootstrap
    println!("\n--- Sequential Bootstrap ---");
    let n_samples = events.len();
    let seq_samples = seq_bootstrap(&ind_matrix, n_samples, 42);
    println!(
        "  {} sequential-bootstrap samples: {:?}",
        seq_samples.len(),
        seq_samples
    );

    // Compare with naive random sampling (uniform)
    let naive: Vec<usize> = (0..n_samples).map(|i| i % events.len()).collect();
    println!("  Naive sequential samples:        {:?}", naive);

    // Show uniqueness of bootstrap samples
    let seq_unique: std::collections::HashSet<usize> = seq_samples.iter().copied().collect();
    println!(
        "  Sequential: {} unique events out of {}",
        seq_unique.len(),
        n_samples
    );

    // Step 6: Time-decay weights
    println!("\n--- Time-Decay Weights ---");
    let uniform_weights = vec![1.0; events.len()];

    // c=1 means oldest weight is 1 (no decay)
    let no_decay = time_decay(&uniform_weights, 1.0);
    // c=0.5 means oldest gets weight 0.5
    let moderate_decay = time_decay(&uniform_weights, 0.5);
    // c=0 means full linear decay to 0
    let full_decay = time_decay(&uniform_weights, 0.0);

    println!("  No decay (c=1.0):       {:?}", format_vec(&no_decay));
    println!(
        "  Moderate decay (c=0.5): {:?}",
        format_vec(&moderate_decay)
    );
    println!("  Full decay (c=0.0):     {:?}", format_vec(&full_decay));

    println!("\nSample weighting complete.");
}

fn format_vec(v: &[f64]) -> String {
    let formatted: Vec<String> = v.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", formatted.join(", "))
}
