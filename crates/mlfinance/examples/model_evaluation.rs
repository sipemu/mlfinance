//! Model evaluation: Event Spans → Purged CV → Bet Sizing → Backtest Stats
//!
//! Demonstrates Chapters 7 and 10 of "Advances in Financial Machine Learning":
//! - Purged K-Fold cross-validation with embargo (Ch. 7)
//! - Bet sizing from predicted probabilities (Ch. 10)
//! - Backtest statistics: Sharpe ratio, drawdowns (Ch. 14)
//!
//! Run with: `cargo run -p mlfinance --example model_evaluation`

use mlfinance::backtesting::bet_sizing::probability_to_size::{power_bet_size, sigmoid_bet_size};
use mlfinance::backtesting::statistics::drawdown::compute_drawdowns;
use mlfinance::backtesting::statistics::sharpe::sharpe_ratio;
use mlfinance::modeling::cross_validation::purged_kfold::PurgedKFold;

fn main() {
    println!("=== Model Evaluation Example ===\n");

    // Step 1: Create event spans (simulating labeled trade events)
    let n_events = 50;
    let event_duration = 5;
    let events: Vec<(usize, usize)> = (0..n_events)
        .map(|i| {
            let start = i * 3; // some overlap between events
            let end = start + event_duration;
            (start, end)
        })
        .collect();

    println!(
        "Created {} events with duration {} bars",
        n_events, event_duration
    );
    println!("  Events span bars 0 to {}", events.last().unwrap().1);

    // Step 2: Purged K-Fold cross-validation
    let n_splits = 5;
    let embargo_pct = 0.02;
    let kfold = PurgedKFold::new(n_splits, embargo_pct);
    let folds = kfold.split(&events, n_events);

    println!(
        "\nPurged K-Fold (n_splits={}, embargo={:.0}%):",
        n_splits,
        embargo_pct * 100.0
    );
    for (i, fold) in folds.iter().enumerate() {
        println!(
            "  Fold {}: train={} samples, test={} samples",
            i,
            fold.train.len(),
            fold.test.len()
        );
    }

    // Verify no leakage
    let mut all_test: Vec<usize> = folds.iter().flat_map(|f| f.test.iter().copied()).collect();
    all_test.sort();
    all_test.dedup();
    println!(
        "  Total unique test samples: {} (should be {})",
        all_test.len(),
        n_events
    );

    // Step 3: Bet sizing from mock probabilities
    println!("\nBet sizing examples:");
    let probabilities = vec![0.3, 0.45, 0.5, 0.55, 0.7, 0.85, 0.95];

    println!(
        "  {:>6} {:>10} {:>10} {:>10}",
        "Prob", "Sigmoid", "Power(2)", "Power(0.5)"
    );
    for &prob in &probabilities {
        let sig = sigmoid_bet_size(prob, 2);
        let pow2 = power_bet_size(prob, 2, 2.0);
        let pow05 = power_bet_size(prob, 2, 0.5);
        println!(
            "  {:>6.2} {:>10.4} {:>10.4} {:>10.4}",
            prob, sig, pow2, pow05
        );
    }

    // Step 4: Simulate backtest returns
    let mock_returns: Vec<f64> = (0..n_events)
        .map(|i| {
            let prob = 0.45 + 0.1 * ((i * 7) as f64 * 0.3).sin();
            let bet = sigmoid_bet_size(prob, 2);
            let underlying_return = ((i * 13) as f64 * 0.5).sin() * 0.02;
            bet * underlying_return
        })
        .collect();

    // Step 5: Compute backtest statistics
    let sr = sharpe_ratio(&mock_returns, 0.0, 252.0);
    let dd = compute_drawdowns(&mock_returns);

    println!("\nBacktest statistics:");
    println!("  Annualized Sharpe ratio:  {:.4}", sr);
    println!("  Maximum drawdown:         {:.4}", dd.max_drawdown);
    println!(
        "  Max drawdown duration:    {} periods",
        dd.max_drawdown_duration
    );
    println!(
        "  Mean return:              {:.6}",
        mock_returns.iter().sum::<f64>() / mock_returns.len() as f64
    );

    println!("\nModel evaluation complete.");
}
