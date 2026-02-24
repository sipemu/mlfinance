//! Backtest Overfitting Detection
//!
//! Demonstrates Chapters 11 and 14 of "Advances in Financial Machine Learning":
//! - Probability of Backtest Overfitting (PBO)
//! - Combinatorially Symmetric Cross-Validation (CSCV)
//! - Bonferroni and Holm multiple testing corrections
//! - Probabilistic Sharpe Ratio (PSR)
//! - Deflated Sharpe Ratio (DSR)
//!
//! Run with: `cargo run -p mlfinance --example overfitting_detection`

use mlfinance::backtesting::{
    overfitting::{
        cscv::cscv,
        multiple_testing::{bonferroni_correction, holm_correction},
        pbo::probability_of_backtest_overfitting,
    },
    statistics::{dsr::deflated_sharpe_ratio, psr::probabilistic_sharpe_ratio},
};
use ndarray::Array2;

fn main() {
    println!("=== Backtest Overfitting Detection ===\n");

    // Step 1: Generate returns for multiple strategies
    let n_strategies = 10;
    let n_periods = 200;
    let returns = generate_strategy_returns(n_strategies, n_periods);
    println!(
        "Simulated {} strategies over {} periods",
        n_strategies, n_periods
    );

    // Print Sharpe ratios for each strategy
    println!("\n--- Strategy Sharpe Ratios ---");
    let mut sharpes = Vec::new();
    for s in 0..n_strategies {
        let strat_returns: Vec<f64> = (0..n_periods).map(|t| returns[[t, s]]).collect();
        let mean = strat_returns.iter().sum::<f64>() / n_periods as f64;
        let std = (strat_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n_periods as f64)
            .sqrt();
        let sr = if std > 0.0 {
            mean / std * (252.0_f64).sqrt()
        } else {
            0.0
        };
        sharpes.push(sr);
        println!("  Strategy {:2}: SR = {:.4}", s, sr);
    }

    let best_idx = sharpes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "  Best: Strategy {} (SR = {:.4})",
        best_idx, sharpes[best_idx]
    );

    // Step 2: Probability of Backtest Overfitting
    println!("\n--- PBO (Probability of Backtest Overfitting) ---");
    let pbo = probability_of_backtest_overfitting(&returns, 8, 42);
    println!("  PBO = {:.4}", pbo);
    println!(
        "  Interpretation: {}",
        if pbo > 0.5 {
            "HIGH risk of overfitting"
        } else {
            "acceptable overfitting risk"
        }
    );

    // Step 3: CSCV
    println!("\n--- CSCV (Combinatorial Cross-Validation) ---");
    let cscv_result = cscv(&returns, 8);
    println!("  CSCV PBO: {:.4}", cscv_result.pbo);
    if !cscv_result.rank_logits.is_empty() {
        let mean_logit =
            cscv_result.rank_logits.iter().sum::<f64>() / cscv_result.rank_logits.len() as f64;
        println!(
            "  Rank logits: {} values, mean = {:.4}",
            cscv_result.rank_logits.len(),
            mean_logit
        );
    }

    // Step 4: Multiple testing corrections
    println!("\n--- Multiple Testing Corrections ---");
    // Simulate p-values from strategy tests
    let p_values: Vec<f64> = sharpes
        .iter()
        .map(|&sr| {
            // Approximate p-value from SR (higher SR → lower p-value)
            let z = sr / (252.0_f64).sqrt() * (n_periods as f64).sqrt();
            // Simple normal CDF approximation for one-sided test
            1.0 / (1.0 + (1.7 * z).exp())
        })
        .collect();

    println!("  Raw p-values:");
    for (i, p) in p_values.iter().enumerate() {
        println!("    Strategy {:2}: p = {:.4}", i, p);
    }

    let bonf = bonferroni_correction(&p_values);
    let holm = holm_correction(&p_values);

    println!("\n  Corrected p-values (alpha=0.05):");
    println!("  Strategy | Raw     | Bonferroni | Holm");
    println!("  ---------|---------|------------|------");
    for i in 0..n_strategies {
        let sig_raw = if p_values[i] < 0.05 { "*" } else { " " };
        let sig_bonf = if bonf[i] < 0.05 { "*" } else { " " };
        let sig_holm = if holm[i] < 0.05 { "*" } else { " " };
        println!(
            "  {:8} | {:.4}{} | {:.4}{}     | {:.4}{}",
            i, p_values[i], sig_raw, bonf[i], sig_bonf, holm[i], sig_holm
        );
    }

    // Step 5: Probabilistic Sharpe Ratio
    println!("\n--- Probabilistic Sharpe Ratio ---");
    let benchmark_sr = 0.0; // Benchmark: zero Sharpe
    let best_returns: Vec<f64> = (0..n_periods).map(|t| returns[[t, best_idx]]).collect();
    let mean = best_returns.iter().sum::<f64>() / n_periods as f64;
    let var = best_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_periods as f64;
    let skew = best_returns
        .iter()
        .map(|r| ((r - mean) / var.sqrt()).powi(3))
        .sum::<f64>()
        / n_periods as f64;
    let kurt = best_returns
        .iter()
        .map(|r| ((r - mean) / var.sqrt()).powi(4))
        .sum::<f64>()
        / n_periods as f64;

    let psr = probabilistic_sharpe_ratio(sharpes[best_idx], benchmark_sr, n_periods, skew, kurt);
    println!(
        "  Best strategy SR: {:.4}, PSR: {:.4}",
        sharpes[best_idx], psr
    );
    println!("  P(true SR > {:.1}) = {:.1}%", benchmark_sr, psr * 100.0);

    // Step 6: Deflated Sharpe Ratio
    println!("\n--- Deflated Sharpe Ratio ---");
    let sr_std = (sharpes
        .iter()
        .map(|s| (s - sharpes.iter().sum::<f64>() / n_strategies as f64).powi(2))
        .sum::<f64>()
        / n_strategies as f64)
        .sqrt();
    let dsr = deflated_sharpe_ratio(
        sharpes[best_idx],
        sr_std,
        n_periods,
        n_strategies,
        skew,
        kurt,
    );
    println!("  DSR: {:.4} (accounts for {} trials)", dsr, n_strategies);
    println!(
        "  Interpretation: {}",
        if dsr > 0.95 {
            "strong evidence of skill"
        } else if dsr > 0.5 {
            "moderate evidence"
        } else {
            "likely overfitting"
        }
    );

    println!("\nOverfitting detection complete.");
}

fn generate_strategy_returns(n_strategies: usize, n_periods: usize) -> Array2<f64> {
    let mut returns = Array2::zeros((n_periods, n_strategies));
    for s in 0..n_strategies {
        for t in 0..n_periods {
            // Each strategy has different characteristics
            let signal = ((t * (s + 1) * 3) as f64 * 0.07).sin() * 0.01;
            let noise = ((t * 17 + s * 31) as f64 * 0.13).sin() * 0.02;
            let alpha = if s == 0 {
                0.0005 // Strategy 0 has slight positive alpha
            } else {
                (s as f64 * 0.0001).sin() * 0.0002
            };
            returns[[t, s]] = alpha + signal + noise;
        }
    }
    returns
}
