//! Strategy Risk Analysis
//!
//! Demonstrates strategy risk metrics from "Advances in Financial Machine Learning":
//! - Expected Sharpe Ratio from precision and frequency
//! - Implied precision for a target Sharpe Ratio
//! - Implied betting frequency for a target Sharpe Ratio
//! - Strategy failure probability
//! - Ornstein-Uhlenbeck process simulation and parameter estimation
//!
//! Run with: `cargo run -p mlfinance --example strategy_risk`

use mlfinance::backtesting::{
    strategy_risk::{
        failure_probability::strategy_failure_probability, implied_frequency::implied_frequency,
        implied_precision::implied_precision, sr_from_precision::sr_from_precision,
    },
    synthetic::ornstein_uhlenbeck::{estimate_ou_params, simulate_ou},
};

fn main() {
    println!("=== Strategy Risk Analysis ===\n");

    // Step 1: Sharpe Ratio from precision and frequency
    println!("--- SR from Precision & Frequency ---");
    let freq = 252.0; // Daily trading
    let win_loss_ratio = 1.0; // Equal average win and loss

    println!(
        "  Frequency: {} trades/year, Win/Loss ratio: {:.1}",
        freq, win_loss_ratio
    );
    println!("  Precision | Expected SR");
    println!("  ----------|------------");
    for &precision in &[0.45, 0.50, 0.51, 0.52, 0.55, 0.60] {
        let sr = sr_from_precision(precision, freq, win_loss_ratio);
        println!("    {:.2}     |  {:.4}", precision, sr);
    }

    // Step 2: Implied precision for target SR
    println!("\n--- Implied Precision ---");
    println!("  Target SR | Required Precision");
    println!("  ----------|-------------------");
    for &target_sr in &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
        let required_precision = implied_precision(target_sr, freq, win_loss_ratio);
        println!(
            "    {:.1}      |  {:.4} ({:.1}%)",
            target_sr,
            required_precision,
            required_precision * 100.0
        );
    }

    // Step 3: Implied frequency for target SR
    println!("\n--- Implied Frequency ---");
    let precision = 0.55; // 55% hit rate
    println!(
        "  Precision: {:.0}%, Win/Loss: {:.1}",
        precision * 100.0,
        win_loss_ratio
    );
    println!("  Target SR | Required Freq (trades/year)");
    println!("  ----------|---------------------------");
    for &target_sr in &[0.5, 1.0, 1.5, 2.0] {
        let required_freq = implied_frequency(target_sr, precision, win_loss_ratio);
        println!(
            "    {:.1}      |  {:.1} ({:.1}/day)",
            target_sr,
            required_freq,
            required_freq / 252.0
        );
    }

    // Step 4: Strategy failure probability
    println!("\n--- Strategy Failure Probability ---");
    let estimated_precision = 0.55;
    let break_even = 0.50; // Break-even at 50% for equal win/loss
    println!(
        "  Estimated precision: {:.0}%, Break-even: {:.0}%",
        estimated_precision * 100.0,
        break_even * 100.0
    );
    println!("  Observations | Failure Prob");
    println!("  -------------|-------------");
    for &n in &[50, 100, 200, 500, 1000] {
        let prob = strategy_failure_probability(estimated_precision, n, break_even);
        let bar = "#".repeat((prob * 40.0) as usize);
        println!("    {:5}      |  {:.4} {}", n, prob, bar);
    }

    // Step 5: Ornstein-Uhlenbeck simulation
    println!("\n--- Ornstein-Uhlenbeck Process ---");
    let theta = 5.0; // Mean reversion speed
    let mu = 100.0; // Long-run mean
    let sigma = 2.0; // Volatility
    let x0 = 95.0; // Start below mean
    let dt = 1.0 / 252.0; // Daily
    let n_steps = 504; // ~2 years

    println!(
        "  Parameters: theta={:.1}, mu={:.1}, sigma={:.1}, x0={:.1}",
        theta, mu, sigma, x0
    );

    let path = simulate_ou(theta, mu, sigma, x0, dt, n_steps, 42);
    println!("  Simulated {} steps ({} days)", n_steps, n_steps);
    let path_mean = path.iter().sum::<f64>() / path.len() as f64;
    let path_min = path.iter().cloned().fold(f64::INFINITY, f64::min);
    let path_max = path.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  Path stats: mean={:.2}, min={:.2}, max={:.2}",
        path_mean, path_min, path_max
    );
    println!(
        "  Start: {:.2}, End: {:.2} (mean={:.1})",
        path[0],
        path[path.len() - 1],
        mu
    );

    // Step 6: Estimate OU parameters from the simulated path
    println!("\n--- OU Parameter Estimation ---");
    let (est_theta, est_mu, est_sigma) = estimate_ou_params(&path, dt);
    println!(
        "  True     → theta={:.2}, mu={:.2}, sigma={:.2}",
        theta, mu, sigma
    );
    println!(
        "  Estimated → theta={:.2}, mu={:.2}, sigma={:.2}",
        est_theta, est_mu, est_sigma
    );
    println!(
        "  Error     → theta={:.1}%, mu={:.1}%, sigma={:.1}%",
        100.0 * (est_theta - theta).abs() / theta,
        100.0 * (est_mu - mu).abs() / mu,
        100.0 * (est_sigma - sigma).abs() / sigma,
    );

    // Step 7: Risk metrics for the mean-reverting strategy
    println!("\n--- Combined Risk Assessment ---");
    let strategy_sr = sr_from_precision(estimated_precision, freq, win_loss_ratio);
    let failure_prob = strategy_failure_probability(estimated_precision, 252, break_even);
    let required_precision_2 = implied_precision(2.0, freq, win_loss_ratio);

    println!("  Strategy expected SR:      {:.4}", strategy_sr);
    println!("  Failure probability (1yr): {:.4}", failure_prob);
    println!(
        "  Precision for SR=2.0:      {:.1}%",
        required_precision_2 * 100.0
    );
    println!(
        "  Current precision gap:     {:.1}pp",
        (required_precision_2 - estimated_precision) * 100.0
    );

    println!("\nStrategy risk analysis complete.");
}
