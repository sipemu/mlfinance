//! Random Matrix Theory Denoising
//!
//! Demonstrates Chapter 2 (Part 2) of "Advances in Financial Machine Learning":
//! - Marcenko-Pastur distribution for noise eigenvalues
//! - Correlation matrix denoising via RMT
//! - Correlation matrix detoning (removing market mode)
//! - Optimal portfolio comparison: noisy vs denoised
//!
//! Run with: `cargo run -p mlfinance --example denoising`

use mlfinance::core::stats::{correlation_matrix, covariance_matrix};
use mlfinance::features::denoising::{
    denoise_corr, detone_corr, marcenko_pastur_pdf, optimal_portfolio,
};
use ndarray::{Array1, Array2};

fn main() {
    println!("=== Random Matrix Theory Denoising ===\n");

    // Step 1: Generate a noisy correlation matrix
    let n_assets = 10;
    let n_obs = 200;
    let q = n_obs as f64 / n_assets as f64; // T/N ratio
    println!(
        "Simulated: {} assets, {} observations (q = T/N = {:.1})",
        n_assets, n_obs, q
    );

    let (corr, cov) = generate_noisy_corr(n_assets, n_obs);
    println!("  Correlation matrix: {}x{}", corr.nrows(), corr.ncols());

    // Step 2: Marcenko-Pastur distribution
    println!("\n--- Marcenko-Pastur Distribution ---");
    let variance = 1.0; // unit variance for noise
    let (mp_x, mp_pdf) = marcenko_pastur_pdf(variance, q, 100).expect("MP PDF failed");
    let lambda_min = mp_x[0];
    let lambda_max = mp_x[mp_x.len() - 1];
    println!(
        "  Noise eigenvalue range: [{:.4}, {:.4}]",
        lambda_min, lambda_max
    );
    println!(
        "  PDF peak: {:.4} at lambda={:.4}",
        mp_pdf.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mp_x[mp_pdf
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)]
    );

    // Step 3: Denoise the correlation matrix
    println!("\n--- Denoising ---");
    let denoised = denoise_corr(&corr, q, None, false, None).expect("Denoise failed");
    println!("  Denoised correlation matrix computed");

    // Compare off-diagonal elements
    let off_diag_orig = off_diagonal_stats(&corr);
    let off_diag_clean = off_diagonal_stats(&denoised);
    println!(
        "  Original off-diag: mean={:.4}, std={:.4}",
        off_diag_orig.0, off_diag_orig.1
    );
    println!(
        "  Denoised off-diag: mean={:.4}, std={:.4}",
        off_diag_clean.0, off_diag_clean.1
    );

    // Step 4: Denoise with shrinkage
    println!("\n--- Denoising with Shrinkage ---");
    let denoised_shrink =
        denoise_corr(&corr, q, None, true, Some(0.5)).expect("Shrinkage denoise failed");
    let off_diag_shrink = off_diagonal_stats(&denoised_shrink);
    println!(
        "  Shrunk off-diag:   mean={:.4}, std={:.4}",
        off_diag_shrink.0, off_diag_shrink.1
    );

    // Step 5: Detone (remove market mode)
    println!("\n--- Detoning ---");
    let detoned = detone_corr(&denoised, 1).expect("Detone failed");
    let off_diag_detone = off_diagonal_stats(&detoned);
    println!(
        "  Detoned off-diag:  mean={:.4}, std={:.4}",
        off_diag_detone.0, off_diag_detone.1
    );

    // Step 6: Optimal portfolio comparison
    println!("\n--- Optimal Portfolios ---");
    let mu = Array1::from_vec(vec![0.05; n_assets]); // Equal expected returns

    let w_noisy = optimal_portfolio(&cov, Some(&mu)).expect("Noisy portfolio failed");
    let w_denoised = {
        // Reconstruct covariance from denoised correlation
        let std_devs: Vec<f64> = (0..n_assets).map(|i| cov[[i, i]].sqrt()).collect();
        let mut denoised_cov = Array2::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..n_assets {
                denoised_cov[[i, j]] = denoised[[i, j]] * std_devs[i] * std_devs[j];
            }
        }
        optimal_portfolio(&denoised_cov, Some(&mu)).expect("Denoised portfolio failed")
    };

    println!("  Noisy weights:    {}", format_weights(&w_noisy));
    println!("  Denoised weights: {}", format_weights(&w_denoised));

    // Compare concentration
    let hhi_noisy: f64 = w_noisy.iter().map(|w| w * w).sum();
    let hhi_denoised: f64 = w_denoised.iter().map(|w| w * w).sum();
    println!(
        "  Noisy HHI:    {:.4} (1/{} = {:.4} ideal)",
        hhi_noisy,
        n_assets,
        1.0 / n_assets as f64
    );
    println!("  Denoised HHI: {:.4}", hhi_denoised);

    println!("\nDenoising analysis complete.");
}

fn off_diagonal_stats(m: &Array2<f64>) -> (f64, f64) {
    let n = m.nrows();
    let mut vals = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                vals.push(m[[i, j]]);
            }
        }
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    (mean, var.sqrt())
}

fn format_weights(w: &Array1<f64>) -> String {
    let s: Vec<String> = w.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", s.join(", "))
}

fn generate_noisy_corr(n: usize, t: usize) -> (Array2<f64>, Array2<f64>) {
    // Generate pseudo-random returns with distinct per-asset frequencies
    let mut data = Array2::zeros((t, n));
    // Use prime-based frequencies to avoid linear dependence between columns
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for i in 0..t {
        for j in 0..n {
            let freq1 = primes[j % primes.len()] as f64;
            let freq2 = primes[(j + 3) % primes.len()] as f64;
            let freq3 = primes[(j + 7) % primes.len()] as f64;
            // Each asset has unique oscillation frequencies
            let val = (i as f64 * freq1 * 0.031).sin() * 0.02
                + (i as f64 * freq2 * 0.017).cos() * 0.015
                + (i as f64 * freq3 * 0.053).sin() * 0.01;
            // Common market factor
            let market = (i as f64 * 0.03).sin() * 0.008;
            data[[i, j]] = val + market;
        }
    }

    // Compute sample covariance and correlation using library functions
    let cov = covariance_matrix(&data).expect("Covariance computation failed");
    let corr = correlation_matrix(&data).expect("Correlation computation failed");

    (corr, cov)
}
