//! Entropy & Complexity Measures
//!
//! Demonstrates Chapter 18 of "Advances in Financial Machine Learning":
//! - Encoding methods: binary, quantile, sigma
//! - Shannon entropy (from a probability distribution)
//! - Plug-in entropy estimator (from a symbol sequence)
//! - Lempel-Ziv complexity (from a binary sequence)
//! - Kontoyiannis entropy estimator (from a symbol sequence)
//!
//! Run with: `cargo run -p mlfinance --example entropy`

use mlfinance::features::entropy::{
    encoding::{binary_encode, quantile_encode, sigma_encode},
    kontoyiannis::kontoyiannis_entropy,
    lempel_ziv::lempel_ziv_complexity,
    plugin::plugin_entropy,
    shannon::shannon_entropy,
};

fn main() {
    println!("=== Entropy & Complexity Example ===\n");

    // Step 1: Generate a synthetic price returns series
    let returns = generate_returns(500);
    println!("Generated {} return observations", returns.len());
    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_ret =
        (returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64).sqrt();
    println!("  Mean return: {:.6}, Std: {:.6}", mean_ret, std_ret);

    // Step 2: Encoding methods
    println!("\n--- Encoding Methods ---");

    // Binary encoding: positive → true, else → false
    let binary = binary_encode(&returns);
    let n_pos = binary.iter().filter(|&&b| b).count();
    println!(
        "  Binary: {} positive, {} negative/zero",
        n_pos,
        returns.len() - n_pos
    );

    // Quantile encoding: 5 bins
    let num_bins = 5;
    let quantile = quantile_encode(&returns, num_bins);
    print!("  Quantile ({} bins): counts = [", num_bins);
    for b in 0..num_bins {
        let count = quantile.iter().filter(|&&q| q == b).count();
        if b > 0 {
            print!(", ");
        }
        print!("{}", count);
    }
    println!("]");

    // Sigma encoding: 4 sigma bands
    let num_bands = 4;
    let sigma = sigma_encode(&returns, num_bands);
    print!("  Sigma ({} bands): counts = [", num_bands);
    for b in 0..num_bands {
        let count = sigma.iter().filter(|&&s| s == b).count();
        if b > 0 {
            print!(", ");
        }
        print!("{}", count);
    }
    println!("]");

    // Step 3: Shannon entropy from a probability distribution
    println!("\n--- Shannon Entropy ---");
    // Build probability distribution from quantile bins
    let mut probs = vec![0.0_f64; num_bins];
    for &q in &quantile {
        probs[q] += 1.0;
    }
    let total = probs.iter().sum::<f64>();
    for p in probs.iter_mut() {
        *p /= total;
    }
    let h_shannon = shannon_entropy(&probs);
    let h_max = (num_bins as f64).log2();
    println!(
        "  H(quantile dist) = {:.4} bits (max possible = {:.4} bits)",
        h_shannon, h_max
    );
    println!("  Efficiency: {:.1}%", 100.0 * h_shannon / h_max);

    // Step 4: Plug-in entropy estimator
    println!("\n--- Plug-in Entropy ---");
    let h_plugin = plugin_entropy(&quantile, num_bins);
    println!("  H_plugin(quantile seq) = {:.4} bits", h_plugin);

    let h_plugin_sigma = plugin_entropy(&sigma, num_bands);
    println!("  H_plugin(sigma seq)    = {:.4} bits", h_plugin_sigma);

    // Step 5: Lempel-Ziv complexity
    println!("\n--- Lempel-Ziv Complexity ---");
    let lz = lempel_ziv_complexity(&binary);
    let normalized_lz = lz as f64 / (returns.len() as f64 / (returns.len() as f64).log2());
    println!("  LZ complexity: {} distinct patterns", lz);
    println!("  Normalized:    {:.4}", normalized_lz);

    // Compare with a pseudo-random binary sequence (Knuth multiplicative hash)
    let random_binary: Vec<bool> = (0..returns.len())
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 16) & 1 == 0)
        .collect();
    let lz_random = lempel_ziv_complexity(&random_binary);
    println!("  Pseudo-random baseline: {} patterns", lz_random);

    // Step 6: Kontoyiannis entropy estimator
    println!("\n--- Kontoyiannis Entropy ---");
    let window = 100;
    let h_konto = kontoyiannis_entropy(&quantile, window);
    println!("  H_konto(quantile, w={}) = {:.4} bits", window, h_konto);

    let h_konto_sigma = kontoyiannis_entropy(&sigma, window);
    println!(
        "  H_konto(sigma, w={})    = {:.4} bits",
        window, h_konto_sigma
    );

    // Step 7: Summary comparison
    println!("\n--- Summary ---");
    println!("  Encoding     | Shannon | Plug-in | Kontoyiannis");
    println!("  -------------|---------|---------|-------------");
    println!(
        "  Quantile(5)  | {:.4}  | {:.4}  | {:.4}",
        h_shannon, h_plugin, h_konto
    );

    // Build sigma probs for Shannon
    let mut sigma_probs = vec![0.0_f64; num_bands];
    for &s in &sigma {
        sigma_probs[s] += 1.0;
    }
    let total_s = sigma_probs.iter().sum::<f64>();
    for p in sigma_probs.iter_mut() {
        *p /= total_s;
    }
    let h_shannon_sigma = shannon_entropy(&sigma_probs);
    println!(
        "  Sigma(4)     | {:.4}  | {:.4}  | {:.4}",
        h_shannon_sigma, h_plugin_sigma, h_konto_sigma
    );

    println!("\nEntropy analysis complete.");
}

fn generate_returns(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let base = (i as f64 * 0.13).sin() * 0.02;
            let noise = ((i * 3) as f64 * 0.07).cos() * 0.015;
            let jump = if i % 47 == 0 { 0.05 } else { 0.0 };
            base + noise + jump
        })
        .collect()
}
