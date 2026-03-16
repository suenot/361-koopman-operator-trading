//! Example: Dynamic Mode Decomposition Analysis
//!
//! This example demonstrates how to perform DMD analysis on
//! cryptocurrency price data to extract coherent modes.
//!
//! Run with: cargo run --example dmd_analysis

use anyhow::Result;
use koopman_trading::{
    api::BybitClient,
    koopman::{DMD, EDMD, PolynomialDictionary, delay_embed},
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Dynamic Mode Decomposition Analysis ===\n");

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching BTCUSDT 1h data...");
    let candles = client.get_klines("BTCUSDT", "1h", 200).await?;
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    println!("Fetched {} price points\n", prices.len());

    // Perform standard DMD
    println!("=== Standard DMD ===\n");
    let embed_dim = 10;
    let dmd = DMD::from_time_series(&prices, embed_dim, 1.0)?;

    println!("Number of modes: {}", dmd.eigenvalues.len());
    println!("State dimension: {}", dmd.state_dim);

    // Analyze eigenvalues
    println!("\nEigenvalue Analysis:");
    println!("{:>5} {:>12} {:>12} {:>12} {:>10}",
        "Mode", "Magnitude", "Frequency", "Growth", "Stable?");
    println!("{}", "-".repeat(55));

    let mut mode_info: Vec<_> = dmd.eigenvalues
        .iter()
        .enumerate()
        .map(|(i, ev)| {
            let mag = ev.norm();
            let freq = ev.im.atan2(ev.re) / (2.0 * std::f64::consts::PI);
            let growth = mag.ln();
            let stable = mag < 1.0;
            (i, mag, freq, growth, stable)
        })
        .collect();

    // Sort by magnitude (descending)
    mode_info.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, mag, freq, growth, stable) in mode_info.iter().take(10) {
        println!(
            "{:>5} {:>12.6} {:>12.6} {:>12.6} {:>10}",
            i, mag, freq, growth,
            if *stable { "Yes" } else { "No" }
        );
    }

    // Stability analysis
    let stability_ratio = dmd.stability_ratio();
    println!("\nStability ratio: {:.2}% of modes are stable", stability_ratio * 100.0);

    // Mode energies
    let energies = dmd.mode_energies();
    let total_energy: f64 = energies.iter().sum();
    println!("\nMode energy distribution:");
    let mut sorted_energies: Vec<_> = energies.iter().enumerate().collect();
    sorted_energies.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let mut cumulative = 0.0;
    for (i, (idx, &energy)) in sorted_energies.iter().take(5).enumerate() {
        let pct = energy / total_energy * 100.0;
        cumulative += pct;
        println!("  Mode {}: {:.2}% (cumulative: {:.2}%)", idx, pct, cumulative);
    }

    // Prediction
    println!("\n=== Predictions ===\n");
    let predictions = dmd.predict(10);
    let current_price = *prices.last().unwrap();

    println!("Current price: ${:.2}", current_price);
    println!("\nPredicted prices (next 10 periods):");
    for (i, pred) in predictions.iter().enumerate() {
        let change_pct = (pred - current_price) / current_price * 100.0;
        println!(
            "  t+{:2}: ${:>10.2} ({:+.2}%)",
            i + 1, pred, change_pct
        );
    }

    // Extended DMD with polynomial dictionary
    println!("\n=== Extended DMD (Polynomial) ===\n");

    let dict = PolynomialDictionary::new(2, embed_dim);
    println!("Dictionary dimension: {}", dict.dim());

    match EDMD::from_time_series(&prices, dict, embed_dim) {
        Ok(edmd) => {
            println!("EDMD fitted successfully");
            println!("Koopman matrix size: {:?}", edmd.koopman_matrix.dim());
            println!("Number of eigenvalues: {}", edmd.eigenvalues.len());

            // Stability
            let stable_count = edmd.eigenvalues.iter()
                .filter(|ev| ev.norm() < 1.0)
                .count();
            println!("Stable modes: {}/{}", stable_count, edmd.eigenvalues.len());
        }
        Err(e) => {
            println!("EDMD fitting failed: {}", e);
        }
    }

    // Analyze delay embedding
    println!("\n=== Delay Embedding Analysis ===\n");
    let embedded = delay_embed(&prices, embed_dim);
    println!("Embedded matrix shape: {:?}", embedded.dim());

    // Compute singular values to assess dimensionality
    println!("\nThis embedded data can be used for:");
    println!("  - Koopman operator approximation");
    println!("  - Attractor reconstruction (Takens' theorem)");
    println!("  - Nonlinear dynamics analysis");

    println!("\nAnalysis complete!");

    Ok(())
}
