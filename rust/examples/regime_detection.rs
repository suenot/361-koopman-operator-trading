//! Example: Market Regime Detection
//!
//! This example demonstrates how to use Koopman spectral analysis
//! to detect market regime changes in cryptocurrency data.
//!
//! Run with: cargo run --example regime_detection

use anyhow::Result;
use koopman_trading::{
    api::BybitClient,
    koopman::DMD,
    trading::RegimeLabel,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Market Regime Detection ===\n");

    // Fetch extended historical data
    let client = BybitClient::new();
    println!("Fetching BTCUSDT 4h data for regime analysis...\n");

    let candles = client.get_klines("BTCUSDT", "4h", 500).await?;
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    println!("Analyzing {} data points\n", prices.len());

    // Parameters
    let window_size = 50;
    let embed_dim = 10;
    let step_size = 10;

    // Rolling regime analysis
    println!("=== Rolling Regime Analysis ===\n");
    println!("{:>10} {:>15} {:>12} {:>12} {:>15}",
        "Window", "Price", "Stability", "Dom. Freq", "Regime");
    println!("{}", "-".repeat(70));

    let mut regimes = Vec::new();
    let mut regime_changes = Vec::new();
    let mut prev_regime = RegimeLabel::Unknown;

    for i in (window_size..prices.len()).step_by(step_size) {
        let window = &prices[i - window_size..i];

        match DMD::from_time_series(window, embed_dim, 1.0) {
            Ok(dmd) => {
                let stability = dmd.stability_ratio();

                // Find dominant eigenvalue
                let dominant = dmd.eigenvalues
                    .iter()
                    .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap());

                let (regime, dom_freq) = match dominant {
                    Some(ev) => {
                        let mag = ev.norm();
                        let phase = ev.im.atan2(ev.re);

                        let regime = if mag > 1.05 {
                            RegimeLabel::Trending
                        } else if mag < 0.95 {
                            RegimeLabel::MeanReverting
                        } else if phase.abs() > 0.1 {
                            RegimeLabel::Oscillatory
                        } else {
                            RegimeLabel::Unknown
                        };

                        let freq = if phase.abs() > 0.01 {
                            phase / (2.0 * std::f64::consts::PI)
                        } else {
                            0.0
                        };

                        (regime, freq)
                    }
                    None => (RegimeLabel::Unknown, 0.0),
                };

                // Check for regime change
                if regime != prev_regime && prev_regime != RegimeLabel::Unknown {
                    regime_changes.push((i, prev_regime, regime));
                }

                let current_price = prices[i - 1];
                println!(
                    "{:>10} {:>15.2} {:>12.2}% {:>12.4} {:>15?}",
                    i,
                    current_price,
                    stability * 100.0,
                    dom_freq,
                    regime
                );

                regimes.push(regime);
                prev_regime = regime;
            }
            Err(_) => {
                regimes.push(RegimeLabel::Unknown);
            }
        }
    }

    // Regime change summary
    println!("\n=== Regime Changes Detected ===\n");
    if regime_changes.is_empty() {
        println!("No significant regime changes detected in this period.");
    } else {
        for (idx, from, to) in &regime_changes {
            println!("  At index {}: {:?} -> {:?}", idx, from, to);
        }
    }

    // Regime statistics
    println!("\n=== Regime Distribution ===\n");
    let total = regimes.len();
    let trending = regimes.iter().filter(|&&r| r == RegimeLabel::Trending).count();
    let mean_rev = regimes.iter().filter(|&&r| r == RegimeLabel::MeanReverting).count();
    let oscillatory = regimes.iter().filter(|&&r| r == RegimeLabel::Oscillatory).count();
    let unknown = regimes.iter().filter(|&&r| r == RegimeLabel::Unknown).count();

    println!("Trending:      {:>5} ({:>5.1}%)", trending, 100.0 * trending as f64 / total as f64);
    println!("Mean-Reverting:{:>5} ({:>5.1}%)", mean_rev, 100.0 * mean_rev as f64 / total as f64);
    println!("Oscillatory:   {:>5} ({:>5.1}%)", oscillatory, 100.0 * oscillatory as f64 / total as f64);
    println!("Unknown:       {:>5} ({:>5.1}%)", unknown, 100.0 * unknown as f64 / total as f64);

    // Spectral distance analysis
    println!("\n=== Spectral Distance Analysis ===\n");

    let n_windows = 5;
    let window_len = prices.len() / n_windows;

    println!("Comparing {} time windows:\n", n_windows);

    // Calculate DMD for each window
    let mut dmds = Vec::new();
    for i in 0..n_windows {
        let start = i * window_len;
        let end = (i + 1) * window_len;
        let window = &prices[start..end];

        if let Ok(dmd) = DMD::from_time_series(window, embed_dim, 1.0) {
            dmds.push((i, dmd));
        }
    }

    // Calculate pairwise spectral distances
    println!("Spectral Distance Matrix:");
    print!("{:>8}", "");
    for i in 0..dmds.len() {
        print!("{:>8}", format!("W{}", i));
    }
    println!();

    for (i, dmd_i) in &dmds {
        print!("{:>8}", format!("W{}", i));
        for (j, dmd_j) in &dmds {
            let distance = koopman_trading::koopman::prediction::spectral_distance(
                &dmd_i.eigenvalues,
                &dmd_j.eigenvalues,
            );
            print!("{:>8.3}", distance);
        }
        println!();
    }

    // Identify most different windows
    let mut max_distance = 0.0;
    let mut max_pair = (0, 0);

    for (i, dmd_i) in &dmds {
        for (j, dmd_j) in &dmds {
            if i < j {
                let dist = koopman_trading::koopman::prediction::spectral_distance(
                    &dmd_i.eigenvalues,
                    &dmd_j.eigenvalues,
                );
                if dist > max_distance {
                    max_distance = dist;
                    max_pair = (*i, *j);
                }
            }
        }
    }

    println!("\nMost different windows: W{} and W{} (distance: {:.4})",
        max_pair.0, max_pair.1, max_distance);

    if max_distance > 0.3 {
        println!("-> Significant regime change occurred between these periods!");
    }

    // Trading implications
    println!("\n=== Trading Implications ===\n");

    let current_regime = regimes.last().unwrap_or(&RegimeLabel::Unknown);
    println!("Current regime: {:?}\n", current_regime);

    match current_regime {
        RegimeLabel::Trending => {
            println!("Recommended strategies:");
            println!("  - Trend-following (momentum)");
            println!("  - Breakout trading");
            println!("  - Avoid mean-reversion");
        }
        RegimeLabel::MeanReverting => {
            println!("Recommended strategies:");
            println!("  - Mean-reversion (buy dips, sell rallies)");
            println!("  - Range trading");
            println!("  - Avoid trend-following");
        }
        RegimeLabel::Oscillatory => {
            println!("Recommended strategies:");
            println!("  - Swing trading");
            println!("  - RSI-based strategies");
            println!("  - Time entries with cycle period");
        }
        _ => {
            println!("Recommended actions:");
            println!("  - Reduce position sizes");
            println!("  - Wait for clearer regime");
            println!("  - Use multiple confirmation signals");
        }
    }

    println!("\nRegime detection complete!");

    Ok(())
}
