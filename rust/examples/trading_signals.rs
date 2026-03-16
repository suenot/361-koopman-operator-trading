//! Example: Trading Signal Generation
//!
//! This example demonstrates how to generate trading signals
//! using Koopman operator methods.
//!
//! Run with: cargo run --example trading_signals

use anyhow::Result;
use koopman_trading::{
    api::BybitClient,
    koopman::DMD,
    trading::{KoopmanTrader, Signal, RegimeLabel},
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Koopman Trading Signal Generation ===\n");

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching market data...\n");

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in &symbols {
        println!("--- {} ---\n", symbol);

        let candles = client.get_klines(symbol, "1h", 200).await?;
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

        if prices.is_empty() {
            println!("No data available for {}\n", symbol);
            continue;
        }

        let current_price = *prices.last().unwrap();
        println!("Current price: ${:.2}", current_price);

        // Create trader
        let embed_dim = 10;
        let horizon = 5;
        let threshold = 0.002; // 0.2%

        match KoopmanTrader::from_prices(&prices, embed_dim, horizon, threshold) {
            Ok(trader) => {
                // Generate signal
                let signal = trader.generate_signal(&prices);

                println!("\nTrading Signal:");
                match &signal {
                    Signal::Long { strength, confidence } => {
                        println!("  Direction: LONG (BUY)");
                        println!("  Strength: {:.2}%", strength * 100.0);
                        println!("  Confidence: {:.2}%", confidence * 100.0);
                    }
                    Signal::Short { strength, confidence } => {
                        println!("  Direction: SHORT (SELL)");
                        println!("  Strength: {:.2}%", strength * 100.0);
                        println!("  Confidence: {:.2}%", confidence * 100.0);
                    }
                    Signal::Neutral => {
                        println!("  Direction: NEUTRAL (HOLD)");
                    }
                }

                // Classify regime
                let regime = trader.classify_regime(&prices);
                println!("\nMarket Regime: {:?}", regime);

                match regime {
                    RegimeLabel::Trending => {
                        println!("  -> Market is trending, momentum strategies may work");
                    }
                    RegimeLabel::MeanReverting => {
                        println!("  -> Market is mean-reverting, contrarian strategies may work");
                    }
                    RegimeLabel::Oscillatory => {
                        println!("  -> Market is oscillatory, timing entries is key");
                    }
                    RegimeLabel::Transition => {
                        println!("  -> Market is in transition, be cautious");
                    }
                    RegimeLabel::Unknown => {
                        println!("  -> Regime unclear, reduce position sizes");
                    }
                }

                // DMD predictions
                let predictions = trader.dmd.predict(horizon);
                println!("\nPrice Predictions:");
                for (i, pred) in predictions.iter().enumerate() {
                    let change = (pred - current_price) / current_price * 100.0;
                    println!("  t+{}: ${:.2} ({:+.2}%)", i + 1, pred, change);
                }

                // Model stability
                let stability = trader.dmd.stability_ratio();
                println!("\nModel Stability: {:.1}%", stability * 100.0);

                // Dominant frequencies
                let freqs = trader.dmd.frequencies();
                if !freqs.is_empty() {
                    let dominant_freq = freqs.iter()
                        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                        .unwrap();
                    if dominant_freq.abs() > 0.01 {
                        let period = 1.0 / dominant_freq.abs();
                        println!("Dominant cycle period: {:.1} hours", period);
                    }
                }
            }
            Err(e) => {
                println!("Failed to create trader: {}", e);
            }
        }

        println!();
    }

    // Multi-signal summary
    println!("=== Signal Summary ===\n");
    println!("Note: These signals are for educational purposes only.");
    println!("Always do your own research before trading.\n");

    // Regime change detection example
    println!("=== Regime Change Detection ===\n");

    let candles = client.get_klines("BTCUSDT", "1h", 300).await?;
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    if prices.len() >= 200 {
        let trader = KoopmanTrader::from_prices(&prices[..200], 10, 5, 0.002)?;

        // Compare two windows
        let window1 = &prices[50..100];
        let window2 = &prices[150..200];

        let regime_distance = trader.detect_regime_change(window1, window2);
        println!("Spectral distance between windows: {:.4}", regime_distance);

        if regime_distance > 0.3 {
            println!("-> Significant regime change detected!");
        } else if regime_distance > 0.1 {
            println!("-> Moderate regime shift");
        } else {
            println!("-> Market regime is stable");
        }
    }

    println!("\nSignal generation complete!");

    Ok(())
}
