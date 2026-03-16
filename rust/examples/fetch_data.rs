//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch market data for Koopman analysis.
//!
//! Run with: cargo run --example fetch_data

use anyhow::Result;
use koopman_trading::api::BybitClient;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Bybit Data Fetching Example ===\n");

    let client = BybitClient::new();

    // Fetch BTCUSDT 1-hour candles
    println!("Fetching BTCUSDT 1h candles...");
    let candles = client.get_klines("BTCUSDT", "1h", 100).await?;
    println!("Fetched {} candles\n", candles.len());

    // Display last 10 candles
    println!("Last 10 candles:");
    println!("{:>20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(95));

    for candle in candles.iter().rev().take(10) {
        println!(
            "{:>20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    // Calculate some basic statistics
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let volatility = variance.sqrt() * (24.0 * 365.0_f64).sqrt(); // Annualized

    println!("\n=== Basic Statistics ===");
    println!("Mean hourly return: {:.4}%", mean_return * 100.0);
    println!("Annualized volatility: {:.2}%", volatility * 100.0);
    println!("Current price: ${:.2}", prices.last().unwrap());

    // Fetch order book
    println!("\n=== Order Book ===");
    let orderbook = client.get_orderbook("BTCUSDT", 10).await?;

    println!("\nBest Bid: ${:.2}", orderbook.best_bid().unwrap_or(0.0));
    println!("Best Ask: ${:.2}", orderbook.best_ask().unwrap_or(0.0));
    println!("Spread: ${:.2}", orderbook.spread().unwrap_or(0.0));
    println!("Order book imbalance: {:.4}", orderbook.imbalance(5));

    // Fetch recent trades
    println!("\n=== Recent Trades ===");
    let trades = client.get_recent_trades("BTCUSDT", 5).await?;

    println!("{:>12} {:>10} {:>8}",
        "Price", "Size", "Side");
    for trade in &trades {
        println!(
            "{:>12.2} {:>10.4} {:>8?}",
            trade.price,
            trade.quantity,
            trade.side
        );
    }

    // Also fetch some altcoins for multi-asset analysis
    println!("\n=== Multi-Asset Data ===");
    let symbols = ["ETHUSDT", "SOLUSDT", "XRPUSDT"];

    for symbol in &symbols {
        let data = client.get_klines(symbol, "1h", 10).await?;
        if let Some(last) = data.last() {
            println!("{}: ${:.4}", symbol, last.close);
        }
    }

    println!("\nData fetching complete!");

    Ok(())
}
