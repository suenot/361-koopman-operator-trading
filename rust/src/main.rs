//! Koopman Trading CLI
//!
//! Command-line interface for Koopman operator trading analysis.

use anyhow::Result;
use clap::{Parser, Subcommand};
use koopman_trading::{
    api::BybitClient,
    koopman::DMD,
    trading::{Backtester, KoopmanTrader, Signal},
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "koopman_trading")]
#[command(about = "Koopman Operator Trading Analysis Tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Timeframe (e.g., 1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "200")]
        limit: usize,
    },

    /// Perform DMD analysis
    Analyze {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Timeframe
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Embedding dimension
        #[arg(short, long, default_value = "10")]
        embed_dim: usize,

        /// Number of candles
        #[arg(short, long, default_value = "200")]
        limit: usize,
    },

    /// Generate trading signals
    Signal {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Timeframe
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Prediction horizon (steps ahead)
        #[arg(short, long, default_value = "5")]
        horizon: usize,
    },

    /// Run backtest
    Backtest {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Timeframe
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Initial capital
        #[arg(short, long, default_value = "10000.0")]
        capital: f64,

        /// Number of candles for backtesting
        #[arg(short, long, default_value = "500")]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            limit,
        } => {
            info!("Fetching {} {} candles for {}", limit, interval, symbol);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, &interval, limit).await?;

            println!("Fetched {} candles", candles.len());
            println!("\nLast 5 candles:");
            for candle in candles.iter().rev().take(5) {
                println!(
                    "  {} | O: {:.2} H: {:.2} L: {:.2} C: {:.2} V: {:.2}",
                    candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume
                );
            }
        }

        Commands::Analyze {
            symbol,
            interval,
            embed_dim,
            limit,
        } => {
            info!("Analyzing {} with DMD", symbol);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, &interval, limit).await?;
            let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

            let dmd = DMD::from_time_series(&prices, embed_dim, 1.0)?;

            println!("\n=== DMD Analysis Results ===\n");
            println!("Number of modes: {}", dmd.eigenvalues.len());

            println!("\nTop 5 modes by magnitude:");
            let mut mode_info: Vec<_> = dmd
                .eigenvalues
                .iter()
                .enumerate()
                .map(|(i, ev)| {
                    let mag = (ev.re * ev.re + ev.im * ev.im).sqrt();
                    let freq = ev.im.atan2(ev.re) / (2.0 * std::f64::consts::PI);
                    let growth = mag.ln();
                    (i, mag, freq, growth)
                })
                .collect();
            mode_info.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, mag, freq, growth) in mode_info.iter().take(5) {
                println!(
                    "  Mode {}: |λ| = {:.4}, freq = {:.4}, growth = {:.4}",
                    i, mag, freq, growth
                );
            }

            // Stability analysis
            let stable_count = dmd.eigenvalues.iter().filter(|ev| {
                let mag = (ev.re * ev.re + ev.im * ev.im).sqrt();
                mag < 1.0
            }).count();
            let total = dmd.eigenvalues.len();

            println!("\nStability: {}/{} modes stable ({:.1}%)",
                stable_count, total, 100.0 * stable_count as f64 / total as f64);
        }

        Commands::Signal {
            symbol,
            interval,
            horizon,
        } => {
            info!("Generating signal for {}", symbol);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, &interval, 200).await?;
            let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

            let dmd = DMD::from_time_series(&prices, 10, 1.0)?;
            let trader = KoopmanTrader::new(dmd, horizon, 0.002);

            let signal = trader.generate_signal(&prices);

            println!("\n=== Trading Signal ===\n");
            println!("Symbol: {}", symbol);
            println!("Current price: ${:.2}", prices.last().unwrap());
            println!("Prediction horizon: {} periods", horizon);

            match signal {
                Signal::Long { strength, confidence } => {
                    println!("\nSignal: LONG (BUY)");
                    println!("Strength: {:.2}%", strength * 100.0);
                    println!("Confidence: {:.2}%", confidence * 100.0);
                }
                Signal::Short { strength, confidence } => {
                    println!("\nSignal: SHORT (SELL)");
                    println!("Strength: {:.2}%", strength * 100.0);
                    println!("Confidence: {:.2}%", confidence * 100.0);
                }
                Signal::Neutral => {
                    println!("\nSignal: NEUTRAL (HOLD)");
                }
            }
        }

        Commands::Backtest {
            symbol,
            interval,
            capital,
            limit,
        } => {
            info!("Running backtest for {}", symbol);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, &interval, limit).await?;

            let backtester = Backtester::new(capital);
            let results = backtester.run(&candles, 50, 10, 5)?;

            println!("\n=== Backtest Results ===\n");
            println!("Symbol: {}", symbol);
            println!("Period: {} candles", limit);
            println!("Initial capital: ${:.2}", capital);
            println!("\nPerformance:");
            println!("  Final value: ${:.2}", results.final_value);
            println!("  Total return: {:.2}%", results.total_return * 100.0);
            println!("  Sharpe ratio: {:.4}", results.sharpe_ratio);
            println!("  Max drawdown: {:.2}%", results.max_drawdown * 100.0);
            println!("  Total trades: {}", results.total_trades);
            println!("  Win rate: {:.1}%", results.win_rate * 100.0);
        }
    }

    Ok(())
}
