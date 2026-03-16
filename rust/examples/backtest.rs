//! Example: Backtesting Koopman Trading Strategy
//!
//! This example demonstrates how to backtest a trading strategy
//! based on Koopman operator predictions.
//!
//! Run with: cargo run --example backtest

use anyhow::Result;
use koopman_trading::{
    api::BybitClient,
    trading::{Backtester, PerformanceMetrics},
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Koopman Trading Strategy Backtest ===\n");

    // Fetch historical data
    let client = BybitClient::new();
    println!("Fetching BTCUSDT 1h data for backtesting...\n");

    let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
    println!("Fetched {} candles for backtest\n", candles.len());

    // Backtest parameters
    let initial_capital = 10000.0;
    let lookback = 50;
    let embed_dim = 10;
    let horizon = 5;

    println!("Backtest Parameters:");
    println!("  Initial capital: ${:.2}", initial_capital);
    println!("  Lookback window: {} periods", lookback);
    println!("  Embedding dimension: {}", embed_dim);
    println!("  Prediction horizon: {} periods", horizon);
    println!();

    // Create backtester
    let backtester = Backtester::new(initial_capital)
        .with_position_size(0.95)
        .with_fee(0.001);

    println!("Running backtest...\n");

    // Run backtest
    let results = backtester.run(&candles, lookback, embed_dim, horizon)?;

    // Display results
    println!("=== Backtest Results ===\n");

    println!("Portfolio Performance:");
    println!("  Initial capital:  ${:>12.2}", initial_capital);
    println!("  Final value:      ${:>12.2}", results.final_value);
    println!("  Total return:     {:>12.2}%", results.total_return * 100.0);
    println!();

    println!("Risk Metrics:");
    println!("  Sharpe ratio:     {:>12.4}", results.sharpe_ratio);
    println!("  Sortino ratio:    {:>12.4}", results.sortino_ratio);
    println!("  Max drawdown:     {:>12.2}%", results.max_drawdown * 100.0);
    println!();

    println!("Trading Statistics:");
    println!("  Total trades:     {:>12}", results.total_trades);
    println!("  Win rate:         {:>12.1}%", results.win_rate * 100.0);
    println!("  Avg trade return: {:>12.4}%", results.avg_trade_return * 100.0);
    println!();

    // Calculate additional metrics from trade returns
    let trade_returns: Vec<f64> = results.trades.iter().map(|t| t.return_pct).collect();
    let metrics = PerformanceMetrics::calculate(
        &trade_returns,
        &results.equity_curve,
        24.0 * 365.0, // Hourly data, annualize
    );

    println!("Extended Metrics:");
    println!("  Annualized return: {:>11.2}%", metrics.annualized_return * 100.0);
    println!("  Volatility:        {:>11.2}%", metrics.volatility * 100.0);
    println!("  Calmar ratio:      {:>11.4}", metrics.calmar_ratio);
    println!("  Profit factor:     {:>11.4}", metrics.profit_factor);
    println!("  Max consec. wins:  {:>11}", metrics.max_consecutive_wins);
    println!("  Max consec. losses:{:>11}", metrics.max_consecutive_losses);
    println!();

    // Trade analysis
    if !results.trades.is_empty() {
        println!("=== Trade Analysis ===\n");

        // Winning vs losing trades
        let winners: Vec<_> = results.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losers: Vec<_> = results.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let avg_win = if !winners.is_empty() {
            winners.iter().map(|t| t.pnl).sum::<f64>() / winners.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losers.is_empty() {
            losers.iter().map(|t| t.pnl).sum::<f64>() / losers.len() as f64
        } else {
            0.0
        };

        println!("Win/Loss Analysis:");
        println!("  Winning trades:    {:>8}", winners.len());
        println!("  Losing trades:     {:>8}", losers.len());
        println!("  Average win:       ${:>7.2}", avg_win);
        println!("  Average loss:      ${:>7.2}", avg_loss);

        if avg_loss.abs() > 0.0 {
            println!("  Win/Loss ratio:    {:>8.2}", avg_win / avg_loss.abs());
        }
        println!();

        // Best and worst trades
        let best_trade = results.trades.iter().max_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap());
        let worst_trade = results.trades.iter().min_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap());

        if let Some(trade) = best_trade {
            println!("Best trade:  ${:>8.2} ({:>+.2}%)", trade.pnl, trade.return_pct * 100.0);
        }
        if let Some(trade) = worst_trade {
            println!("Worst trade: ${:>8.2} ({:>+.2}%)", trade.pnl, trade.return_pct * 100.0);
        }
        println!();

        // Recent trades
        println!("Last 5 trades:");
        println!("{:>12} {:>12} {:>12} {:>10} {:>12}",
            "Entry", "Exit", "Direction", "PnL", "Return");
        println!("{}", "-".repeat(60));

        for trade in results.trades.iter().rev().take(5) {
            let direction = if trade.direction > 0 { "LONG" } else { "SHORT" };
            println!(
                "${:>11.2} ${:>11.2} {:>12} ${:>9.2} {:>11.2}%",
                trade.entry_price,
                trade.exit_price,
                direction,
                trade.pnl,
                trade.return_pct * 100.0
            );
        }
    }

    // Equity curve summary
    println!("\n=== Equity Curve Summary ===\n");

    let equity = &results.equity_curve;
    if equity.len() >= 10 {
        let step = equity.len() / 10;
        println!("Equity progression:");
        for i in (0..equity.len()).step_by(step) {
            let pct_time = 100.0 * i as f64 / equity.len() as f64;
            let pct_return = (equity[i] - initial_capital) / initial_capital * 100.0;
            println!("  {:>5.1}% of time: ${:>10.2} ({:>+.2}%)",
                pct_time, equity[i], pct_return);
        }
        println!("  100.0% of time: ${:>10.2} ({:>+.2}%)",
            equity.last().unwrap(),
            results.total_return * 100.0);
    }

    // Compare with buy-and-hold
    println!("\n=== Strategy vs Buy-and-Hold ===\n");

    let first_price = candles.first().map(|c| c.close).unwrap_or(0.0);
    let last_price = candles.last().map(|c| c.close).unwrap_or(0.0);
    let bnh_return = (last_price - first_price) / first_price;

    println!("Strategy return:     {:>+.2}%", results.total_return * 100.0);
    println!("Buy-and-hold return: {:>+.2}%", bnh_return * 100.0);
    println!("Excess return:       {:>+.2}%", (results.total_return - bnh_return) * 100.0);

    if results.total_return > bnh_return {
        println!("\n-> Strategy outperformed buy-and-hold!");
    } else {
        println!("\n-> Buy-and-hold outperformed the strategy.");
    }

    println!("\nBacktest complete!");
    println!("\nNote: Past performance does not guarantee future results.");
    println!("This is for educational purposes only.");

    Ok(())
}
