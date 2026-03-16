//! Backtesting framework for Koopman trading strategies

use crate::data::Candle;
use crate::koopman::DMD;
use crate::trading::{KoopmanTrader, Signal};
use anyhow::Result;

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Final portfolio value
    pub final_value: f64,
    /// Total return (final / initial - 1)
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Win rate (profitable trades / total trades)
    pub win_rate: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Equity curve (portfolio values over time)
    pub equity_curve: Vec<f64>,
    /// Trade log
    pub trades: Vec<TradeRecord>,
}

/// Individual trade record
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Entry timestamp
    pub entry_time: u64,
    /// Exit timestamp
    pub exit_time: u64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// Trade direction (1 for long, -1 for short)
    pub direction: i32,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

/// Backtester for Koopman trading strategies
#[derive(Debug)]
pub struct Backtester {
    /// Initial capital
    pub initial_capital: f64,
    /// Position sizing (fraction of capital per trade)
    pub position_size: f64,
    /// Trading fee (as fraction, e.g., 0.001 for 0.1%)
    pub trading_fee: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

impl Backtester {
    /// Create a new backtester
    ///
    /// # Arguments
    ///
    /// * `initial_capital` - Starting capital
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            position_size: 0.95, // Use 95% of capital per trade
            trading_fee: 0.001,  // 0.1% fee
            slippage: 0.0005,    // 0.05% slippage
        }
    }

    /// Set position sizing
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size.clamp(0.01, 1.0);
        self
    }

    /// Set trading fee
    pub fn with_fee(mut self, fee: f64) -> Self {
        self.trading_fee = fee.max(0.0);
        self
    }

    /// Run backtest on candle data
    ///
    /// # Arguments
    ///
    /// * `candles` - Historical candle data
    /// * `lookback` - Lookback window for DMD fitting
    /// * `embed_dim` - Embedding dimension
    /// * `horizon` - Prediction horizon
    pub fn run(
        &self,
        candles: &[Candle],
        lookback: usize,
        embed_dim: usize,
        horizon: usize,
    ) -> Result<BacktestResult> {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

        if prices.len() < lookback + embed_dim + horizon {
            return Err(anyhow::anyhow!("Not enough data for backtest"));
        }

        let mut capital = self.initial_capital;
        let mut equity_curve = vec![capital];
        let mut trades = Vec::new();
        let mut current_position: Option<Position> = None;

        let threshold = 0.002; // 0.2% threshold

        for i in lookback..prices.len() {
            let window = &prices[i - lookback..i];
            let current_price = prices[i];
            let timestamp = candles[i].timestamp;

            // Generate signal
            let signal = match DMD::from_time_series(window, embed_dim, 1.0) {
                Ok(dmd) => {
                    let trader = KoopmanTrader::new(dmd, horizon, threshold);
                    trader.generate_signal(window)
                }
                Err(_) => Signal::Neutral,
            };

            // Check if we need to close position
            if let Some(ref pos) = current_position {
                let should_close = match (&signal, pos.direction) {
                    (Signal::Short { .. }, 1) => true,  // Close long on short signal
                    (Signal::Long { .. }, -1) => true,  // Close short on long signal
                    _ => false,
                };

                if should_close {
                    // Close position
                    let exit_price = self.apply_slippage(current_price, -pos.direction);
                    let pnl = self.calculate_pnl(pos, exit_price);

                    capital += pnl;

                    trades.push(TradeRecord {
                        entry_time: pos.entry_time,
                        exit_time: timestamp,
                        entry_price: pos.entry_price,
                        exit_price,
                        size: pos.size,
                        direction: pos.direction,
                        pnl,
                        return_pct: pnl / (pos.entry_price * pos.size),
                    });

                    current_position = None;
                }
            }

            // Open new position if no current position
            if current_position.is_none() && signal.is_actionable() {
                let direction = signal.direction();
                let size = (capital * self.position_size) / current_price;
                let entry_price = self.apply_slippage(current_price, direction);

                // Apply entry fee
                capital -= size * entry_price * self.trading_fee;

                current_position = Some(Position {
                    entry_price,
                    size,
                    direction,
                    entry_time: timestamp,
                });
            }

            // Update equity curve
            let position_value = current_position
                .as_ref()
                .map(|pos| {
                    let unrealized_pnl = (current_price - pos.entry_price) * pos.size * pos.direction as f64;
                    unrealized_pnl
                })
                .unwrap_or(0.0);

            equity_curve.push(capital + position_value);
        }

        // Close any remaining position
        if let Some(pos) = current_position {
            let exit_price = self.apply_slippage(*prices.last().unwrap(), -pos.direction);
            let pnl = self.calculate_pnl(&pos, exit_price);
            capital += pnl;

            trades.push(TradeRecord {
                entry_time: pos.entry_time,
                exit_time: candles.last().unwrap().timestamp,
                entry_price: pos.entry_price,
                exit_price,
                size: pos.size,
                direction: pos.direction,
                pnl,
                return_pct: pnl / (pos.entry_price * pos.size),
            });
        }

        // Calculate metrics
        let final_value = capital;
        let total_return = (final_value - self.initial_capital) / self.initial_capital;

        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let sharpe_ratio = calculate_sharpe(&returns, 0.0);
        let sortino_ratio = calculate_sortino(&returns, 0.0);
        let max_drawdown = calculate_max_drawdown(&equity_curve);

        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if trades.is_empty() {
            0.0
        } else {
            winning_trades as f64 / trades.len() as f64
        };

        let avg_trade_return = if trades.is_empty() {
            0.0
        } else {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / trades.len() as f64
        };

        Ok(BacktestResult {
            final_value,
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            total_trades: trades.len(),
            win_rate,
            avg_trade_return,
            equity_curve,
            trades,
        })
    }

    /// Apply slippage to price
    fn apply_slippage(&self, price: f64, direction: i32) -> f64 {
        price * (1.0 + self.slippage * direction as f64)
    }

    /// Calculate PnL for a position
    fn calculate_pnl(&self, pos: &Position, exit_price: f64) -> f64 {
        let gross_pnl = (exit_price - pos.entry_price) * pos.size * pos.direction as f64;
        let exit_fee = pos.size * exit_price * self.trading_fee;
        gross_pnl - exit_fee
    }
}

/// Internal position tracking
#[derive(Debug)]
struct Position {
    entry_price: f64,
    size: f64,
    direction: i32,
    entry_time: u64,
}

/// Calculate Sharpe ratio
pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean - risk_free_rate / 252.0; // Daily risk-free rate

    if returns.len() < 2 {
        return 0.0;
    }

    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    let std = variance.sqrt();

    if std < 1e-10 {
        return 0.0;
    }

    (excess_return / std) * (252.0_f64).sqrt() // Annualized
}

/// Calculate Sortino ratio
pub fn calculate_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean - risk_free_rate / 252.0;

    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    if downside_returns.is_empty() {
        return if excess_return > 0.0 { f64::INFINITY } else { 0.0 };
    }

    let downside_variance = downside_returns
        .iter()
        .map(|r| r.powi(2))
        .sum::<f64>()
        / downside_returns.len() as f64;
    let downside_std = downside_variance.sqrt();

    if downside_std < 1e-10 {
        return 0.0;
    }

    (excess_return / downside_std) * (252.0_f64).sqrt()
}

/// Calculate maximum drawdown
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];

    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, -0.005, 0.008, 0.012, -0.003];
        let sharpe = calculate_sharpe(&returns, 0.0);
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0, 90.0, 95.0];
        let dd = calculate_max_drawdown(&equity);
        // Max drawdown from 110 to 90 = 20/110 = 0.1818
        assert!((dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_backtester_creation() {
        let bt = Backtester::new(10000.0)
            .with_position_size(0.5)
            .with_fee(0.002);

        assert_eq!(bt.initial_capital, 10000.0);
        assert_eq!(bt.position_size, 0.5);
        assert_eq!(bt.trading_fee, 0.002);
    }
}
