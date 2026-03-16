//! Trading module for Koopman-based strategies
//!
//! Provides:
//! - Signal generation based on Koopman predictions
//! - Backtesting framework
//! - Performance metrics

mod signals;
mod backtest;
mod metrics;

pub use signals::{KoopmanTrader, Signal, RegimeLabel};
pub use backtest::{Backtester, BacktestResult};
pub use metrics::{PerformanceMetrics, calculate_sharpe, calculate_sortino, calculate_max_drawdown};
