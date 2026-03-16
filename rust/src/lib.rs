//! # Koopman Operator Trading
//!
//! This crate implements Koopman operator methods for cryptocurrency trading.
//! It provides Dynamic Mode Decomposition (DMD) and Extended DMD for analyzing
//! and predicting price dynamics.
//!
//! ## Features
//!
//! - **DMD**: Standard Dynamic Mode Decomposition for linear approximation of nonlinear dynamics
//! - **EDMD**: Extended DMD with dictionary functions (polynomial, RBF)
//! - **Trading Signals**: Generate trading signals based on Koopman predictions
//! - **Regime Detection**: Detect market regime changes via spectral analysis
//! - **Bybit Integration**: Fetch real-time and historical data from Bybit exchange
//!
//! ## Example
//!
//! ```rust,no_run
//! use koopman_trading::{api::BybitClient, koopman::DMD};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 200).await?;
//!
//!     // Extract prices
//!     let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
//!
//!     // Perform DMD analysis
//!     let dmd = DMD::from_time_series(&prices, 10, 1.0)?;
//!
//!     // Get predictions
//!     let predictions = dmd.predict(5);
//!     println!("Next 5 predicted values: {:?}", predictions);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod features;
pub mod koopman;
pub mod trading;

// Re-exports for convenience
pub use api::BybitClient;
pub use data::types::{Candle, OrderBook, Trade};
pub use koopman::{DMD, EDMD};
pub use trading::{KoopmanTrader, Signal};
