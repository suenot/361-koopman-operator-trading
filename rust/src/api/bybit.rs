//! Bybit exchange API client
//!
//! Provides methods to fetch market data from Bybit:
//! - Kline (candlestick) data
//! - Order book data
//! - Recent trades

use crate::data::types::{Candle, OrderBook, OrderBookLevel, Trade, TradeSide};
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;

/// Bybit API base URL
const BASE_URL: &str = "https://api.bybit.com";

/// Bybit API client for fetching market data
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

/// Response wrapper from Bybit API
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from Bybit API
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Order book result from Bybit API
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64, // timestamp
}

/// Trade result from Bybit API
#[derive(Debug, Deserialize)]
struct TradeResult {
    #[allow(dead_code)]
    category: String,
    list: Vec<TradeItem>,
}

#[derive(Debug, Deserialize)]
struct TradeItem {
    #[serde(rename = "execId")]
    exec_id: String,
    symbol: String,
    price: String,
    size: String,
    side: String,
    time: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Create a testnet client
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Convert interval string to Bybit format
    fn parse_interval(interval: &str) -> Result<&'static str> {
        match interval.to_lowercase().as_str() {
            "1m" | "1min" | "1" => Ok("1"),
            "3m" | "3min" | "3" => Ok("3"),
            "5m" | "5min" | "5" => Ok("5"),
            "15m" | "15min" | "15" => Ok("15"),
            "30m" | "30min" | "30" => Ok("30"),
            "1h" | "60" | "60m" => Ok("60"),
            "2h" | "120" => Ok("120"),
            "4h" | "240" => Ok("240"),
            "6h" | "360" => Ok("360"),
            "12h" | "720" => Ok("720"),
            "1d" | "d" | "day" => Ok("D"),
            "1w" | "w" | "week" => Ok("W"),
            "1M" | "month" => Ok("M"),
            _ => Err(anyhow!("Invalid interval: {}", interval)),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1h", "4h", "1d")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of candles sorted by time (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let interval_str = Self::parse_interval(interval)?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval_str,
            limit.min(1000)
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Fetch order book data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Depth limit (1, 25, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            limit.min(200)
        );

        let response: BybitResponse<OrderBookResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let parse_levels = |levels: &[Vec<String>]| -> Vec<OrderBookLevel> {
            levels
                .iter()
                .filter_map(|level| {
                    if level.len() >= 2 {
                        Some(OrderBookLevel {
                            price: level[0].parse().ok()?,
                            quantity: level[1].parse().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        };

        Ok(OrderBook {
            symbol: response.result.s,
            timestamp: response.result.ts,
            bids: parse_levels(&response.result.b),
            asks: parse_levels(&response.result.a),
        })
    }

    /// Fetch recent trades
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Number of trades to fetch (max 1000)
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            limit.min(1000)
        );

        let response: BybitResponse<TradeResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let trades: Vec<Trade> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                Some(Trade {
                    id: item.exec_id.clone(),
                    symbol: item.symbol.clone(),
                    price: item.price.parse().ok()?,
                    quantity: item.size.parse().ok()?,
                    side: if item.side == "Buy" {
                        TradeSide::Buy
                    } else {
                        TradeSide::Sell
                    },
                    timestamp: item.time.parse().ok()?,
                })
            })
            .collect();

        Ok(trades)
    }

    /// Get available trading symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct SymbolResult {
            list: Vec<SymbolInfo>,
        }

        #[derive(Deserialize)]
        struct SymbolInfo {
            symbol: String,
        }

        let url = format!(
            "{}/v5/market/instruments-info?category=spot",
            self.base_url
        );

        let response: BybitResponse<SymbolResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        Ok(response.result.list.into_iter().map(|s| s.symbol).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_parsing() {
        assert_eq!(BybitClient::parse_interval("1h").unwrap(), "60");
        assert_eq!(BybitClient::parse_interval("4h").unwrap(), "240");
        assert_eq!(BybitClient::parse_interval("1d").unwrap(), "D");
        assert!(BybitClient::parse_interval("invalid").is_err());
    }
}
