//! Financial observable functions for Koopman analysis
//!
//! Observable functions transform raw price data into features that
//! better capture market dynamics for Koopman operator analysis.

use ndarray::Array1;

/// Financial observables for Koopman analysis
///
/// Computes various technical indicators and features from price data
/// to create a richer state representation for DMD/EDMD.
#[derive(Debug, Clone)]
pub struct FinancialObservables {
    /// Lookback period for indicators
    pub lookback: usize,
    /// Include momentum features
    pub include_momentum: bool,
    /// Include volatility features
    pub include_volatility: bool,
    /// Include moving averages
    pub include_ma: bool,
}

impl Default for FinancialObservables {
    fn default() -> Self {
        Self {
            lookback: 20,
            include_momentum: true,
            include_volatility: true,
            include_ma: true,
        }
    }
}

impl FinancialObservables {
    /// Create new observable calculator
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            ..Default::default()
        }
    }

    /// Compute observables from price series
    ///
    /// # Arguments
    ///
    /// * `prices` - Price series (must have at least `lookback` elements)
    ///
    /// # Returns
    ///
    /// Array of observable values
    pub fn compute(&self, prices: &[f64]) -> Array1<f64> {
        let n = prices.len();
        if n < self.lookback {
            return Array1::zeros(0);
        }

        let mut observables = Vec::new();

        // Current price (normalized)
        let current_price = prices[n - 1];
        let price_mean = prices.iter().sum::<f64>() / n as f64;
        observables.push(current_price / price_mean);

        // Log returns at various lags
        for lag in [1, 2, 5, 10].iter() {
            if n > *lag {
                let log_ret = (prices[n - 1] / prices[n - 1 - lag]).ln();
                observables.push(log_ret);
            }
        }

        // Moving averages
        if self.include_ma {
            for window in [5, 10, 20].iter() {
                if n >= *window {
                    let ma: f64 = prices[n - window..].iter().sum::<f64>() / *window as f64;
                    observables.push(current_price / ma - 1.0); // Deviation from MA
                }
            }
        }

        // Volatility (realized)
        if self.include_volatility {
            let returns: Vec<f64> = prices
                .windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();

            for window in [5, 10, 20].iter() {
                if returns.len() >= *window {
                    let recent = &returns[returns.len() - window..];
                    let mean = recent.iter().sum::<f64>() / *window as f64;
                    let vol = (recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / *window as f64)
                        .sqrt();
                    observables.push(vol);
                }
            }
        }

        // Momentum
        if self.include_momentum {
            for period in [5, 10, 20].iter() {
                if n > *period {
                    let momentum = prices[n - 1] / prices[n - period] - 1.0;
                    observables.push(momentum);
                }
            }

            // Rate of change
            if n > 10 {
                let roc = (prices[n - 1] - prices[n - 10]) / prices[n - 10];
                observables.push(roc);
            }
        }

        // RSI-like feature
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.len() >= 14 {
            let recent = &returns[returns.len() - 14..];
            let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
            let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
            let rsi = if losses > 0.0 {
                gains / (gains + losses)
            } else {
                1.0
            };
            observables.push(rsi);
        }

        // Bollinger Band position
        if n >= 20 {
            let ma20: f64 = prices[n - 20..].iter().sum::<f64>() / 20.0;
            let std20 = (prices[n - 20..]
                .iter()
                .map(|p| (p - ma20).powi(2))
                .sum::<f64>()
                / 20.0)
                .sqrt();

            if std20 > 0.0 {
                let bb_position = (current_price - ma20) / (2.0 * std20);
                observables.push(bb_position);
            }
        }

        Array1::from_vec(observables)
    }

    /// Get expected dimension of observables
    pub fn dim(&self) -> usize {
        let mut dim = 1; // Current price

        dim += 4; // Log returns

        if self.include_ma {
            dim += 3;
        }

        if self.include_volatility {
            dim += 3;
        }

        if self.include_momentum {
            dim += 4;
        }

        dim += 1; // RSI
        dim += 1; // Bollinger

        dim
    }
}

/// Compute simple returns
pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Compute log returns
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

/// Compute simple moving average
pub fn sma(prices: &[f64], window: usize) -> Vec<f64> {
    if prices.len() < window {
        return Vec::new();
    }

    prices
        .windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

/// Compute exponential moving average
pub fn ema(prices: &[f64], window: usize) -> Vec<f64> {
    if prices.is_empty() || window == 0 {
        return Vec::new();
    }

    let alpha = 2.0 / (window as f64 + 1.0);
    let mut result = Vec::with_capacity(prices.len());
    result.push(prices[0]);

    for i in 1..prices.len() {
        let prev = result[i - 1];
        result.push(alpha * prices[i] + (1.0 - alpha) * prev);
    }

    result
}

/// Compute realized volatility
pub fn realized_volatility(prices: &[f64], window: usize) -> Vec<f64> {
    let returns = compute_log_returns(prices);

    if returns.len() < window {
        return Vec::new();
    }

    returns
        .windows(window)
        .map(|w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            (w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64).sqrt()
        })
        .collect()
}

/// Compute RSI (Relative Strength Index)
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let returns = compute_returns(prices);

    if returns.len() < period {
        return Vec::new();
    }

    let mut result = Vec::new();

    for i in period..=returns.len() {
        let window = &returns[i - period..i];
        let gains: f64 = window.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = window.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();

        let rsi_val = if losses > 0.0 {
            100.0 * gains / (gains + losses)
        } else {
            100.0
        };
        result.push(rsi_val);
    }

    result
}

/// Compute MACD (Moving Average Convergence Divergence)
pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = ema(prices, fast);
    let ema_slow = ema(prices, slow);

    if ema_fast.len() != ema_slow.len() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = ema(&macd_line, signal);

    let histogram: Vec<f64> = if macd_line.len() == signal_line.len() {
        macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect()
    } else {
        Vec::new()
    };

    (macd_line, signal_line, histogram)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_financial_observables() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let obs = FinancialObservables::default();
        let features = obs.compute(&prices);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = sma(&prices, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
        assert!((ma[1] - 3.0).abs() < 1e-10);
        assert!((ma[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&prices, 3);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi_values = rsi(&prices, 14);
        assert!(!rsi_values.is_empty());
        // Trending up should have high RSI
        assert!(rsi_values.last().unwrap() > &50.0);
    }
}
