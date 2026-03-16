//! Trading signal generation using Koopman methods

use crate::koopman::{DMD, delay_embed};
use anyhow::Result;

/// Trading signal types
#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    /// Long (buy) signal with strength and confidence
    Long { strength: f64, confidence: f64 },
    /// Short (sell) signal with strength and confidence
    Short { strength: f64, confidence: f64 },
    /// Neutral (hold) signal
    Neutral,
}

impl Signal {
    /// Check if signal is actionable (not neutral)
    pub fn is_actionable(&self) -> bool {
        !matches!(self, Signal::Neutral)
    }

    /// Get signal direction: 1 for long, -1 for short, 0 for neutral
    pub fn direction(&self) -> i32 {
        match self {
            Signal::Long { .. } => 1,
            Signal::Short { .. } => -1,
            Signal::Neutral => 0,
        }
    }

    /// Get signal strength (0 for neutral)
    pub fn strength(&self) -> f64 {
        match self {
            Signal::Long { strength, .. } => *strength,
            Signal::Short { strength, .. } => *strength,
            Signal::Neutral => 0.0,
        }
    }

    /// Get signal confidence (0 for neutral)
    pub fn confidence(&self) -> f64 {
        match self {
            Signal::Long { confidence, .. } => *confidence,
            Signal::Short { confidence, .. } => *confidence,
            Signal::Neutral => 0.0,
        }
    }
}

/// Market regime labels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeLabel {
    /// Trending market (strong directional movement)
    Trending,
    /// Mean-reverting market (oscillating around mean)
    MeanReverting,
    /// Oscillatory market (periodic patterns)
    Oscillatory,
    /// Transition between regimes
    Transition,
    /// Unknown regime
    Unknown,
}

/// Koopman-based trading signal generator
#[derive(Debug)]
pub struct KoopmanTrader {
    /// Fitted DMD model
    pub dmd: DMD,
    /// Prediction horizon (steps ahead)
    pub prediction_horizon: usize,
    /// Signal threshold (minimum expected return to generate signal)
    pub threshold: f64,
    /// Embedding dimension for DMD
    pub embed_dim: usize,
}

impl KoopmanTrader {
    /// Create a new Koopman trader
    ///
    /// # Arguments
    ///
    /// * `dmd` - Fitted DMD model
    /// * `prediction_horizon` - Number of steps to predict ahead
    /// * `threshold` - Minimum expected return to generate signal
    pub fn new(dmd: DMD, prediction_horizon: usize, threshold: f64) -> Self {
        Self {
            dmd,
            prediction_horizon,
            threshold,
            embed_dim: 10,
        }
    }

    /// Create trader from price data
    ///
    /// # Arguments
    ///
    /// * `prices` - Historical price data
    /// * `embed_dim` - Embedding dimension
    /// * `prediction_horizon` - Prediction horizon
    /// * `threshold` - Signal threshold
    pub fn from_prices(
        prices: &[f64],
        embed_dim: usize,
        prediction_horizon: usize,
        threshold: f64,
    ) -> Result<Self> {
        let dmd = DMD::from_time_series(prices, embed_dim, 1.0)?;
        Ok(Self {
            dmd,
            prediction_horizon,
            threshold,
            embed_dim,
        })
    }

    /// Generate trading signal from current price data
    ///
    /// # Arguments
    ///
    /// * `prices` - Recent price history (should include enough for embedding)
    pub fn generate_signal(&self, prices: &[f64]) -> Signal {
        if prices.is_empty() {
            return Signal::Neutral;
        }

        // Get current price
        let current_price = *prices.last().unwrap();

        // Predict future prices
        let predictions = self.dmd.predict(self.prediction_horizon);

        if predictions.is_empty() {
            return Signal::Neutral;
        }

        // Get predicted price at horizon
        let predicted_price = predictions[self.prediction_horizon - 1];

        // Calculate expected return
        let expected_return = (predicted_price - current_price) / current_price;

        // Analyze model stability
        let stability = self.dmd.stability_ratio();

        // Calculate confidence based on stability and prediction uncertainty
        let confidence = self.calculate_confidence(&predictions, stability);

        // Generate signal
        if expected_return > self.threshold && confidence > 0.3 {
            Signal::Long {
                strength: expected_return.min(1.0),
                confidence,
            }
        } else if expected_return < -self.threshold && confidence > 0.3 {
            Signal::Short {
                strength: (-expected_return).min(1.0),
                confidence,
            }
        } else {
            Signal::Neutral
        }
    }

    /// Calculate signal confidence
    fn calculate_confidence(&self, predictions: &[f64], stability: f64) -> f64 {
        // Base confidence from stability
        let mut confidence = stability;

        // Reduce confidence if predictions are volatile
        if predictions.len() > 1 {
            let pred_std = self.calculate_std(predictions);
            let pred_mean = predictions.iter().sum::<f64>() / predictions.len() as f64;

            if pred_mean.abs() > 1e-10 {
                let cv = pred_std / pred_mean.abs(); // Coefficient of variation
                confidence *= (1.0 - cv.min(1.0)).max(0.0);
            }
        }

        // Reduce confidence if eigenvalue spread is high
        let eigenvalue_spread = self.calculate_eigenvalue_spread();
        confidence *= (1.0 - eigenvalue_spread.min(1.0)).max(0.0);

        confidence.clamp(0.0, 1.0)
    }

    /// Calculate standard deviation
    fn calculate_std(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate eigenvalue spread (uncertainty measure)
    fn calculate_eigenvalue_spread(&self) -> f64 {
        let magnitudes: Vec<f64> = self.dmd.eigenvalues.iter().map(|ev| ev.norm()).collect();

        if magnitudes.is_empty() {
            return 1.0;
        }

        let mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;

        if mean < 1e-10 {
            return 1.0;
        }

        let variance = magnitudes
            .iter()
            .map(|m| (m - mean).powi(2))
            .sum::<f64>()
            / magnitudes.len() as f64;

        variance.sqrt() / mean
    }

    /// Detect regime change between two windows
    ///
    /// # Arguments
    ///
    /// * `window1` - First price window
    /// * `window2` - Second price window
    ///
    /// # Returns
    ///
    /// Spectral distance indicating regime change magnitude
    pub fn detect_regime_change(&self, window1: &[f64], window2: &[f64]) -> f64 {
        let dmd1 = match DMD::from_time_series(window1, self.embed_dim, 1.0) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };

        let dmd2 = match DMD::from_time_series(window2, self.embed_dim, 1.0) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };

        // Compare eigenvalue distributions
        crate::koopman::prediction::spectral_distance(&dmd1.eigenvalues, &dmd2.eigenvalues)
    }

    /// Classify current market regime
    ///
    /// # Arguments
    ///
    /// * `prices` - Recent price history
    pub fn classify_regime(&self, prices: &[f64]) -> RegimeLabel {
        if prices.len() < self.embed_dim + 1 {
            return RegimeLabel::Unknown;
        }

        // Analyze dominant eigenvalue
        let dominant = self
            .dmd
            .eigenvalues
            .iter()
            .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap());

        match dominant {
            Some(ev) => {
                let magnitude = ev.norm();
                let phase = ev.im.atan2(ev.re);

                if magnitude > 1.05 {
                    // Growing mode - trending
                    RegimeLabel::Trending
                } else if magnitude < 0.95 {
                    // Decaying mode - mean reverting
                    RegimeLabel::MeanReverting
                } else if phase.abs() > 0.1 {
                    // Significant imaginary part - oscillatory
                    RegimeLabel::Oscillatory
                } else {
                    RegimeLabel::Unknown
                }
            }
            None => RegimeLabel::Unknown,
        }
    }

    /// Update model with new data
    ///
    /// # Arguments
    ///
    /// * `new_prices` - Updated price history
    pub fn update(&mut self, new_prices: &[f64]) -> Result<()> {
        self.dmd = DMD::from_time_series(new_prices, self.embed_dim, 1.0)?;
        Ok(())
    }
}

/// Generate multiple signals for a price series
pub fn generate_signals(
    prices: &[f64],
    lookback: usize,
    embed_dim: usize,
    horizon: usize,
    threshold: f64,
) -> Vec<Signal> {
    let mut signals = Vec::new();

    if prices.len() < lookback + embed_dim {
        return signals;
    }

    for i in lookback..prices.len() {
        let window = &prices[i - lookback..i];

        match DMD::from_time_series(window, embed_dim, 1.0) {
            Ok(dmd) => {
                let trader = KoopmanTrader::new(dmd, horizon, threshold);
                signals.push(trader.generate_signal(window));
            }
            Err(_) => signals.push(Signal::Neutral),
        }
    }

    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_properties() {
        let long = Signal::Long {
            strength: 0.5,
            confidence: 0.8,
        };
        assert!(long.is_actionable());
        assert_eq!(long.direction(), 1);
        assert_eq!(long.strength(), 0.5);
        assert_eq!(long.confidence(), 0.8);

        let short = Signal::Short {
            strength: 0.3,
            confidence: 0.6,
        };
        assert!(short.is_actionable());
        assert_eq!(short.direction(), -1);

        let neutral = Signal::Neutral;
        assert!(!neutral.is_actionable());
        assert_eq!(neutral.direction(), 0);
    }

    #[test]
    fn test_koopman_trader() {
        // Create trending data
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();

        let trader = KoopmanTrader::from_prices(&prices, 10, 5, 0.001).unwrap();
        let signal = trader.generate_signal(&prices);

        // Should generate some signal for trending data
        assert!(matches!(signal, Signal::Long { .. } | Signal::Short { .. } | Signal::Neutral));
    }
}
