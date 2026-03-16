# Chapter 361: Koopman Operator Trading — Linearizing Nonlinear Market Dynamics

## Overview

Financial markets are inherently nonlinear dynamical systems. Traditional linear methods struggle to capture complex patterns, regime changes, and non-stationary behavior. The **Koopman operator** provides a revolutionary approach: it transforms nonlinear dynamics into a **linear** (but infinite-dimensional) system through observable functions.

This chapter explores how to leverage Koopman operator theory and its data-driven approximations (Dynamic Mode Decomposition, Extended DMD, Deep Koopman Networks) for trading applications including prediction, regime detection, and feature extraction.

## Trading Strategy

**Core Concept:** Use Koopman operator methods to:
1. **Linearize** complex market dynamics in a lifted feature space
2. **Extract coherent modes** (dynamic modes) that capture price evolution
3. **Predict future states** using the learned linear dynamics
4. **Detect regime changes** through spectral analysis of the Koopman operator

**Edge:** While others model raw price series with nonlinear methods, Koopman approaches find the intrinsic linear structure hidden in the data, providing more interpretable and stable predictions.

## Mathematical Foundation

### The Koopman Operator

For a discrete dynamical system:
```
x_{t+1} = F(x_t)
```

The Koopman operator K acts on **observables** (scalar functions of state):
```
(K g)(x) = g(F(x))
```

Key insight: K is **linear** even when F is nonlinear!

### Eigenfunction Decomposition

If φ is an eigenfunction of K with eigenvalue λ:
```
K φ = λ φ
```

Then:
```
φ(x_t) = λ^t φ(x_0)
```

This means **any observable** can be decomposed into modes with known temporal evolution.

### Dynamic Mode Decomposition (DMD)

DMD approximates the Koopman operator from data snapshots:

Given data matrices:
```
X = [x_1, x_2, ..., x_{m-1}]
Y = [x_2, x_3, ..., x_m]
```

Find linear operator A such that Y ≈ AX:
```
A = Y X^†  (pseudo-inverse)
```

DMD modes are eigenvectors of A, with eigenvalues giving growth rates and frequencies.

### Extended DMD (EDMD)

Lift data to higher-dimensional space using dictionary functions:
```
Ψ(x) = [ψ_1(x), ψ_2(x), ..., ψ_N(x)]^T
```

Then apply DMD to lifted data:
```
Ψ(Y) ≈ K̂ Ψ(X)
```

### Deep Koopman Networks

Use neural networks to learn optimal lifting functions:
```
Encoder: x → Ψ(x)
Koopman layer: Ψ(x_t) → K Ψ(x_t) ≈ Ψ(x_{t+1})
Decoder: Ψ(x) → x
```

Loss functions:
1. **Reconstruction loss**: ||x - Decoder(Encoder(x))||²
2. **Prediction loss**: ||x_{t+1} - Decoder(K · Encoder(x_t))||²
3. **Linearity loss**: ||Encoder(x_{t+1}) - K · Encoder(x_t)||²

## Technical Specification

### Module Structure

```
361_koopman_operator_trading/
├── README.md                 # Main chapter (English)
├── README.ru.md              # Russian translation
├── readme.simple.md          # Simple explanation (English)
├── readme.simple.ru.md       # Simple explanation (Russian)
├── README.specify.md         # Specification
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs            # Library root
        ├── main.rs           # CLI entry point
        ├── api/
        │   ├── mod.rs
        │   └── bybit.rs      # Bybit API client
        ├── koopman/
        │   ├── mod.rs
        │   ├── dmd.rs        # Dynamic Mode Decomposition
        │   ├── edmd.rs       # Extended DMD
        │   ├── kernels.rs    # Kernel functions for EDMD
        │   └── prediction.rs # Prediction utilities
        ├── features/
        │   ├── mod.rs
        │   ├── observables.rs # Observable functions
        │   └── lifting.rs     # Feature lifting
        ├── trading/
        │   ├── mod.rs
        │   ├── signals.rs    # Trading signal generation
        │   ├── backtest.rs   # Backtesting engine
        │   └── metrics.rs    # Performance metrics
        └── data/
            ├── mod.rs
            └── types.rs      # Data structures
```

### Core Algorithms

#### 1. Standard DMD

```rust
/// Dynamic Mode Decomposition
pub struct DMD {
    pub modes: Array2<Complex64>,    // DMD modes (eigenvectors)
    pub eigenvalues: Array1<Complex64>, // Koopman eigenvalues
    pub amplitudes: Array1<Complex64>,  // Mode amplitudes
    pub dt: f64,                      // Time step
}

impl DMD {
    pub fn fit(data: &Array2<f64>, dt: f64) -> Result<Self> {
        let (n, m) = data.dim();

        // Split into X and Y matrices
        let x = data.slice(s![.., ..m-1]).to_owned();
        let y = data.slice(s![.., 1..]).to_owned();

        // SVD of X
        let (u, s, vt) = svd(&x)?;

        // Rank truncation
        let r = optimal_rank(&s);
        let u_r = u.slice(s![.., ..r]);
        let s_r = Array::from_diag(&s.slice(s![..r]));
        let vt_r = vt.slice(s![..r, ..]);

        // Build reduced matrix
        let a_tilde = u_r.t().dot(&y).dot(&vt_r.t()).dot(&s_r.inv());

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = eig(&a_tilde)?;

        // Recover full DMD modes
        let modes = y.dot(&vt_r.t()).dot(&s_r.inv()).dot(&eigenvectors);

        // Compute amplitudes
        let amplitudes = lstsq(&modes, &data.column(0))?;

        Ok(Self {
            modes,
            eigenvalues,
            amplitudes,
            dt,
        })
    }

    pub fn predict(&self, steps: usize, x0: &Array1<f64>) -> Array2<f64> {
        let mut predictions = Array2::zeros((x0.len(), steps));

        for t in 0..steps {
            let time_dynamics: Array1<Complex64> = self.eigenvalues
                .iter()
                .zip(self.amplitudes.iter())
                .map(|(λ, b)| b * λ.powf(t as f64))
                .collect();

            let pred = self.modes.dot(&time_dynamics);
            predictions.column_mut(t).assign(&pred.mapv(|c| c.re));
        }

        predictions
    }

    pub fn continuous_eigenvalues(&self) -> Array1<Complex64> {
        self.eigenvalues.mapv(|λ| λ.ln() / self.dt)
    }

    pub fn frequencies(&self) -> Array1<f64> {
        self.continuous_eigenvalues().mapv(|ω| ω.im / (2.0 * PI))
    }

    pub fn growth_rates(&self) -> Array1<f64> {
        self.continuous_eigenvalues().mapv(|ω| ω.re)
    }
}
```

#### 2. Extended DMD with Dictionary

```rust
/// Dictionary functions for EDMD
pub trait Dictionary: Send + Sync {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64>;
    fn dim(&self) -> usize;
}

/// Polynomial dictionary
pub struct PolynomialDictionary {
    pub degree: usize,
    pub input_dim: usize,
}

impl Dictionary for PolynomialDictionary {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut features = vec![1.0]; // Constant term

        // Add monomials up to degree
        for d in 1..=self.degree {
            for combo in combinations_with_replacement(0..self.input_dim, d) {
                let term: f64 = combo.iter().map(|&i| x[i]).product();
                features.push(term);
            }
        }

        Array1::from_vec(features)
    }

    fn dim(&self) -> usize {
        binomial(self.input_dim + self.degree, self.degree)
    }
}

/// RBF (Radial Basis Function) dictionary
pub struct RBFDictionary {
    pub centers: Array2<f64>,
    pub sigma: f64,
}

impl Dictionary for RBFDictionary {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64> {
        let n_centers = self.centers.nrows();
        let mut features = Array1::zeros(n_centers + 1);
        features[0] = 1.0; // Constant term

        for (i, center) in self.centers.rows().into_iter().enumerate() {
            let diff = x - &center.to_owned();
            let dist_sq = diff.dot(&diff);
            features[i + 1] = (-dist_sq / (2.0 * self.sigma.powi(2))).exp();
        }

        features
    }

    fn dim(&self) -> usize {
        self.centers.nrows() + 1
    }
}

/// Extended Dynamic Mode Decomposition
pub struct EDMD<D: Dictionary> {
    pub dictionary: D,
    pub koopman_matrix: Array2<f64>,
    pub eigenvalues: Array1<Complex64>,
    pub eigenvectors: Array2<Complex64>,
}

impl<D: Dictionary> EDMD<D> {
    pub fn fit(data: &Array2<f64>, dictionary: D) -> Result<Self> {
        let (n, m) = data.dim();

        // Lift all data points
        let lifted_dim = dictionary.dim();
        let mut psi_x = Array2::zeros((lifted_dim, m - 1));
        let mut psi_y = Array2::zeros((lifted_dim, m - 1));

        for i in 0..m-1 {
            psi_x.column_mut(i).assign(&dictionary.lift(&data.column(i).to_owned()));
            psi_y.column_mut(i).assign(&dictionary.lift(&data.column(i + 1).to_owned()));
        }

        // Solve for Koopman matrix: Psi_Y = K * Psi_X
        let g = psi_x.dot(&psi_x.t());
        let a = psi_x.dot(&psi_y.t());
        let koopman_matrix = lstsq(&g, &a)?;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = eig(&koopman_matrix)?;

        Ok(Self {
            dictionary,
            koopman_matrix,
            eigenvalues,
            eigenvectors,
        })
    }

    pub fn predict(&self, x0: &Array1<f64>, steps: usize) -> Vec<Array1<f64>> {
        let mut predictions = Vec::with_capacity(steps);
        let mut psi = self.dictionary.lift(x0);

        for _ in 0..steps {
            psi = self.koopman_matrix.dot(&psi);
            // Extract observable (assuming first n components are original state)
            predictions.push(psi.slice(s![1..x0.len()+1]).to_owned());
        }

        predictions
    }
}
```

#### 3. Trading Signal Generation

```rust
/// Koopman-based trading signals
pub struct KoopmanTrader {
    pub dmd: DMD,
    pub prediction_horizon: usize,
    pub threshold: f64,
}

impl KoopmanTrader {
    pub fn new(dmd: DMD, prediction_horizon: usize, threshold: f64) -> Self {
        Self {
            dmd,
            prediction_horizon,
            threshold,
        }
    }

    pub fn generate_signal(&self, current_state: &Array1<f64>) -> Signal {
        // Predict future prices
        let predictions = self.dmd.predict(self.prediction_horizon, current_state);
        let current_price = current_state[0];
        let predicted_price = predictions[[0, self.prediction_horizon - 1]];

        // Calculate expected return
        let expected_return = (predicted_price - current_price) / current_price;

        // Analyze mode stability
        let stable_modes = self.dmd.growth_rates()
            .iter()
            .filter(|&&r| r < 0.0)
            .count();
        let stability_ratio = stable_modes as f64 / self.dmd.eigenvalues.len() as f64;

        // Generate signal with confidence
        let confidence = stability_ratio * (1.0 - self.prediction_uncertainty());

        if expected_return > self.threshold && confidence > 0.5 {
            Signal::Long {
                strength: expected_return.min(1.0),
                confidence,
            }
        } else if expected_return < -self.threshold && confidence > 0.5 {
            Signal::Short {
                strength: (-expected_return).min(1.0),
                confidence,
            }
        } else {
            Signal::Neutral
        }
    }

    fn prediction_uncertainty(&self) -> f64 {
        // Estimate uncertainty from eigenvalue spread
        let magnitudes: Vec<f64> = self.dmd.eigenvalues.iter()
            .map(|λ| λ.norm())
            .collect();
        let mean_mag = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        let variance = magnitudes.iter()
            .map(|m| (m - mean_mag).powi(2))
            .sum::<f64>() / magnitudes.len() as f64;
        variance.sqrt() / mean_mag
    }

    /// Detect regime changes via spectral analysis
    pub fn detect_regime_change(&self, window1: &Array2<f64>, window2: &Array2<f64>) -> f64 {
        let dmd1 = DMD::fit(window1, self.dmd.dt).unwrap();
        let dmd2 = DMD::fit(window2, self.dmd.dt).unwrap();

        // Compare eigenvalue distributions
        spectral_distance(&dmd1.eigenvalues, &dmd2.eigenvalues)
    }
}

/// Calculate spectral distance between two sets of eigenvalues
fn spectral_distance(λ1: &Array1<Complex64>, λ2: &Array1<Complex64>) -> f64 {
    // Use Wasserstein distance on eigenvalue magnitudes
    let mut mags1: Vec<f64> = λ1.iter().map(|λ| λ.norm()).collect();
    let mut mags2: Vec<f64> = λ2.iter().map(|λ| λ.norm()).collect();

    mags1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mags2.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Pad to same length
    let n = mags1.len().max(mags2.len());
    mags1.resize(n, 0.0);
    mags2.resize(n, 0.0);

    mags1.iter()
        .zip(mags2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>() / n as f64
}
```

### Observable Functions for Finance

```rust
/// Financial observables for Koopman analysis
pub struct FinancialObservables {
    pub lookback: usize,
}

impl FinancialObservables {
    pub fn compute(&self, prices: &[f64]) -> Array1<f64> {
        let n = prices.len();
        if n < self.lookback {
            return Array1::zeros(0);
        }

        let mut observables = Vec::new();

        // Price itself
        observables.push(prices[n-1]);

        // Log returns
        for lag in 1..=self.lookback.min(5) {
            if n > lag {
                let log_ret = (prices[n-1] / prices[n-1-lag]).ln();
                observables.push(log_ret);
            }
        }

        // Moving averages
        for window in [5, 10, 20].iter() {
            if n >= *window {
                let ma: f64 = prices[n-window..].iter().sum::<f64>() / *window as f64;
                observables.push(ma);
            }
        }

        // Volatility
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        if returns.len() >= 10 {
            let recent_returns = &returns[returns.len()-10..];
            let mean = recent_returns.iter().sum::<f64>() / 10.0;
            let vol = (recent_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / 10.0).sqrt();
            observables.push(vol);
        }

        // Momentum
        if n >= 10 {
            let momentum = prices[n-1] / prices[n-10] - 1.0;
            observables.push(momentum);
        }

        // RSI-like feature
        if returns.len() >= 14 {
            let recent = &returns[returns.len()-14..];
            let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
            let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
            let rsi = if losses > 0.0 { gains / (gains + losses) } else { 1.0 };
            observables.push(rsi);
        }

        Array1::from_vec(observables)
    }
}
```

## Trading Strategies

### Strategy 1: DMD Mode Prediction

Use dominant DMD modes for short-term price prediction:

```rust
pub fn mode_prediction_strategy(
    candles: &[Candle],
    lookback: usize,
    prediction_horizon: usize,
) -> Vec<Signal> {
    let mut signals = Vec::new();
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    for i in lookback..prices.len() {
        // Build state matrix from recent prices
        let window = &prices[i-lookback..i];
        let state_matrix = delay_embed(window, 10);

        // Fit DMD
        if let Ok(dmd) = DMD::fit(&state_matrix, 1.0) {
            // Filter for stable, significant modes
            let stable_modes: Vec<_> = dmd.eigenvalues.iter()
                .enumerate()
                .filter(|(_, λ)| λ.norm() < 1.0 && λ.norm() > 0.1)
                .collect();

            // Predict next step
            let current_state = state_matrix.column(state_matrix.ncols() - 1).to_owned();
            let prediction = dmd.predict(prediction_horizon, &current_state);

            let current_price = window[window.len() - 1];
            let predicted_price = prediction[[0, prediction_horizon - 1]];
            let expected_return = (predicted_price - current_price) / current_price;

            signals.push(if expected_return > 0.001 {
                Signal::Long { strength: expected_return, confidence: 0.5 }
            } else if expected_return < -0.001 {
                Signal::Short { strength: -expected_return, confidence: 0.5 }
            } else {
                Signal::Neutral
            });
        } else {
            signals.push(Signal::Neutral);
        }
    }

    signals
}
```

### Strategy 2: Regime Detection

Detect regime changes by monitoring Koopman spectrum:

```rust
pub fn regime_detection_strategy(
    candles: &[Candle],
    window_size: usize,
    threshold: f64,
) -> Vec<RegimeLabel> {
    let mut regimes = Vec::new();
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    let mut prev_spectrum: Option<Array1<Complex64>> = None;

    for i in window_size..prices.len() {
        let window = &prices[i-window_size..i];
        let state_matrix = delay_embed(window, 5);

        if let Ok(dmd) = DMD::fit(&state_matrix, 1.0) {
            // Sort eigenvalues by magnitude for consistent comparison
            let mut spectrum = dmd.eigenvalues.clone();

            if let Some(ref prev) = prev_spectrum {
                let distance = spectral_distance(prev, &spectrum);

                if distance > threshold {
                    regimes.push(RegimeLabel::Transition);
                } else {
                    // Classify by dominant mode
                    let dominant = spectrum.iter()
                        .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap())
                        .unwrap();

                    if dominant.im.abs() > 0.1 {
                        regimes.push(RegimeLabel::Oscillatory);
                    } else if dominant.re > 0.0 {
                        regimes.push(RegimeLabel::Trending);
                    } else {
                        regimes.push(RegimeLabel::MeanReverting);
                    }
                }
            } else {
                regimes.push(RegimeLabel::Unknown);
            }

            prev_spectrum = Some(spectrum);
        } else {
            regimes.push(RegimeLabel::Unknown);
        }
    }

    regimes
}
```

### Strategy 3: Multi-Asset Koopman

Analyze cross-asset dynamics:

```rust
pub struct MultiAssetKoopman {
    pub symbols: Vec<String>,
    pub dmd: DMD,
}

impl MultiAssetKoopman {
    pub fn fit(price_matrix: &Array2<f64>, dt: f64) -> Result<Self> {
        let dmd = DMD::fit(price_matrix, dt)?;
        Ok(Self {
            symbols: Vec::new(),
            dmd,
        })
    }

    pub fn mode_contributions(&self) -> Array2<f64> {
        // Each row = asset, each column = mode contribution
        let n_assets = self.dmd.modes.nrows();
        let n_modes = self.dmd.modes.ncols();

        let mut contributions = Array2::zeros((n_assets, n_modes));

        for (j, (mode, amp)) in self.dmd.modes.columns()
            .into_iter()
            .zip(self.dmd.amplitudes.iter())
            .enumerate()
        {
            for i in 0..n_assets {
                contributions[[i, j]] = (mode[i] * amp).norm();
            }
        }

        // Normalize by row
        for mut row in contributions.rows_mut() {
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        contributions
    }

    pub fn lead_lag_analysis(&self) -> Array2<f64> {
        // Analyze phase differences between assets in each mode
        let n_assets = self.dmd.modes.nrows();
        let mut lead_lag = Array2::zeros((n_assets, n_assets));

        for mode in self.dmd.modes.columns() {
            let phases: Vec<f64> = mode.iter().map(|c| c.arg()).collect();

            for i in 0..n_assets {
                for j in 0..n_assets {
                    let phase_diff = phases[i] - phases[j];
                    lead_lag[[i, j]] += phase_diff;
                }
            }
        }

        lead_lag
    }
}
```

## Key Metrics

### Model Quality
- **Reconstruction error**: ||X - X_reconstructed||
- **Prediction RMSE**: Root mean squared error of forecasts
- **Mode stability**: Fraction of eigenvalues inside unit circle

### Trading Performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### Spectral Metrics
- **Spectral gap**: Separation between dominant and secondary modes
- **Coherence**: How well modes explain variance
- **Frequency content**: Dominant oscillation frequencies

## Dependencies

```toml
[dependencies]
ndarray = { version = "0.15", features = ["serde"] }
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
num-complex = "0.4"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
thiserror = "1.0"
```

## Implementation Notes

### Numerical Stability

1. **SVD truncation**: Always use truncated SVD for rank-deficient matrices
2. **Regularization**: Add small diagonal regularization to prevent singular matrices
3. **Normalization**: Normalize data before DMD to improve numerical stability

### Practical Considerations

1. **Delay embedding**: Use delay coordinates to convert scalar time series to state space
2. **Window selection**: Balance between capturing dynamics (large window) and stationarity (small window)
3. **Mode selection**: Filter modes by stability, energy, and relevance
4. **Online updates**: Consider incremental DMD for streaming applications

## Expected Outcomes

1. **DMD implementation** with prediction capabilities
2. **EDMD with multiple dictionaries** (polynomial, RBF, custom)
3. **Trading signal generator** based on Koopman predictions
4. **Regime detection** via spectral analysis
5. **Multi-asset analysis** including lead-lag relationships
6. **Backtesting framework** for Koopman strategies

## References

1. **Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics**
   - URL: https://arxiv.org/abs/1712.09707
   - Key contribution: Deep Koopman autoencoders

2. **Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems**
   - Authors: Kutz, Brunton, Brunton, Proctor
   - Comprehensive DMD textbook

3. **Data-Driven Science and Engineering**
   - Authors: Brunton & Kutz
   - Chapter on DMD and Koopman operator

4. **Extended Dynamic Mode Decomposition with Dictionary Learning**
   - URL: https://arxiv.org/abs/1510.04765
   - EDMD theoretical foundations

5. **Koopman Operator Theory for Financial Markets**
   - Applications to regime detection and prediction

## Difficulty Level

**Expert** (5/5)

Prerequisites:
- Linear algebra (SVD, eigendecomposition)
- Dynamical systems theory
- Time series analysis
- Rust programming

This chapter combines advanced mathematical concepts with practical trading applications.
