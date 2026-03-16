//! Dynamic Mode Decomposition (DMD)
//!
//! DMD is a data-driven method for extracting coherent structures (modes)
//! from time series data. It approximates the Koopman operator on a finite-
//! dimensional subspace.

use super::prediction::{delay_embed, optimal_rank};
use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Axis};
use num_complex::Complex64;

/// Dynamic Mode Decomposition
///
/// Extracts coherent spatio-temporal patterns from data by finding the best-fit
/// linear operator that advances the state in time.
#[derive(Debug, Clone)]
pub struct DMD {
    /// DMD modes (eigenvectors of the linear operator)
    pub modes: Vec<Array1<Complex64>>,
    /// Koopman eigenvalues
    pub eigenvalues: Vec<Complex64>,
    /// Mode amplitudes (for reconstruction)
    pub amplitudes: Vec<Complex64>,
    /// Time step
    pub dt: f64,
    /// State dimension
    pub state_dim: usize,
    /// Original data mean (for denormalization)
    pub data_mean: f64,
    /// Original data std (for denormalization)
    pub data_std: f64,
}

impl DMD {
    /// Create DMD from a time series using delay embedding
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `embed_dim` - Embedding dimension (delay coordinates)
    /// * `dt` - Time step between observations
    ///
    /// # Returns
    ///
    /// Fitted DMD model
    pub fn from_time_series(data: &[f64], embed_dim: usize, dt: f64) -> Result<Self> {
        if data.len() < embed_dim + 1 {
            return Err(anyhow!(
                "Data length {} too short for embedding dimension {}",
                data.len(),
                embed_dim
            ));
        }

        // Normalize data
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / data.len() as f64)
            .sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };

        let normalized: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        // Create delay embedding matrix
        let state_matrix = delay_embed(&normalized, embed_dim);

        // Perform DMD on embedded data
        let mut dmd = Self::fit(&state_matrix, dt)?;
        dmd.data_mean = mean;
        dmd.data_std = std;

        Ok(dmd)
    }

    /// Fit DMD to a state matrix
    ///
    /// # Arguments
    ///
    /// * `data` - State matrix where each column is a snapshot
    /// * `dt` - Time step between snapshots
    pub fn fit(data: &Array2<f64>, dt: f64) -> Result<Self> {
        let (n, m) = data.dim();
        if m < 2 {
            return Err(anyhow!("Need at least 2 snapshots for DMD"));
        }

        // Split into X and Y matrices
        let x = data.slice(s![.., ..m - 1]).to_owned();
        let y = data.slice(s![.., 1..]).to_owned();

        // Compute SVD of X using simple power iteration
        // For a proper implementation, use a linear algebra library
        let (u, s, vt) = simple_svd(&x)?;

        // Determine rank based on singular value decay
        let r = optimal_rank(&s);
        let r = r.max(1).min(n.min(m - 1));

        // Truncate to rank r
        let u_r = u.slice(s![.., ..r]).to_owned();
        let s_r: Vec<f64> = s.iter().take(r).cloned().collect();
        let vt_r = vt.slice(s![..r, ..]).to_owned();

        // Build reduced matrix: A_tilde = U_r^T * Y * V_r * S_r^{-1}
        let y_proj = u_r.t().dot(&y);
        let a_tilde = compute_a_tilde(&y_proj, &vt_r, &s_r);

        // Eigendecomposition of A_tilde
        let (eigenvalues, eigenvectors) = simple_eig(&a_tilde)?;

        // Compute DMD modes: Phi = Y * V_r * S_r^{-1} * W
        let modes = compute_modes(&y, &vt_r, &s_r, &eigenvectors);

        // Compute amplitudes by solving Phi * b = x_0
        let x0 = data.column(0).to_owned();
        let amplitudes = compute_amplitudes(&modes, &x0);

        Ok(Self {
            modes,
            eigenvalues,
            amplitudes,
            dt,
            state_dim: n,
            data_mean: 0.0,
            data_std: 1.0,
        })
    }

    /// Predict future states
    ///
    /// # Arguments
    ///
    /// * `steps` - Number of steps to predict
    ///
    /// # Returns
    ///
    /// Vector of predicted values (first component of state)
    pub fn predict(&self, steps: usize) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(steps);

        for t in 1..=steps {
            // Time dynamics: λ^t * b
            let mut pred = Complex64::new(0.0, 0.0);

            for (i, (lambda, b)) in self.eigenvalues.iter().zip(self.amplitudes.iter()).enumerate() {
                let time_factor = lambda.powf(t as f64);
                if i < self.modes.len() && !self.modes[i].is_empty() {
                    // Use first component of mode
                    pred += self.modes[i][0] * time_factor * b;
                }
            }

            // Denormalize and take real part
            let value = pred.re * self.data_std + self.data_mean;
            predictions.push(value);
        }

        predictions
    }

    /// Get continuous-time eigenvalues: ω = ln(λ) / dt
    pub fn continuous_eigenvalues(&self) -> Vec<Complex64> {
        self.eigenvalues
            .iter()
            .map(|lambda| lambda.ln() / self.dt)
            .collect()
    }

    /// Get oscillation frequencies
    pub fn frequencies(&self) -> Vec<f64> {
        self.continuous_eigenvalues()
            .iter()
            .map(|omega| omega.im / (2.0 * std::f64::consts::PI))
            .collect()
    }

    /// Get growth/decay rates
    pub fn growth_rates(&self) -> Vec<f64> {
        self.continuous_eigenvalues()
            .iter()
            .map(|omega| omega.re)
            .collect()
    }

    /// Get mode energies (squared amplitudes)
    pub fn mode_energies(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .zip(self.modes.iter())
            .map(|(b, mode)| {
                let mode_norm: f64 = mode.iter().map(|c| c.norm_sqr()).sum();
                b.norm_sqr() * mode_norm
            })
            .collect()
    }

    /// Reconstruct data at a given time
    pub fn reconstruct(&self, t: f64) -> Array1<f64> {
        let mut result = Array1::zeros(self.state_dim);

        for (i, ((lambda, b), mode)) in self
            .eigenvalues
            .iter()
            .zip(self.amplitudes.iter())
            .zip(self.modes.iter())
            .enumerate()
        {
            let time_factor = lambda.powc(Complex64::new(t / self.dt, 0.0));
            for j in 0..self.state_dim.min(mode.len()) {
                result[j] += (mode[j] * time_factor * b).re;
            }
        }

        // Denormalize
        result.mapv(|x| x * self.data_std + self.data_mean)
    }

    /// Check if model is stable (all eigenvalues inside unit circle)
    pub fn is_stable(&self) -> bool {
        self.eigenvalues.iter().all(|lambda| lambda.norm() < 1.0)
    }

    /// Get percentage of stable modes
    pub fn stability_ratio(&self) -> f64 {
        let stable = self
            .eigenvalues
            .iter()
            .filter(|lambda| lambda.norm() < 1.0)
            .count();
        stable as f64 / self.eigenvalues.len() as f64
    }
}

/// Simple SVD using power iteration (for educational purposes)
/// In production, use a proper linear algebra library like ndarray-linalg
fn simple_svd(a: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let (n, m) = a.dim();
    let k = n.min(m);

    let mut u = Array2::zeros((n, k));
    let mut s = Vec::with_capacity(k);
    let mut vt = Array2::zeros((k, m));

    let mut a_remaining = a.clone();

    for i in 0..k {
        // Power iteration to find dominant singular value/vectors
        let (sigma, ui, vi) = power_iteration(&a_remaining, 100)?;

        if sigma < 1e-10 {
            // Remaining singular values are effectively zero
            s.push(0.0);
            for j in 0..n {
                u[[j, i]] = 0.0;
            }
            for j in 0..m {
                vt[[i, j]] = 0.0;
            }
        } else {
            s.push(sigma);
            for j in 0..n {
                u[[j, i]] = ui[j];
            }
            for j in 0..m {
                vt[[i, j]] = vi[j];
            }

            // Deflate: A = A - sigma * u * v^T
            for j in 0..n {
                for l in 0..m {
                    a_remaining[[j, l]] -= sigma * ui[j] * vi[l];
                }
            }
        }
    }

    Ok((u, s, vt))
}

/// Power iteration to find largest singular value and vectors
fn power_iteration(a: &Array2<f64>, max_iter: usize) -> Result<(f64, Vec<f64>, Vec<f64>)> {
    let (n, m) = a.dim();

    // Initialize random vector
    let mut v: Vec<f64> = (0..m).map(|i| ((i * 7 + 3) % 11) as f64 / 11.0 - 0.5).collect();

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }

    for _ in 0..max_iter {
        // u = A * v
        let mut u = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                u[i] += a[[i, j]] * v[j];
            }
        }

        // Normalize u
        let norm_u: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_u < 1e-10 {
            return Ok((0.0, vec![0.0; n], vec![0.0; m]));
        }
        u.iter_mut().for_each(|x| *x /= norm_u);

        // v = A^T * u
        v = vec![0.0; m];
        for j in 0..m {
            for i in 0..n {
                v[j] += a[[i, j]] * u[i];
            }
        }

        // Normalize v
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_v < 1e-10 {
            return Ok((0.0, vec![0.0; n], vec![0.0; m]));
        }
        v.iter_mut().for_each(|x| *x /= norm_v);
    }

    // Compute singular value: sigma = ||A * v||
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            av[i] += a[[i, j]] * v[j];
        }
    }
    let sigma: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Compute left singular vector
    let u: Vec<f64> = if sigma > 1e-10 {
        av.iter().map(|x| x / sigma).collect()
    } else {
        vec![0.0; n]
    };

    Ok((sigma, u, v))
}

/// Compute reduced A matrix
fn compute_a_tilde(y_proj: &Array2<f64>, vt_r: &Array2<f64>, s_r: &[f64]) -> Array2<f64> {
    let r = s_r.len();
    let mut a_tilde = Array2::zeros((r, r));

    // A_tilde = Y_proj * V_r * S_r^{-1}
    for i in 0..r {
        for j in 0..r {
            let mut sum = 0.0;
            for k in 0..vt_r.ncols() {
                sum += y_proj[[i, k]] * vt_r[[j, k]];
            }
            if s_r[j].abs() > 1e-10 {
                a_tilde[[i, j]] = sum / s_r[j];
            }
        }
    }

    a_tilde
}

/// Simple eigendecomposition using power iteration with deflation
fn simple_eig(a: &Array2<f64>) -> Result<(Vec<Complex64>, Vec<Array1<Complex64>>)> {
    let n = a.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n);

    let mut a_remaining = a.clone();

    for _ in 0..n {
        // Power iteration for dominant eigenvalue
        let mut v: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 / 11.0).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);

        let mut lambda = 0.0;

        for _ in 0..100 {
            // w = A * v
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += a_remaining[[i, j]] * v[j];
                }
            }

            // Compute eigenvalue estimate
            lambda = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum();

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        eigenvalues.push(Complex64::new(lambda, 0.0));
        eigenvectors.push(Array1::from_iter(v.iter().map(|&x| Complex64::new(x, 0.0))));

        // Deflate: A = A - lambda * v * v^T
        let v_ref = &eigenvectors.last().unwrap();
        for i in 0..n {
            for j in 0..n {
                a_remaining[[i, j]] -= lambda * v_ref[i].re * v_ref[j].re;
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Compute DMD modes from eigendecomposition
fn compute_modes(
    y: &Array2<f64>,
    vt_r: &Array2<f64>,
    s_r: &[f64],
    eigenvectors: &[Array1<Complex64>],
) -> Vec<Array1<Complex64>> {
    let n = y.nrows();
    let r = s_r.len();

    // Phi = Y * V_r * S_r^{-1} * W
    eigenvectors
        .iter()
        .map(|w| {
            let mut mode = Array1::zeros(n);

            // First: V_r * S_r^{-1} * w
            let mut temp = vec![0.0; vt_r.ncols()];
            for k in 0..vt_r.ncols() {
                for i in 0..r {
                    if s_r[i].abs() > 1e-10 {
                        temp[k] += vt_r[[i, k]] * w[i].re / s_r[i];
                    }
                }
            }

            // Then: Y * temp
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..y.ncols().min(temp.len()) {
                    sum += y[[i, k]] * temp[k];
                }
                mode[i] = Complex64::new(sum, 0.0);
            }

            mode
        })
        .collect()
}

/// Compute mode amplitudes by least squares
fn compute_amplitudes(modes: &[Array1<Complex64>], x0: &Array1<f64>) -> Vec<Complex64> {
    // Simple approach: use diagonal of pseudo-inverse
    modes
        .iter()
        .map(|mode| {
            let mode_norm: f64 = mode.iter().map(|c| c.norm_sqr()).sum();
            if mode_norm > 1e-10 {
                let dot: Complex64 = mode
                    .iter()
                    .zip(x0.iter())
                    .map(|(m, x)| m.conj() * x)
                    .sum();
                dot / mode_norm
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmd_sine_wave() {
        // Create a simple sine wave
        let n = 100;
        let dt = 0.1;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 * dt / 10.0).sin())
            .collect();

        let dmd = DMD::from_time_series(&data, 10, dt).unwrap();

        // Should have modes
        assert!(!dmd.eigenvalues.is_empty());

        // Predictions should be reasonable
        let predictions = dmd.predict(5);
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_dmd_stability() {
        // Decaying exponential should be stable
        let data: Vec<f64> = (0..100).map(|i| (-0.1 * i as f64).exp()).collect();

        let dmd = DMD::from_time_series(&data, 5, 1.0).unwrap();

        // Stability ratio should be high for decaying signal
        assert!(dmd.stability_ratio() > 0.5);
    }
}
