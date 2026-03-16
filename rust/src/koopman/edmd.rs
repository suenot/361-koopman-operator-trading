//! Extended Dynamic Mode Decomposition (EDMD)
//!
//! EDMD extends DMD by lifting data to a higher-dimensional feature space
//! using dictionary functions. This provides better approximations of the
//! Koopman operator for nonlinear systems.

use super::prediction::delay_embed;
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Dictionary trait for EDMD lifting functions
pub trait Dictionary: Send + Sync {
    /// Lift a state vector to the dictionary space
    fn lift(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Get the dimension of the lifted space
    fn dim(&self) -> usize;

    /// Get dictionary name
    fn name(&self) -> &str;
}

/// Polynomial dictionary for EDMD
///
/// Creates polynomial features up to a specified degree.
#[derive(Debug, Clone)]
pub struct PolynomialDictionary {
    /// Maximum polynomial degree
    pub degree: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Computed output dimension
    output_dim: usize,
}

impl PolynomialDictionary {
    /// Create a new polynomial dictionary
    ///
    /// # Arguments
    ///
    /// * `degree` - Maximum polynomial degree
    /// * `input_dim` - Dimension of input vectors
    pub fn new(degree: usize, input_dim: usize) -> Self {
        // Calculate output dimension using stars and bars formula
        let output_dim = Self::compute_dim(degree, input_dim);
        Self {
            degree,
            input_dim,
            output_dim,
        }
    }

    fn compute_dim(degree: usize, input_dim: usize) -> usize {
        // Sum of C(n + k - 1, k) for k = 0 to degree
        let mut dim = 0;
        for k in 0..=degree {
            dim += binomial(input_dim + k - 1, k);
        }
        dim
    }
}

impl Dictionary for PolynomialDictionary {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut features = vec![1.0]; // Constant term

        // First-order terms
        for i in 0..self.input_dim.min(x.len()) {
            features.push(x[i]);
        }

        // Higher-order terms
        if self.degree >= 2 && x.len() >= self.input_dim {
            // Second-order: x_i * x_j for i <= j
            for i in 0..self.input_dim {
                for j in i..self.input_dim {
                    features.push(x[i] * x[j]);
                }
            }

            // Third-order and beyond (simplified)
            if self.degree >= 3 {
                for i in 0..self.input_dim {
                    features.push(x[i].powi(3));
                }
            }
        }

        Array1::from_vec(features)
    }

    fn dim(&self) -> usize {
        self.output_dim
    }

    fn name(&self) -> &str {
        "Polynomial"
    }
}

/// Radial Basis Function (RBF) dictionary for EDMD
///
/// Uses Gaussian RBFs centered at specified points.
#[derive(Debug, Clone)]
pub struct RBFDictionary {
    /// RBF center points
    pub centers: Array2<f64>,
    /// RBF width parameter (sigma)
    pub sigma: f64,
}

impl RBFDictionary {
    /// Create a new RBF dictionary
    ///
    /// # Arguments
    ///
    /// * `centers` - Matrix of center points (each row is a center)
    /// * `sigma` - RBF width parameter
    pub fn new(centers: Array2<f64>, sigma: f64) -> Self {
        Self { centers, sigma }
    }

    /// Create RBF dictionary from data using k-means-like initialization
    ///
    /// # Arguments
    ///
    /// * `data` - Training data (each column is a sample)
    /// * `n_centers` - Number of RBF centers
    /// * `sigma` - RBF width parameter
    pub fn from_data(data: &Array2<f64>, n_centers: usize, sigma: f64) -> Self {
        let (n, m) = data.dim();
        let n_centers = n_centers.min(m);

        // Simple initialization: use evenly spaced data points
        let mut centers = Array2::zeros((n_centers, n));
        let step = m / n_centers;

        for i in 0..n_centers {
            let idx = (i * step).min(m - 1);
            for j in 0..n {
                centers[[i, j]] = data[[j, idx]];
            }
        }

        Self { centers, sigma }
    }
}

impl Dictionary for RBFDictionary {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64> {
        let n_centers = self.centers.nrows();
        let mut features = Array1::zeros(n_centers + 1);

        // Constant term
        features[0] = 1.0;

        // RBF features
        for (i, center) in self.centers.rows().into_iter().enumerate() {
            let mut dist_sq = 0.0;
            for j in 0..x.len().min(center.len()) {
                let diff = x[j] - center[j];
                dist_sq += diff * diff;
            }
            features[i + 1] = (-dist_sq / (2.0 * self.sigma * self.sigma)).exp();
        }

        features
    }

    fn dim(&self) -> usize {
        self.centers.nrows() + 1
    }

    fn name(&self) -> &str {
        "RBF"
    }
}

/// Extended Dynamic Mode Decomposition
///
/// EDMD uses dictionary functions to lift data to a higher-dimensional space
/// where the Koopman operator can be better approximated.
#[derive(Debug)]
pub struct EDMD<D: Dictionary> {
    /// Dictionary for lifting
    pub dictionary: D,
    /// Approximated Koopman matrix in dictionary space
    pub koopman_matrix: Array2<f64>,
    /// Eigenvalues of the Koopman matrix
    pub eigenvalues: Vec<Complex64>,
    /// Eigenvectors of the Koopman matrix
    pub eigenvectors: Vec<Array1<Complex64>>,
    /// State dimension
    pub state_dim: usize,
    /// Data mean for normalization
    pub data_mean: f64,
    /// Data std for normalization
    pub data_std: f64,
}

impl<D: Dictionary> EDMD<D> {
    /// Fit EDMD to time series data
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `dictionary` - Dictionary for lifting
    /// * `embed_dim` - Embedding dimension for delay coordinates
    pub fn from_time_series(data: &[f64], dictionary: D, embed_dim: usize) -> Result<Self> {
        if data.len() < embed_dim + 1 {
            return Err(anyhow!("Data too short for embedding"));
        }

        // Normalize
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / data.len() as f64)
            .sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let normalized: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        // Create delay embedding
        let state_matrix = delay_embed(&normalized, embed_dim);

        let mut edmd = Self::fit(&state_matrix, dictionary)?;
        edmd.data_mean = mean;
        edmd.data_std = std;

        Ok(edmd)
    }

    /// Fit EDMD to state matrix data
    ///
    /// # Arguments
    ///
    /// * `data` - State matrix (each column is a snapshot)
    /// * `dictionary` - Dictionary for lifting
    pub fn fit(data: &Array2<f64>, dictionary: D) -> Result<Self> {
        let (n, m) = data.dim();
        if m < 2 {
            return Err(anyhow!("Need at least 2 snapshots"));
        }

        let lifted_dim = dictionary.dim();

        // Lift all data points
        let mut psi_x = Array2::zeros((lifted_dim, m - 1));
        let mut psi_y = Array2::zeros((lifted_dim, m - 1));

        for i in 0..(m - 1) {
            let x_i = data.column(i).to_owned();
            let y_i = data.column(i + 1).to_owned();

            let psi_x_i = dictionary.lift(&x_i);
            let psi_y_i = dictionary.lift(&y_i);

            for j in 0..lifted_dim.min(psi_x_i.len()) {
                psi_x[[j, i]] = psi_x_i[j];
                psi_y[[j, i]] = psi_y_i[j];
            }
        }

        // Solve for Koopman matrix: Psi_Y = K * Psi_X
        // Using least squares: K = Psi_Y * Psi_X^T * (Psi_X * Psi_X^T)^{-1}
        let g = psi_x.dot(&psi_x.t());
        let a = psi_y.dot(&psi_x.t());

        let koopman_matrix = solve_least_squares(&g, &a)?;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = simple_eig_edmd(&koopman_matrix)?;

        Ok(Self {
            dictionary,
            koopman_matrix,
            eigenvalues,
            eigenvectors,
            state_dim: n,
            data_mean: 0.0,
            data_std: 1.0,
        })
    }

    /// Predict future states
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial state
    /// * `steps` - Number of steps to predict
    pub fn predict(&self, x0: &Array1<f64>, steps: usize) -> Vec<f64> {
        let mut psi = self.dictionary.lift(x0);
        let mut predictions = Vec::with_capacity(steps);

        for _ in 0..steps {
            // Advance in dictionary space
            psi = self.koopman_matrix.dot(&psi);

            // Extract observable (first component after constant)
            let value = if psi.len() > 1 {
                psi[1] * self.data_std + self.data_mean
            } else {
                psi[0] * self.data_std + self.data_mean
            };
            predictions.push(value);
        }

        predictions
    }

    /// Get Koopman eigenvalues
    pub fn get_eigenvalues(&self) -> &[Complex64] {
        &self.eigenvalues
    }

    /// Check stability
    pub fn is_stable(&self) -> bool {
        self.eigenvalues.iter().all(|ev| ev.norm() < 1.0)
    }
}

/// Solve least squares system using simple Gauss-Jordan elimination
fn solve_least_squares(g: &Array2<f64>, a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = g.nrows();

    // Add regularization for numerical stability
    let reg = 1e-6;
    let mut g_reg = g.clone();
    for i in 0..n {
        g_reg[[i, i]] += reg;
    }

    // Solve G * K^T = A^T using simple iteration
    let mut k = Array2::zeros((n, n));

    // Gauss-Seidel iteration
    for _ in 0..100 {
        for i in 0..n {
            for j in 0..n {
                let mut sum = a[[i, j]];
                for l in 0..n {
                    if l != i {
                        sum -= g_reg[[i, l]] * k[[l, j]];
                    }
                }
                if g_reg[[i, i]].abs() > 1e-10 {
                    k[[i, j]] = sum / g_reg[[i, i]];
                }
            }
        }
    }

    Ok(k)
}

/// Simple eigendecomposition for EDMD
fn simple_eig_edmd(a: &Array2<f64>) -> Result<(Vec<Complex64>, Vec<Array1<Complex64>>)> {
    let n = a.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n);

    let mut a_remaining = a.clone();

    for _ in 0..n.min(10) {
        // Limit number of eigenvalues
        let mut v: Vec<f64> = (0..n).map(|i| ((i * 13 + 7) % 17) as f64 / 17.0).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v.iter_mut().for_each(|x| *x /= norm);
        }

        let mut lambda = 0.0;

        for _ in 0..50 {
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += a_remaining[[i, j]] * v[j];
                }
            }

            lambda = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum();

            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        eigenvalues.push(Complex64::new(lambda, 0.0));
        eigenvectors.push(Array1::from_iter(v.iter().map(|&x| Complex64::new(x, 0.0))));

        // Deflate
        let v_ref = &v;
        for i in 0..n {
            for j in 0..n {
                a_remaining[[i, j]] -= lambda * v_ref[i] * v_ref[j];
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Binomial coefficient
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1usize;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_dictionary() {
        let dict = PolynomialDictionary::new(2, 3);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let lifted = dict.lift(&x);

        // Should include: 1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2
        assert!(lifted.len() >= 10);
        assert_eq!(lifted[0], 1.0); // Constant
        assert_eq!(lifted[1], 1.0); // x1
        assert_eq!(lifted[2], 2.0); // x2
        assert_eq!(lifted[3], 3.0); // x3
    }

    #[test]
    fn test_rbf_dictionary() {
        let centers = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let dict = RBFDictionary::new(centers, 1.0);

        let x = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let lifted = dict.lift(&x);

        assert_eq!(lifted.len(), 3); // 1 constant + 2 RBFs
        assert_eq!(lifted[0], 1.0); // Constant
        assert!((lifted[1] - 1.0).abs() < 1e-6); // RBF at origin = 1
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(4, 0), 1);
        assert_eq!(binomial(4, 4), 1);
    }
}
