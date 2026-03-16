//! Feature lifting utilities for Koopman analysis
//!
//! Provides functions to create enhanced feature representations
//! from raw price data.

use ndarray::Array2;

/// Create delay coordinate embedding
///
/// Transforms a scalar time series into a matrix where each column
/// is a delay-embedded state vector.
///
/// # Arguments
///
/// * `data` - Input time series
/// * `embed_dim` - Number of delay coordinates
/// * `delay` - Time delay between coordinates (default: 1)
///
/// # Returns
///
/// Matrix of delay-embedded states
pub fn delay_coordinates(data: &[f64], embed_dim: usize, delay: usize) -> Array2<f64> {
    let n = data.len();
    let total_delay = (embed_dim - 1) * delay;

    if n <= total_delay {
        return Array2::zeros((embed_dim, 1));
    }

    let n_cols = n - total_delay;
    let mut matrix = Array2::zeros((embed_dim, n_cols));

    for j in 0..n_cols {
        for i in 0..embed_dim {
            matrix[[i, j]] = data[j + i * delay];
        }
    }

    matrix
}

/// Create feature matrix from multiple time series
///
/// Combines multiple time series into a single state matrix.
///
/// # Arguments
///
/// * `series` - Vector of time series (all same length)
///
/// # Returns
///
/// Matrix where each row is a different series
pub fn create_feature_matrix(series: &[Vec<f64>]) -> Array2<f64> {
    if series.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_features = series.len();
    let n_samples = series[0].len();

    let mut matrix = Array2::zeros((n_features, n_samples));

    for (i, s) in series.iter().enumerate() {
        for (j, &val) in s.iter().enumerate().take(n_samples) {
            matrix[[i, j]] = val;
        }
    }

    matrix
}

/// Create polynomial features from a state vector
///
/// # Arguments
///
/// * `x` - Input state vector
/// * `degree` - Maximum polynomial degree
///
/// # Returns
///
/// Vector of polynomial features
pub fn polynomial_features(x: &[f64], degree: usize) -> Vec<f64> {
    let mut features = vec![1.0]; // Constant term

    // First order
    features.extend_from_slice(x);

    if degree >= 2 {
        // Second order: x_i * x_j for i <= j
        for i in 0..x.len() {
            for j in i..x.len() {
                features.push(x[i] * x[j]);
            }
        }
    }

    if degree >= 3 {
        // Third order: x_i^3
        for &xi in x {
            features.push(xi.powi(3));
        }
    }

    features
}

/// Create RBF (Radial Basis Function) features
///
/// # Arguments
///
/// * `x` - Input vector
/// * `centers` - RBF center points
/// * `sigma` - RBF width parameter
///
/// # Returns
///
/// Vector of RBF features
pub fn rbf_features(x: &[f64], centers: &[Vec<f64>], sigma: f64) -> Vec<f64> {
    let mut features = vec![1.0]; // Constant

    for center in centers {
        let dist_sq: f64 = x
            .iter()
            .zip(center.iter())
            .map(|(xi, ci)| (xi - ci).powi(2))
            .sum();

        features.push((-dist_sq / (2.0 * sigma * sigma)).exp());
    }

    features
}

/// Create Fourier features for time series
///
/// # Arguments
///
/// * `t` - Time values
/// * `n_harmonics` - Number of harmonic frequencies
/// * `period` - Base period
///
/// # Returns
///
/// Matrix of sine and cosine features
pub fn fourier_features(t: &[f64], n_harmonics: usize, period: f64) -> Array2<f64> {
    let n = t.len();
    let n_features = 2 * n_harmonics + 1;
    let mut features = Array2::zeros((n_features, n));

    // Constant term
    for j in 0..n {
        features[[0, j]] = 1.0;
    }

    // Sine and cosine terms
    for k in 1..=n_harmonics {
        let freq = 2.0 * std::f64::consts::PI * k as f64 / period;
        for j in 0..n {
            features[[2 * k - 1, j]] = (freq * t[j]).sin();
            features[[2 * k, j]] = (freq * t[j]).cos();
        }
    }

    features
}

/// Normalize features to zero mean and unit variance
///
/// # Arguments
///
/// * `features` - Feature matrix (features x samples)
///
/// # Returns
///
/// Normalized feature matrix and (means, stds) for denormalization
pub fn normalize_features(features: &Array2<f64>) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
    let (n_features, n_samples) = features.dim();

    let mut means = Vec::with_capacity(n_features);
    let mut stds = Vec::with_capacity(n_features);
    let mut normalized = features.clone();

    for i in 0..n_features {
        let row = features.row(i);
        let mean = row.sum() / n_samples as f64;
        let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let std = variance.sqrt().max(1e-10);

        means.push(mean);
        stds.push(std);

        for j in 0..n_samples {
            normalized[[i, j]] = (features[[i, j]] - mean) / std;
        }
    }

    (normalized, means, stds)
}

/// Apply Hankel matrix transformation
///
/// Creates a Hankel matrix from a time series, useful for
/// analyzing linear dynamics.
///
/// # Arguments
///
/// * `data` - Input time series
/// * `n_rows` - Number of rows in Hankel matrix
///
/// # Returns
///
/// Hankel matrix
pub fn hankel_matrix(data: &[f64], n_rows: usize) -> Array2<f64> {
    let n = data.len();
    if n < n_rows {
        return Array2::zeros((n_rows, 1));
    }

    let n_cols = n - n_rows + 1;
    let mut h = Array2::zeros((n_rows, n_cols));

    for j in 0..n_cols {
        for i in 0..n_rows {
            h[[i, j]] = data[i + j];
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_coordinates() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedded = delay_coordinates(&data, 3, 1);
        assert_eq!(embedded.dim(), (3, 3));
    }

    #[test]
    fn test_delay_coordinates_with_delay() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let embedded = delay_coordinates(&data, 3, 2);
        // With delay=2: [x_0, x_2, x_4], [x_1, x_3, x_5], [x_2, x_4, x_6]
        assert_eq!(embedded.dim(), (3, 3));
        assert_eq!(embedded[[0, 0]], 1.0);
        assert_eq!(embedded[[1, 0]], 3.0);
        assert_eq!(embedded[[2, 0]], 5.0);
    }

    #[test]
    fn test_polynomial_features() {
        let x = vec![1.0, 2.0];
        let poly = polynomial_features(&x, 2);
        // 1, x1, x2, x1^2, x1*x2, x2^2
        assert_eq!(poly.len(), 6);
        assert_eq!(poly[0], 1.0);
        assert_eq!(poly[1], 1.0);
        assert_eq!(poly[2], 2.0);
        assert_eq!(poly[3], 1.0); // x1^2
        assert_eq!(poly[4], 2.0); // x1*x2
        assert_eq!(poly[5], 4.0); // x2^2
    }

    #[test]
    fn test_rbf_features() {
        let x = vec![0.0, 0.0];
        let centers = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let features = rbf_features(&x, &centers, 1.0);

        assert_eq!(features.len(), 3);
        assert_eq!(features[0], 1.0); // Constant
        assert!((features[1] - 1.0).abs() < 1e-10); // RBF at origin
    }

    #[test]
    fn test_hankel_matrix() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let h = hankel_matrix(&data, 3);
        assert_eq!(h.dim(), (3, 3));
        assert_eq!(h[[0, 0]], 1.0);
        assert_eq!(h[[1, 0]], 2.0);
        assert_eq!(h[[2, 0]], 3.0);
        assert_eq!(h[[0, 2]], 3.0);
        assert_eq!(h[[2, 2]], 5.0);
    }

    #[test]
    fn test_normalize_features() {
        let features = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
            .unwrap();
        let (normalized, means, stds) = normalize_features(&features);

        assert_eq!(normalized.dim(), (2, 4));
        // Check that each row has approximately zero mean
        for i in 0..2 {
            let row_mean: f64 = normalized.row(i).sum() / 4.0;
            assert!(row_mean.abs() < 1e-10);
        }
    }
}
