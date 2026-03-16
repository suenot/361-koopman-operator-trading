//! Prediction utilities for Koopman methods
//!
//! Provides helper functions for:
//! - Delay embedding (time-delay coordinates)
//! - Optimal rank selection
//! - Forecast evaluation

use ndarray::Array2;

/// Create delay embedding matrix from time series
///
/// Converts a scalar time series into a state-space representation
/// using time-delay coordinates (Takens' embedding theorem).
///
/// # Arguments
///
/// * `data` - Input time series
/// * `embed_dim` - Embedding dimension (number of delay coordinates)
///
/// # Returns
///
/// State matrix where each column is a delay-embedded state vector
///
/// # Example
///
/// ```
/// use koopman_trading::koopman::delay_embed;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let embedded = delay_embed(&data, 3);
/// // Creates matrix:
/// // [1, 2, 3]
/// // [2, 3, 4]
/// // [3, 4, 5]
/// ```
pub fn delay_embed(data: &[f64], embed_dim: usize) -> Array2<f64> {
    let n = data.len();
    if n <= embed_dim {
        return Array2::zeros((embed_dim, 1));
    }

    let n_cols = n - embed_dim + 1;
    let mut matrix = Array2::zeros((embed_dim, n_cols));

    for j in 0..n_cols {
        for i in 0..embed_dim {
            matrix[[i, j]] = data[j + i];
        }
    }

    matrix
}

/// Determine optimal rank for truncation based on singular value decay
///
/// Uses the "elbow" method to find where singular values drop significantly.
///
/// # Arguments
///
/// * `singular_values` - Vector of singular values (sorted descending)
///
/// # Returns
///
/// Optimal rank for truncation
pub fn optimal_rank(singular_values: &[f64]) -> usize {
    if singular_values.is_empty() {
        return 0;
    }

    if singular_values.len() == 1 {
        return 1;
    }

    // Find total energy
    let total_energy: f64 = singular_values.iter().map(|s| s * s).sum();

    if total_energy < 1e-10 {
        return 1;
    }

    // Find rank that captures 99% of energy
    let threshold = 0.99 * total_energy;
    let mut cumulative_energy = 0.0;

    for (i, s) in singular_values.iter().enumerate() {
        cumulative_energy += s * s;
        if cumulative_energy >= threshold {
            return (i + 1).max(1);
        }
    }

    singular_values.len()
}

/// Calculate Mean Squared Error between predictions and actual values
pub fn mse(predictions: &[f64], actual: &[f64]) -> f64 {
    if predictions.len() != actual.len() || predictions.is_empty() {
        return f64::INFINITY;
    }

    predictions
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}

/// Calculate Root Mean Squared Error
pub fn rmse(predictions: &[f64], actual: &[f64]) -> f64 {
    mse(predictions, actual).sqrt()
}

/// Calculate Mean Absolute Error
pub fn mae(predictions: &[f64], actual: &[f64]) -> f64 {
    if predictions.len() != actual.len() || predictions.is_empty() {
        return f64::INFINITY;
    }

    predictions
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / predictions.len() as f64
}

/// Calculate directional accuracy (percentage of correct direction predictions)
pub fn directional_accuracy(predictions: &[f64], actual: &[f64]) -> f64 {
    if predictions.len() < 2 || actual.len() < 2 {
        return 0.0;
    }

    let n = predictions.len().min(actual.len()) - 1;
    let mut correct = 0;

    for i in 0..n {
        let pred_dir = predictions[i + 1] - predictions[i];
        let actual_dir = actual[i + 1] - actual[i];

        if (pred_dir >= 0.0) == (actual_dir >= 0.0) {
            correct += 1;
        }
    }

    correct as f64 / n as f64
}

/// Rolling prediction evaluation
///
/// Evaluates prediction performance using rolling windows.
///
/// # Arguments
///
/// * `data` - Full time series
/// * `window_size` - Size of training window
/// * `horizon` - Prediction horizon
/// * `predictor` - Function that takes training data and returns predictions
///
/// # Returns
///
/// Tuple of (predictions, actuals) for evaluation
pub fn rolling_evaluation<F>(
    data: &[f64],
    window_size: usize,
    horizon: usize,
    predictor: F,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    let n = data.len();
    if n < window_size + horizon {
        return (predictions, actuals);
    }

    for i in window_size..(n - horizon) {
        let window = &data[i - window_size..i];
        let preds = predictor(window);

        if let Some(&pred) = preds.first() {
            predictions.push(pred);
            actuals.push(data[i]);
        }
    }

    (predictions, actuals)
}

/// Calculate spectral distance between two sets of eigenvalues
///
/// Uses Wasserstein distance on eigenvalue magnitudes.
pub fn spectral_distance(
    eigenvalues1: &[num_complex::Complex64],
    eigenvalues2: &[num_complex::Complex64],
) -> f64 {
    let mut mags1: Vec<f64> = eigenvalues1.iter().map(|ev| ev.norm()).collect();
    let mut mags2: Vec<f64> = eigenvalues2.iter().map(|ev| ev.norm()).collect();

    // Sort for comparison
    mags1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    mags2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Pad to same length
    let n = mags1.len().max(mags2.len());
    mags1.resize(n, 0.0);
    mags2.resize(n, 0.0);

    // Calculate L1 distance
    mags1
        .iter()
        .zip(mags2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_embed() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedded = delay_embed(&data, 3);

        assert_eq!(embedded.dim(), (3, 3));
        assert_eq!(embedded[[0, 0]], 1.0);
        assert_eq!(embedded[[1, 0]], 2.0);
        assert_eq!(embedded[[2, 0]], 3.0);
        assert_eq!(embedded[[0, 2]], 3.0);
        assert_eq!(embedded[[2, 2]], 5.0);
    }

    #[test]
    fn test_optimal_rank() {
        // Strong decay
        let sv = vec![10.0, 1.0, 0.1, 0.01];
        let rank = optimal_rank(&sv);
        assert!(rank <= 3);

        // Uniform
        let sv = vec![1.0, 1.0, 1.0, 1.0];
        let rank = optimal_rank(&sv);
        assert_eq!(rank, 4);
    }

    #[test]
    fn test_mse() {
        let pred = vec![1.0, 2.0, 3.0];
        let actual = vec![1.1, 2.1, 3.1];
        let error = mse(&pred, &actual);
        assert!((error - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_directional_accuracy() {
        let pred = vec![1.0, 2.0, 1.5, 2.5];
        let actual = vec![1.0, 2.0, 1.0, 3.0];
        let acc = directional_accuracy(&pred, &actual);
        assert!((acc - 2.0 / 3.0).abs() < 1e-10);
    }
}
