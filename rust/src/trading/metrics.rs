//! Performance metrics for trading strategy evaluation

/// Collection of trading performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Maximum consecutive wins
    pub max_consecutive_wins: usize,
    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,
}

impl PerformanceMetrics {
    /// Calculate all metrics from trade returns and equity curve
    pub fn calculate(
        trade_returns: &[f64],
        equity_curve: &[f64],
        periods_per_year: f64,
    ) -> Self {
        let n = equity_curve.len();

        // Total return
        let total_return = if n >= 2 && equity_curve[0] > 0.0 {
            (equity_curve[n - 1] - equity_curve[0]) / equity_curve[0]
        } else {
            0.0
        };

        // Daily returns from equity curve
        let daily_returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| if w[0] > 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 })
            .collect();

        // Volatility
        let volatility = calculate_volatility(&daily_returns, periods_per_year);

        // Annualized return
        let annualized_return = if n > 1 {
            let years = n as f64 / periods_per_year;
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Sharpe and Sortino
        let sharpe_ratio = calculate_sharpe(&daily_returns, 0.0);
        let sortino_ratio = calculate_sortino(&daily_returns, 0.0);

        // Max drawdown
        let max_drawdown = calculate_max_drawdown(equity_curve);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Win rate and profit factor
        let (win_rate, profit_factor, avg_trade_return) = if trade_returns.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let wins: Vec<f64> = trade_returns.iter().filter(|&&r| r > 0.0).cloned().collect();
            let losses: Vec<f64> = trade_returns.iter().filter(|&&r| r < 0.0).cloned().collect();

            let win_rate = wins.len() as f64 / trade_returns.len() as f64;
            let gross_profit: f64 = wins.iter().sum();
            let gross_loss: f64 = losses.iter().map(|l| -l).sum();
            let profit_factor = if gross_loss > 0.0 {
                gross_profit / gross_loss
            } else if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            let avg_trade_return = trade_returns.iter().sum::<f64>() / trade_returns.len() as f64;

            (win_rate, profit_factor, avg_trade_return)
        };

        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) =
            calculate_consecutive_runs(trade_returns);

        Self {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            avg_trade_return,
            max_consecutive_wins,
            max_consecutive_losses,
        }
    }
}

/// Calculate annualized volatility
pub fn calculate_volatility(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;

    variance.sqrt() * periods_per_year.sqrt()
}

/// Calculate Sharpe ratio
pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean - risk_free_rate / 252.0;

    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    let std = variance.sqrt();

    if std < 1e-10 {
        return 0.0;
    }

    (excess_return / std) * (252.0_f64).sqrt()
}

/// Calculate Sortino ratio
pub fn calculate_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean - risk_free_rate / 252.0;

    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    if downside_returns.is_empty() {
        return if excess_return > 0.0 { f64::INFINITY } else { 0.0 };
    }

    let downside_variance = downside_returns
        .iter()
        .map(|r| r.powi(2))
        .sum::<f64>()
        / downside_returns.len() as f64;
    let downside_std = downside_variance.sqrt();

    if downside_std < 1e-10 {
        return 0.0;
    }

    (excess_return / downside_std) * (252.0_f64).sqrt()
}

/// Calculate maximum drawdown from equity curve
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];

    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate maximum consecutive winning and losing streaks
fn calculate_consecutive_runs(returns: &[f64]) -> (usize, usize) {
    let mut max_wins = 0;
    let mut max_losses = 0;
    let mut current_wins = 0;
    let mut current_losses = 0;

    for &ret in returns {
        if ret > 0.0 {
            current_wins += 1;
            current_losses = 0;
            max_wins = max_wins.max(current_wins);
        } else if ret < 0.0 {
            current_losses += 1;
            current_wins = 0;
            max_losses = max_losses.max(current_losses);
        } else {
            current_wins = 0;
            current_losses = 0;
        }
    }

    (max_wins, max_losses)
}

/// Calculate Information Ratio
pub fn information_ratio(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    if portfolio_returns.len() != benchmark_returns.len() || portfolio_returns.len() < 2 {
        return 0.0;
    }

    let excess_returns: Vec<f64> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(p, b)| p - b)
        .collect();

    let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
    let tracking_error = calculate_volatility(&excess_returns, 252.0);

    if tracking_error < 1e-10 {
        return 0.0;
    }

    (mean_excess * 252.0) / tracking_error
}

/// Calculate Value at Risk (historical method)
pub fn calculate_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * returns.len() as f64).floor() as usize;
    let index = index.min(returns.len() - 1);

    -sorted_returns[index]
}

/// Calculate Conditional VaR (Expected Shortfall)
pub fn calculate_cvar(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cutoff_index = ((1.0 - confidence) * returns.len() as f64).ceil() as usize;
    let cutoff_index = cutoff_index.max(1).min(returns.len());

    let tail_returns: &[f64] = &sorted_returns[..cutoff_index];
    let mean_tail = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;

    -mean_tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
        let vol = calculate_volatility(&returns, 252.0);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_var() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let var_95 = calculate_var(&returns, 0.95);
        assert!(var_95 > 0.0);
    }

    #[test]
    fn test_consecutive_runs() {
        let returns = vec![0.01, 0.02, 0.01, -0.01, -0.02, 0.03];
        let (wins, losses) = calculate_consecutive_runs(&returns);
        assert_eq!(wins, 3);
        assert_eq!(losses, 2);
    }
}
