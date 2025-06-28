"""
Statistical utility functions for the DOAC system
Includes half-life calculation, Hurst exponent, variance ratio test, and other statistical measures
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def calculate_half_life(spread: np.ndarray, 
                       max_lag: int = 50) -> float:
    """
    Calculate half-life of mean reversion using Ornstein-Uhlenbeck process
    
    Args:
        spread: Time series of spread values
        max_lag: Maximum lag to consider for AR(1) estimation
        
    Returns:
        Half-life in the same units as the input frequency
    """
    try:
        spread = np.array(spread)
        spread = spread[~np.isnan(spread)]  # Remove NaN values
        
        if len(spread) < 10:
            return np.inf
        
        # Use AR(1) model: spread[t] = μ + φ * spread[t-1] + ε[t]
        # Half-life = -log(2) / log(φ)
        
        y = spread[1:]  # spread[t]
        x = spread[:-1]  # spread[t-1]
        
        # Add constant term
        X = np.column_stack([np.ones(len(x)), x])
        
        # OLS regression
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            phi = beta[1]  # AR(1) coefficient
            
            if phi >= 1 or phi <= 0:
                return np.inf  # No mean reversion
            
            half_life = -np.log(2) / np.log(phi)
            return max(half_life, 0.1)  # Minimum half-life of 0.1 periods
            
        except np.linalg.LinAlgError:
            return np.inf
            
    except Exception:
        return np.inf


def calculate_hurst_exponent(time_series: np.ndarray, 
                           max_lag: int = 100) -> float:
    """
    Calculate Hurst exponent using R/S analysis
    
    Args:
        time_series: Time series data
        max_lag: Maximum lag for R/S calculation
        
    Returns:
        Hurst exponent (H < 0.5 indicates mean reversion)
    """
    try:
        time_series = np.array(time_series)
        time_series = time_series[~np.isnan(time_series)]
        
        if len(time_series) < 20:
            return 0.5  # Random walk default
        
        # Calculate cumulative deviations from mean
        mean_ts = np.mean(time_series)
        cumulative_deviations = np.cumsum(time_series - mean_ts)
        
        # Range of cumulative deviations
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        
        # Standard deviation
        S = np.std(time_series)
        
        if S == 0:
            return 0.5
        
        # R/S ratio
        rs_ratio = R / S
        
        # For a single series, approximate Hurst exponent
        n = len(time_series)
        hurst = np.log(rs_ratio) / np.log(n)
        
        # Ensure reasonable bounds
        return max(0.0, min(1.0, hurst))
        
    except Exception:
        return 0.5  # Default to random walk


def variance_ratio_test(time_series: np.ndarray, 
                       lags: List[int] = [2, 4, 8]) -> Tuple[float, float]:
    """
    Variance ratio test for random walk hypothesis
    
    Args:
        time_series: Time series data
        lags: List of lags to test
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    try:
        time_series = np.array(time_series)
        time_series = time_series[~np.isnan(time_series)]
        
        if len(time_series) < 20:
            return 0.0, 1.0
        
        # Calculate returns
        returns = np.diff(time_series)
        n = len(returns)
        
        if n < 10:
            return 0.0, 1.0
        
        # Variance of 1-period returns
        var_1 = np.var(returns, ddof=1)
        
        if var_1 == 0:
            return 0.0, 1.0
        
        # Calculate variance ratios for different lags
        variance_ratios = []
        
        for lag in lags:
            if lag >= n:
                continue
                
            # Calculate k-period returns
            k_returns = []
            for i in range(0, n - lag + 1, lag):
                k_return = np.sum(returns[i:i+lag])
                k_returns.append(k_return)
            
            if len(k_returns) < 2:
                continue
            
            # Variance of k-period returns
            var_k = np.var(k_returns, ddof=1)
            
            # Variance ratio
            vr = var_k / (lag * var_1)
            variance_ratios.append(vr)
        
        if not variance_ratios:
            return 0.0, 1.0
        
        # Test statistic: deviation from 1 (random walk expectation)
        mean_vr = np.mean(variance_ratios)
        test_statistic = abs(mean_vr - 1.0)
        
        # Approximate p-value (simplified)
        # In practice, would use proper variance ratio test statistics
        z_score = test_statistic / (0.1 + np.std(variance_ratios))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return test_statistic, min(p_value, 1.0)
        
    except Exception:
        return 0.0, 1.0


def calculate_ou_parameters(spread: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate Ornstein-Uhlenbeck process parameters
    
    dX_t = θ(μ - X_t)dt + σ dW_t
    
    Args:
        spread: Time series of spread values
        
    Returns:
        Tuple of (theta, mu, sigma) parameters
    """
    try:
        spread = np.array(spread)
        spread = spread[~np.isnan(spread)]
        
        if len(spread) < 10:
            return 0.0, 0.0, 1.0
        
        # Discrete approximation: X[t+1] - X[t] = θ(μ - X[t])Δt + σ√Δt * ε[t]
        # Rearranging: ΔX[t] = α + β*X[t] + ε[t]
        # where α = θμΔt, β = -θΔt
        
        delta_x = np.diff(spread)
        x_lagged = spread[:-1]
        
        # OLS regression: ΔX = α + β*X + ε
        X = np.column_stack([np.ones(len(x_lagged)), x_lagged])
        
        try:
            coeffs = np.linalg.lstsq(X, delta_x, rcond=None)[0]
            alpha, beta = coeffs
            
            # Extract OU parameters (assuming Δt = 1)
            theta = -beta
            mu = alpha / theta if theta != 0 else np.mean(spread)
            
            # Estimate sigma from residuals
            residuals = delta_x - alpha - beta * x_lagged
            sigma = np.std(residuals)
            
            # Ensure reasonable parameter bounds
            theta = max(theta, 1e-6)  # Positive mean reversion
            sigma = max(sigma, 1e-6)  # Positive volatility
            
            return theta, mu, sigma
            
        except np.linalg.LinAlgError:
            return 0.01, np.mean(spread), np.std(spread)
            
    except Exception:
        return 0.01, 0.0, 1.0


def calculate_z_score(spread: np.ndarray, 
                     theta: float, 
                     mu: float, 
                     sigma: float) -> np.ndarray:
    """
    Calculate OU-adjusted z-score
    
    z_t = (s_t - μ) / (σ / √(2θ))
    
    Args:
        spread: Spread values
        theta: Mean reversion speed
        mu: Long-term mean
        sigma: Volatility
        
    Returns:
        Array of z-scores
    """
    try:
        spread = np.array(spread)
        
        if theta <= 0 or sigma <= 0:
            # Fallback to standard z-score
            return (spread - np.mean(spread)) / (np.std(spread) + 1e-6)
        
        # OU equilibrium standard deviation
        ou_std = sigma / np.sqrt(2 * theta)
        
        z_scores = (spread - mu) / (ou_std + 1e-6)
        return z_scores
        
    except Exception:
        return np.zeros_like(spread)


def calculate_rolling_statistics(data: pd.Series, 
                                window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling statistical measures
    
    Args:
        data: Time series data
        window: Rolling window size
        
    Returns:
        DataFrame with rolling statistics
    """
    rolling_stats = pd.DataFrame(index=data.index)
    
    rolling_stats['mean'] = data.rolling(window).mean()
    rolling_stats['std'] = data.rolling(window).std()
    rolling_stats['skew'] = data.rolling(window).skew()
    rolling_stats['kurt'] = data.rolling(window).kurt()
    rolling_stats['min'] = data.rolling(window).min()
    rolling_stats['max'] = data.rolling(window).max()
    
    # Percentiles
    rolling_stats['p10'] = data.rolling(window).quantile(0.1)
    rolling_stats['p25'] = data.rolling(window).quantile(0.25)
    rolling_stats['p75'] = data.rolling(window).quantile(0.75)
    rolling_stats['p90'] = data.rolling(window).quantile(0.9)
    
    # Z-score
    rolling_stats['z_score'] = (data - rolling_stats['mean']) / (rolling_stats['std'] + 1e-6)
    
    return rolling_stats


def calculate_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """
    Calculate rolling maximum drawdown
    
    Args:
        cumulative_returns: Cumulative return series
        
    Returns:
        Series of drawdown values
    """
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    try:
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-6)
    except Exception:
        return 0.0


def calculate_maximum_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns: Cumulative return series
        
    Returns:
        Maximum drawdown as positive percentage
    """
    try:
        drawdown = calculate_drawdown(cumulative_returns)
        return abs(drawdown.min())
    except Exception:
        return 0.0


def calculate_var(returns: pd.Series, 
                 confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Return series
        confidence: Confidence level
        
    Returns:
        VaR as positive value
    """
    try:
        return abs(np.percentile(returns.dropna(), (1 - confidence) * 100))
    except Exception:
        return 0.0


def winsorize_series(series: pd.Series, 
                    lower_percentile: float = 0.01,
                    upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize extreme values in a time series
    
    Args:
        series: Input series
        lower_percentile: Lower bound percentile
        upper_percentile: Upper bound percentile
        
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    
    return series.clip(lower=lower_bound, upper=upper_bound) 