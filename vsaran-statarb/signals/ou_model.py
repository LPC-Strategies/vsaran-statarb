"""
Ornstein-Uhlenbeck Process Modeling and Signal Generation
Implements the core OU layer for spread modeling and z-score generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

from config.settings import config
from utils.statistics import calculate_ou_parameters, calculate_z_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class OUProcess:
    """Ornstein-Uhlenbeck process for spread modeling"""
    
    def __init__(self):
        self.theta = 0.0  # Mean reversion speed
        self.mu = 0.0     # Long-term mean
        self.sigma = 1.0  # Volatility
        self.fitted = False
        self.spread_history = []
        
    def fit(self, spread: np.ndarray, 
            method: str = 'mle') -> Dict[str, float]:
        """
        Fit OU process parameters to spread data
        
        Args:
            spread: Time series of spread values
            method: Estimation method ('mle', 'ols', 'moments')
            
        Returns:
            Dictionary with fitted parameters and statistics
        """
        spread = np.array(spread)
        spread = spread[~np.isnan(spread)]
        
        if len(spread) < config.OU_FITTING_WINDOW // 4:
            logger.warning(f"Insufficient data for OU fitting: {len(spread)} observations")
            return {}
        
        if method == 'mle':
            return self._fit_mle(spread)
        elif method == 'ols':
            return self._fit_ols(spread)
        elif method == 'moments':
            return self._fit_moments(spread)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
    
    def _fit_mle(self, spread: np.ndarray) -> Dict[str, float]:
        """
        Maximum likelihood estimation for OU parameters
        
        Args:
            spread: Spread time series
            
        Returns:
            Dictionary with MLE estimates
        """
        try:
            def negative_log_likelihood(params):
                theta, mu, sigma = params
                
                if theta <= 0 or sigma <= 0:
                    return 1e10
                
                dt = 1.0  # Assuming daily data
                n = len(spread) - 1
                
                # Calculate innovations
                innovations = np.diff(spread) - theta * (mu - spread[:-1]) * dt
                
                # Log-likelihood
                log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt)
                log_likelihood -= 0.5 * np.sum(innovations**2) / (sigma**2 * dt)
                
                return -log_likelihood
            
            # Initial parameter estimates
            theta_init, mu_init, sigma_init = calculate_ou_parameters(spread)
            initial_params = [theta_init, mu_init, sigma_init]
            
            # Bounds for parameters
            bounds = [(1e-6, 10), (-10, 10), (1e-6, 10)]
            
            # Optimize
            result = minimize(negative_log_likelihood, initial_params, 
                            method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                self.theta, self.mu, self.sigma = result.x
                self.fitted = True
                
                # Calculate additional statistics
                half_life = np.log(2) / self.theta
                equilibrium_var = self.sigma**2 / (2 * self.theta)
                
                return {
                    'theta': self.theta,
                    'mu': self.mu,
                    'sigma': self.sigma,
                    'half_life': half_life,
                    'equilibrium_var': equilibrium_var,
                    'log_likelihood': -result.fun,
                    'method': 'mle',
                    'success': True
                }
            else:
                logger.warning("MLE optimization failed, falling back to OLS")
                return self._fit_ols(spread)
                
        except Exception as e:
            logger.warning(f"MLE fitting failed: {e}, falling back to OLS")
            return self._fit_ols(spread)
    
    def _fit_ols(self, spread: np.ndarray) -> Dict[str, float]:
        """
        Ordinary least squares estimation for OU parameters
        
        Args:
            spread: Spread time series
            
        Returns:
            Dictionary with OLS estimates
        """
        try:
            self.theta, self.mu, self.sigma = calculate_ou_parameters(spread)
            self.fitted = True
            
            # Calculate additional statistics
            half_life = np.log(2) / self.theta if self.theta > 0 else np.inf
            equilibrium_var = self.sigma**2 / (2 * self.theta) if self.theta > 0 else np.inf
            
            return {
                'theta': self.theta,
                'mu': self.mu,
                'sigma': self.sigma,
                'half_life': half_life,
                'equilibrium_var': equilibrium_var,
                'method': 'ols',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"OLS fitting failed: {e}")
            return {'success': False}
    
    def _fit_moments(self, spread: np.ndarray) -> Dict[str, float]:
        """
        Method of moments estimation for OU parameters
        
        Args:
            spread: Spread time series
            
        Returns:
            Dictionary with moment estimates
        """
        try:
            # Sample moments
            sample_mean = np.mean(spread)
            sample_var = np.var(spread)
            
            # Lag-1 autocovariance
            lag1_cov = np.cov(spread[1:], spread[:-1])[0, 1]
            
            # OU parameter estimates from moments
            if sample_var > 0:
                rho = lag1_cov / sample_var  # AR(1) coefficient
                self.theta = -np.log(abs(rho)) if abs(rho) > 0 and abs(rho) < 1 else 0.01
            else:
                self.theta = 0.01
            
            self.mu = sample_mean
            
            # Estimate sigma from equilibrium variance
            equilibrium_var = sample_var
            self.sigma = np.sqrt(2 * self.theta * equilibrium_var) if self.theta > 0 else np.std(np.diff(spread))
            
            self.fitted = True
            
            half_life = np.log(2) / self.theta if self.theta > 0 else np.inf
            
            return {
                'theta': self.theta,
                'mu': self.mu,
                'sigma': self.sigma,
                'half_life': half_life,
                'equilibrium_var': equilibrium_var,
                'method': 'moments',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Moments fitting failed: {e}")
            return {'success': False}
    
    def calculate_z_score(self, spread: np.ndarray) -> np.ndarray:
        """
        Calculate OU-adjusted z-scores
        
        Args:
            spread: Current spread values
            
        Returns:
            Array of z-scores
        """
        if not self.fitted:
            logger.warning("OU process not fitted, using standard z-score")
            return (spread - np.mean(spread)) / (np.std(spread) + 1e-6)
        
        return calculate_z_score(spread, self.theta, self.mu, self.sigma)
    
    def predict(self, current_spread: float, 
                horizon: int = 1) -> Dict[str, float]:
        """
        Predict future spread values using OU process
        
        Args:
            current_spread: Current spread value
            horizon: Prediction horizon
            
        Returns:
            Dictionary with prediction statistics
        """
        if not self.fitted:
            return {}
        
        dt = horizon  # Assuming daily frequency
        
        # Expected value at time t+dt
        expected_spread = self.mu + (current_spread - self.mu) * np.exp(-self.theta * dt)
        
        # Variance at time t+dt
        if self.theta > 0:
            variance = (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * dt))
        else:
            variance = self.sigma**2 * dt
        
        std_dev = np.sqrt(variance)
        
        return {
            'expected_spread': expected_spread,
            'variance': variance,
            'std_dev': std_dev,
            'confidence_95_lower': expected_spread - 1.96 * std_dev,
            'confidence_95_upper': expected_spread + 1.96 * std_dev,
            'horizon': horizon
        }
    
    def get_equilibrium_stats(self) -> Dict[str, float]:
        """
        Get equilibrium statistics of the OU process
        
        Returns:
            Dictionary with equilibrium statistics
        """
        if not self.fitted:
            return {}
        
        if self.theta > 0:
            equilibrium_var = self.sigma**2 / (2 * self.theta)
            equilibrium_std = np.sqrt(equilibrium_var)
        else:
            equilibrium_var = np.inf
            equilibrium_std = np.inf
        
        half_life = np.log(2) / self.theta if self.theta > 0 else np.inf
        
        return {
            'long_term_mean': self.mu,
            'equilibrium_variance': equilibrium_var,
            'equilibrium_std': equilibrium_std,
            'half_life': half_life,
            'mean_reversion_speed': self.theta
        }


class SpreadSignalGenerator:
    """Generate trading signals from OU-modeled spreads"""
    
    def __init__(self):
        self.ou_models = {}  # Store OU models for each pair
        self.signal_history = {}
        
    def initialize_pair(self, 
                       pair_key: Tuple[str, str],
                       spread_data: np.ndarray,
                       kalman_params: Dict = None) -> bool:
        """
        Initialize OU model for a trading pair
        
        Args:
            pair_key: Tuple of (symbol1, symbol2)
            spread_data: Historical spread data
            kalman_params: Optional Kalman filter parameters
            
        Returns:
            Success flag
        """
        try:
            ou_model = OUProcess()
            
            # Fit OU parameters
            fit_result = ou_model.fit(spread_data, method='mle')
            
            if fit_result.get('success', False):
                self.ou_models[pair_key] = {
                    'model': ou_model,
                    'fit_result': fit_result,
                    'kalman_params': kalman_params or {},
                    'last_update': pd.Timestamp.now()
                }
                
                logger.info(f"Initialized OU model for {pair_key}: "
                           f"θ={ou_model.theta:.4f}, μ={ou_model.mu:.4f}, "
                           f"σ={ou_model.sigma:.4f}, half-life={fit_result.get('half_life', 0):.2f}")
                return True
            else:
                logger.warning(f"Failed to fit OU model for {pair_key}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing OU model for {pair_key}: {e}")
            return False
    
    def generate_signals(self, 
                        pair_key: Tuple[str, str],
                        current_spread: float,
                        spread_history: np.ndarray = None) -> Dict[str, float]:
        """
        Generate trading signals for a pair
        
        Args:
            pair_key: Tuple of (symbol1, symbol2)
            current_spread: Current spread value
            spread_history: Recent spread history for dynamic updates
            
        Returns:
            Dictionary with signal information
        """
        if pair_key not in self.ou_models:
            logger.warning(f"No OU model found for {pair_key}")
            return {}
        
        try:
            ou_data = self.ou_models[pair_key]
            ou_model = ou_data['model']
            
            # Update model if new data available
            if spread_history is not None and len(spread_history) > config.OU_FITTING_WINDOW // 2:
                self._update_model(pair_key, spread_history)
            
            # Calculate z-score
            z_score = ou_model.calculate_z_score(np.array([current_spread]))[0]
            
            # Generate entry/exit signals
            entry_signal = 0
            exit_signal = 0
            confidence = 0.0
            
            # Entry conditions (|z| >= entry threshold)
            if abs(z_score) >= config.ENTRY_Z_THRESHOLD:
                entry_signal = -1 if z_score > 0 else 1  # Sell spread if z > 0, buy if z < 0
                confidence = min(abs(z_score) / config.ENTRY_Z_THRESHOLD, 3.0)  # Cap at 3x threshold
            
            # Exit conditions (|z| <= exit threshold)
            elif abs(z_score) <= config.EXIT_Z_THRESHOLD:
                exit_signal = 1  # Exit any position
                confidence = 1.0 - abs(z_score) / config.EXIT_Z_THRESHOLD
            
            # Get prediction for position sizing
            prediction = ou_model.predict(current_spread, horizon=1)
            
            # Calculate position sizing hint based on Kelly criterion
            expected_return = abs(prediction.get('expected_spread', 0) - current_spread)
            prediction_std = prediction.get('std_dev', ou_model.sigma)
            
            if prediction_std > 0:
                kelly_fraction = expected_return / (prediction_std ** 2)
                kelly_fraction = min(kelly_fraction, config.MAX_POSITION_SIZE)
            else:
                kelly_fraction = 0.0
            
            # Time-based exit (position holding period)
            equilibrium_stats = ou_model.get_equilibrium_stats()
            max_holding_period = min(
                equilibrium_stats.get('half_life', config.MAX_HOLDING_PERIOD) * 2,
                config.MAX_HOLDING_PERIOD
            )
            
            signal_data = {
                'z_score': z_score,
                'entry_signal': entry_signal,
                'exit_signal': exit_signal,
                'confidence': confidence,
                'kelly_fraction': kelly_fraction,
                'expected_spread': prediction.get('expected_spread', current_spread),
                'prediction_std': prediction_std,
                'max_holding_period': max_holding_period,
                'ou_theta': ou_model.theta,
                'ou_mu': ou_model.mu,
                'ou_sigma': ou_model.sigma,
                'timestamp': pd.Timestamp.now()
            }
            
            # Store signal history
            if pair_key not in self.signal_history:
                self.signal_history[pair_key] = []
            
            self.signal_history[pair_key].append(signal_data.copy())
            
            # Keep only recent history
            if len(self.signal_history[pair_key]) > 1000:
                self.signal_history[pair_key] = self.signal_history[pair_key][-1000:]
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error generating signals for {pair_key}: {e}")
            return {}
    
    def _update_model(self, 
                     pair_key: Tuple[str, str],
                     new_spread_data: np.ndarray) -> bool:
        """
        Update OU model parameters with new data
        
        Args:
            pair_key: Pair identifier
            new_spread_data: New spread observations
            
        Returns:
            Success flag
        """
        try:
            if pair_key not in self.ou_models:
                return False
            
            ou_data = self.ou_models[pair_key]
            ou_model = ou_data['model']
            
            # Refit model with new data
            fit_result = ou_model.fit(new_spread_data, method='mle')
            
            if fit_result.get('success', False):
                ou_data['fit_result'] = fit_result
                ou_data['last_update'] = pd.Timestamp.now()
                
                logger.debug(f"Updated OU model for {pair_key}: "
                           f"θ={ou_model.theta:.4f}, μ={ou_model.mu:.4f}")
                return True
            else:
                logger.warning(f"Failed to update OU model for {pair_key}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating OU model for {pair_key}: {e}")
            return False
    
    def get_signal_summary(self, 
                          pair_key: Tuple[str, str],
                          lookback_periods: int = 100) -> pd.DataFrame:
        """
        Get summary of recent signals for a pair
        
        Args:
            pair_key: Pair identifier
            lookback_periods: Number of recent periods to include
            
        Returns:
            DataFrame with signal history
        """
        if pair_key not in self.signal_history:
            return pd.DataFrame()
        
        history = self.signal_history[pair_key][-lookback_periods:]
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_all_current_signals(self) -> Dict[Tuple[str, str], Dict]:
        """
        Get current signals for all active pairs
        
        Returns:
            Dictionary mapping pair keys to current signal data
        """
        current_signals = {}
        
        for pair_key in self.ou_models.keys():
            if pair_key in self.signal_history and self.signal_history[pair_key]:
                latest_signal = self.signal_history[pair_key][-1]
                current_signals[pair_key] = latest_signal
        
        return current_signals


# Global instance
signal_generator = SpreadSignalGenerator() 