"""
Pair & Basket Discovery Pipeline for the DOAC system
Implements the discovery pipeline:
1. ρ-screen: rolling 60-day Spearman correlation > 0.6
2. Stationarity lab: Johansen test, variance-ratio test  
3. State-space fit: Kalman filter for time-varying β
4. Half-life & Hurst exponent analysis
5. Capacity stress testing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
from numba import jit
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import config
from utils.statistics import calculate_half_life, calculate_hurst_exponent, variance_ratio_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class PairDiscovery:
    """Main pair discovery engine implementing the five-step pipeline"""
    
    def __init__(self):
        self.discovered_pairs = {}
        self.discovery_stats = {}
        
    def screen_correlations(self, 
                           price_data: pd.DataFrame,
                           symbols: List[str],
                           min_correlation: float = None) -> Dict[Tuple[str, str], float]:
        """
        Step 1: Screen for high rolling Spearman correlations
        
        Args:
            price_data: Historical price data
            symbols: Universe symbols to screen
            min_correlation: Minimum correlation threshold
            
        Returns:
            Dictionary of qualifying pairs with correlation values
        """
        if min_correlation is None:
            min_correlation = config.MIN_CORRELATION
            
        logger.info(f"Screening correlations for {len(symbols)} symbols")
        
        # Extract log prices for correlation analysis
        log_prices = {}
        for symbol in symbols:
            if symbol in price_data.columns.get_level_values(0):
                prices = price_data[symbol]['Close'].dropna()
                if len(prices) > config.CORRELATION_WINDOW:
                    log_prices[symbol] = np.log(prices)
        
        valid_symbols = list(log_prices.keys())
        logger.info(f"Valid symbols for correlation screening: {len(valid_symbols)}")
        
        if len(valid_symbols) < 2:
            return {}
        
        qualifying_pairs = {}
        
        # Calculate rolling correlations for all pairs
        for i, symbol1 in enumerate(valid_symbols):
            for j, symbol2 in enumerate(valid_symbols[i+1:], i+1):
                try:
                    series1 = log_prices[symbol1]
                    series2 = log_prices[symbol2]
                    
                    # Align series
                    aligned_data = pd.concat([series1, series2], axis=1).dropna()
                    if len(aligned_data) < config.CORRELATION_WINDOW + 20:
                        continue
                    
                    aligned_data.columns = [symbol1, symbol2]
                    
                    # Calculate rolling Spearman correlation
                    rolling_corr = aligned_data[symbol1].rolling(
                        window=config.CORRELATION_WINDOW
                    ).corr(aligned_data[symbol2], method='spearman')
                    
                    # Check if average correlation meets threshold
                    avg_correlation = rolling_corr.dropna().mean()
                    recent_correlation = rolling_corr.dropna().iloc[-10:].mean()  # Last 10 days
                    
                    if (abs(avg_correlation) >= min_correlation and 
                        abs(recent_correlation) >= min_correlation * 0.8):  # Allow some degradation
                        pair_key = (symbol1, symbol2)
                        qualifying_pairs[pair_key] = {
                            'avg_correlation': avg_correlation,
                            'recent_correlation': recent_correlation,
                            'correlation_stability': rolling_corr.std(),
                            'observations': len(aligned_data)
                        }
                        
                except Exception as e:
                    logger.warning(f"Correlation calculation failed for {symbol1}-{symbol2}: {e}")
        
        logger.info(f"Correlation screening: {len(qualifying_pairs)} pairs qualify")
        return qualifying_pairs
    
    def test_stationarity(self, 
                         price_data: pd.DataFrame,
                         pair_candidates: Dict) -> Dict[Tuple[str, str], Dict]:
        """
        Step 2: Apply stationarity tests (Johansen, variance-ratio)
        
        Args:
            price_data: Historical price data
            pair_candidates: Pairs from correlation screening
            
        Returns:
            Dictionary of pairs passing stationarity tests
        """
        logger.info(f"Testing stationarity for {len(pair_candidates)} pairs")
        
        stationary_pairs = {}
        
        for pair_key, corr_stats in pair_candidates.items():
            symbol1, symbol2 = pair_key
            
            try:
                # Get aligned log price data
                series1 = np.log(price_data[symbol1]['Close'].dropna())
                series2 = np.log(price_data[symbol2]['Close'].dropna())
                
                aligned_data = pd.concat([series1, series2], axis=1).dropna()
                if len(aligned_data) < 100:  # Need sufficient data
                    continue
                
                aligned_data.columns = [symbol1, symbol2]
                
                # Johansen cointegration test
                johansen_result = coint_johansen(aligned_data.values, det_order=0, k_ar_diff=1)
                trace_stat = johansen_result.lr1[0]  # Trace statistic
                critical_value = johansen_result.cvt[0, 1]  # 5% critical value
                johansen_p_value = 1.0 if trace_stat < critical_value else 0.001  # Approximation
                
                johansen_passed = johansen_p_value < config.JOHANSEN_P_VALUE
                
                # Simple spread for additional tests
                # Use OLS beta for initial spread calculation
                X = aligned_data[symbol2].values.reshape(-1, 1)
                y = aligned_data[symbol1].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                spread = y - beta * aligned_data[symbol2].values
                
                # Variance ratio test for random walk rejection
                vr_statistic, vr_p_value = variance_ratio_test(spread, lags=[2, 4, 8])
                variance_ratio_passed = vr_p_value < config.VARIANCE_RATIO_P_VALUE
                
                # Augmented Dickey-Fuller test for unit root
                adf_statistic, adf_p_value = adfuller(spread)[:2]
                adf_passed = adf_p_value < 0.05  # Standard 5% threshold
                
                # Combined stationarity check
                stationarity_passed = (johansen_passed and 
                                     variance_ratio_passed and 
                                     adf_passed)
                
                if stationarity_passed:
                    stationarity_stats = {
                        'johansen_trace': trace_stat,
                        'johansen_critical': critical_value,
                        'johansen_passed': johansen_passed,
                        'variance_ratio_stat': vr_statistic,
                        'variance_ratio_p': vr_p_value,
                        'variance_ratio_passed': variance_ratio_passed,
                        'adf_statistic': adf_statistic,
                        'adf_p_value': adf_p_value,
                        'adf_passed': adf_passed,
                        'ols_beta': beta,
                        'spread_mean': np.mean(spread),
                        'spread_std': np.std(spread)
                    }
                    
                    # Combine with correlation stats
                    combined_stats = {**corr_stats, **stationarity_stats}
                    stationary_pairs[pair_key] = combined_stats
                    
            except Exception as e:
                logger.warning(f"Stationarity test failed for {pair_key}: {e}")
        
        logger.info(f"Stationarity tests: {len(stationary_pairs)} pairs passed")
        return stationary_pairs
    
    def fit_state_space_models(self, 
                              price_data: pd.DataFrame,
                              stationary_pairs: Dict) -> Dict[Tuple[str, str], Dict]:
        """
        Step 3: Fit Kalman filter for time-varying beta estimation
        
        Args:
            price_data: Historical price data
            stationary_pairs: Pairs passing stationarity tests
            
        Returns:
            Dictionary of pairs with Kalman filter results
        """
        logger.info(f"Fitting state-space models for {len(stationary_pairs)} pairs")
        
        from pykalman import KalmanFilter
        
        kalman_pairs = {}
        
        for pair_key, stats in stationary_pairs.items():
            symbol1, symbol2 = pair_key
            
            try:
                # Get aligned data
                series1 = np.log(price_data[symbol1]['Close'].dropna())
                series2 = np.log(price_data[symbol2]['Close'].dropna())
                
                aligned_data = pd.concat([series1, series2], axis=1).dropna()
                aligned_data.columns = [symbol1, symbol2]
                
                y = aligned_data[symbol1].values
                x = aligned_data[symbol2].values.reshape(-1, 1)
                
                # Set up Kalman filter for time-varying beta
                # State: [beta_t, alpha_t]
                # Observation: y_t = alpha_t + beta_t * x_t + noise
                
                transition_matrices = np.array([[1, 0], [0, 1]])  # Random walk for beta and alpha
                observation_matrices = np.column_stack([x, np.ones(len(x))])  # [x_t, 1]
                
                # Initialize Kalman filter
                kf = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                    transition_covariance=np.eye(2) * config.KALMAN_TRANSITION_COV,
                    observation_covariance=config.KALMAN_OBSERVATION_COV,
                    initial_state_mean=[stats['ols_beta'], 0],
                    initial_state_covariance=np.eye(2) * 1e-3
                )
                
                # Fit the model
                state_means, state_covariances = kf.em(y, n_iter=10).smooth()[0:2]
                
                # Extract time-varying parameters
                time_varying_betas = state_means[:, 0]
                time_varying_alphas = state_means[:, 1]
                
                # Calculate Kalman-smoothed spread
                kalman_spread = y - time_varying_betas * x.flatten() - time_varying_alphas
                
                # Calculate spread statistics
                spread_mean = np.mean(kalman_spread)
                spread_std = np.std(kalman_spread)
                
                kalman_stats = {
                    'kalman_betas': time_varying_betas,
                    'kalman_alphas': time_varying_alphas,
                    'kalman_spread': kalman_spread,
                    'kalman_spread_mean': spread_mean,
                    'kalman_spread_std': spread_std,
                    'beta_stability': np.std(time_varying_betas),
                    'final_beta': time_varying_betas[-1],
                    'beta_trend': np.polyfit(range(len(time_varying_betas)), time_varying_betas, 1)[0]
                }
                
                # Combine with previous stats
                combined_stats = {**stats, **kalman_stats}
                kalman_pairs[pair_key] = combined_stats
                
            except Exception as e:
                logger.warning(f"Kalman filter fitting failed for {pair_key}: {e}")
        
        logger.info(f"State-space modeling: {len(kalman_pairs)} pairs successful")
        return kalman_pairs
    
    def analyze_mean_reversion(self, 
                              kalman_pairs: Dict) -> Dict[Tuple[str, str], Dict]:
        """
        Step 4: Analyze half-life and Hurst exponent
        
        Args:
            kalman_pairs: Pairs with Kalman filter results
            
        Returns:
            Dictionary of pairs passing mean-reversion criteria
        """
        logger.info(f"Analyzing mean-reversion for {len(kalman_pairs)} pairs")
        
        mean_reverting_pairs = {}
        
        for pair_key, stats in kalman_pairs.items():
            try:
                kalman_spread = stats['kalman_spread']
                
                # Calculate half-life of mean reversion
                half_life = calculate_half_life(kalman_spread)
                
                # Calculate Hurst exponent
                hurst_exponent = calculate_hurst_exponent(kalman_spread)
                
                # Apply criteria
                half_life_passed = half_life <= config.MAX_HALF_LIFE
                hurst_passed = hurst_exponent <= config.MAX_HURST
                
                if half_life_passed and hurst_passed:
                    reversion_stats = {
                        'half_life': half_life,
                        'hurst_exponent': hurst_exponent,
                        'half_life_passed': half_life_passed,
                        'hurst_passed': hurst_passed
                    }
                    
                    # Combine with previous stats
                    combined_stats = {**stats, **reversion_stats}
                    mean_reverting_pairs[pair_key] = combined_stats
                    
            except Exception as e:
                logger.warning(f"Mean-reversion analysis failed for {pair_key}: {e}")
        
        logger.info(f"Mean-reversion analysis: {len(mean_reverting_pairs)} pairs qualify")
        return mean_reverting_pairs
    
    def stress_test_capacity(self, 
                           price_data: pd.DataFrame,
                           mean_reverting_pairs: Dict,
                           target_nav: float = 100_000_000) -> Dict[Tuple[str, str], Dict]:
        """
        Step 5: Capacity stress testing
        
        Args:
            price_data: Historical price data
            mean_reverting_pairs: Pairs passing mean-reversion tests
            target_nav: Target NAV for capacity calculation
            
        Returns:
            Dictionary of pairs passing capacity tests
        """
        logger.info(f"Stress testing capacity for {len(mean_reverting_pairs)} pairs")
        
        capacity_qualified_pairs = {}
        
        for pair_key, stats in mean_reverting_pairs.items():
            symbol1, symbol2 = pair_key
            
            try:
                # Get volume data
                volume1 = price_data[symbol1]['Volume_USD'].dropna()
                volume2 = price_data[symbol2]['Volume_USD'].dropna()
                
                # Calculate average daily volumes
                avg_volume1 = volume1.rolling(30).mean().iloc[-1]
                avg_volume2 = volume2.rolling(30).mean().iloc[-1]
                
                # Estimate position sizes based on Kelly criterion (simplified)
                spread_std = stats['kalman_spread_std']
                expected_return = abs(stats['kalman_spread_mean'])  # Assuming mean-reversion
                
                # Kelly fraction (simplified)
                kelly_fraction = expected_return / (spread_std ** 2) if spread_std > 0 else 0
                kelly_fraction = min(kelly_fraction, config.MAX_POSITION_SIZE)  # Cap at 20%
                
                # Estimate required position size
                position_value = target_nav * kelly_fraction
                
                # Estimate daily turnover (based on half-life)
                half_life = stats['half_life']
                daily_turnover_rate = 1 / max(half_life, 1)  # Fraction of position turned over daily
                daily_turnover_value = position_value * daily_turnover_rate
                
                # Calculate market impact approximation
                # Impact = (turnover / volume) * price_impact_coefficient
                impact_coeff = 0.1  # 10% impact coefficient (institutional assumption)
                
                impact1 = (daily_turnover_value / 2) / avg_volume1 * impact_coeff
                impact2 = (daily_turnover_value / 2) / avg_volume2 * impact_coeff
                total_impact = impact1 + impact2
                
                # Impact × turnover as percentage of NAV
                impact_turnover_ratio = total_impact * daily_turnover_value / target_nav
                
                # Apply capacity constraint
                capacity_passed = impact_turnover_ratio <= config.MAX_IMPACT_TURNOVER
                
                if capacity_passed:
                    capacity_stats = {
                        'kelly_fraction': kelly_fraction,
                        'position_value': position_value,
                        'daily_turnover_rate': daily_turnover_rate,
                        'daily_turnover_value': daily_turnover_value,
                        'market_impact': total_impact,
                        'impact_turnover_ratio': impact_turnover_ratio,
                        'capacity_passed': capacity_passed,
                        'avg_volume1': avg_volume1,
                        'avg_volume2': avg_volume2,
                        'capacity_utilization': 'High' if kelly_fraction > 0.15 else 
                                              'Medium' if kelly_fraction > 0.05 else 'Low'
                    }
                    
                    # Combine with previous stats
                    combined_stats = {**stats, **capacity_stats}
                    capacity_qualified_pairs[pair_key] = combined_stats
                    
            except Exception as e:
                logger.warning(f"Capacity stress test failed for {pair_key}: {e}")
        
        logger.info(f"Capacity stress testing: {len(capacity_qualified_pairs)} pairs qualify")
        return capacity_qualified_pairs
    
    def discover_pairs(self, 
                      price_data: pd.DataFrame,
                      universe_symbols: List[str],
                      max_workers: int = 4) -> Dict[Tuple[str, str], Dict]:
        """
        Run the complete pair discovery pipeline
        
        Args:
            price_data: Historical price data
            universe_symbols: Filtered universe symbols
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary of discovered pairs with full statistics
        """
        logger.info(f"Starting pair discovery pipeline for {len(universe_symbols)} symbols")
        
        # Step 1: Correlation screening
        correlation_pairs = self.screen_correlations(price_data, universe_symbols)
        
        if not correlation_pairs:
            logger.warning("No pairs passed correlation screening")
            return {}
        
        # Step 2: Stationarity testing
        stationary_pairs = self.test_stationarity(price_data, correlation_pairs)
        
        if not stationary_pairs:
            logger.warning("No pairs passed stationarity tests")
            return {}
        
        # Step 3: State-space modeling
        kalman_pairs = self.fit_state_space_models(price_data, stationary_pairs)
        
        if not kalman_pairs:
            logger.warning("No pairs passed state-space modeling")
            return {}
        
        # Step 4: Mean-reversion analysis
        mean_reverting_pairs = self.analyze_mean_reversion(kalman_pairs)
        
        if not mean_reverting_pairs:
            logger.warning("No pairs passed mean-reversion criteria")
            return {}
        
        # Step 5: Capacity stress testing
        final_pairs = self.stress_test_capacity(price_data, mean_reverting_pairs)
        
        # Store discovery statistics
        self.discovery_stats = {
            'initial_symbols': len(universe_symbols),
            'max_possible_pairs': len(universe_symbols) * (len(universe_symbols) - 1) // 2,
            'correlation_pairs': len(correlation_pairs),
            'stationary_pairs': len(stationary_pairs),
            'kalman_pairs': len(kalman_pairs),
            'mean_reverting_pairs': len(mean_reverting_pairs),
            'final_pairs': len(final_pairs),
            'correlation_retention': len(correlation_pairs) / (len(universe_symbols) * (len(universe_symbols) - 1) // 2),
            'stationarity_retention': len(stationary_pairs) / len(correlation_pairs) if correlation_pairs else 0,
            'kalman_retention': len(kalman_pairs) / len(stationary_pairs) if stationary_pairs else 0,
            'reversion_retention': len(mean_reverting_pairs) / len(kalman_pairs) if kalman_pairs else 0,
            'capacity_retention': len(final_pairs) / len(mean_reverting_pairs) if mean_reverting_pairs else 0
        }
        
        logger.info(f"Pair discovery complete: {len(final_pairs)} pairs discovered")
        logger.info(f"Discovery funnel: {len(universe_symbols)} symbols → "
                   f"{len(correlation_pairs)} correlation → "
                   f"{len(stationary_pairs)} stationary → " 
                   f"{len(kalman_pairs)} kalman → "
                   f"{len(mean_reverting_pairs)} mean-reverting → "
                   f"{len(final_pairs)} final")
        
        self.discovered_pairs = final_pairs
        return final_pairs
    
    def get_discovery_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of discovered pairs
        
        Returns:
            DataFrame with pair statistics
        """
        if not self.discovered_pairs:
            return pd.DataFrame()
        
        summary_data = []
        
        for pair_key, stats in self.discovered_pairs.items():
            symbol1, symbol2 = pair_key
            
            summary_row = {
                'symbol_1': symbol1,
                'symbol_2': symbol2,
                'avg_correlation': stats.get('avg_correlation', np.nan),
                'half_life_days': stats.get('half_life', np.nan),
                'hurst_exponent': stats.get('hurst_exponent', np.nan),
                'kelly_fraction': stats.get('kelly_fraction', np.nan),
                'capacity_utilization': stats.get('capacity_utilization', 'Unknown'),
                'final_beta': stats.get('final_beta', np.nan),
                'spread_std': stats.get('kalman_spread_std', np.nan),
                'johansen_trace': stats.get('johansen_trace', np.nan),
                'impact_turnover_ratio': stats.get('impact_turnover_ratio', np.nan)
            }
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)


# Global instance
pair_discovery = PairDiscovery() 