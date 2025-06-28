"""
DOAC System Demonstration
Simplified demo showing key functionality without external data dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from config.settings import config
from utils.statistics import calculate_half_life, calculate_hurst_exponent, calculate_ou_parameters
from signals.ou_model import OUProcess
from models.meta_filter import FeatureEngineering


def generate_synthetic_data(n_days=1000, n_symbols=2):
    """Generate synthetic price data for demonstration"""
    logger.info(f"Generating synthetic data: {n_days} days, {n_symbols} symbols")
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate correlated random walks
    np.random.seed(42)  # For reproducibility
    
    # Base return process
    base_returns = np.random.normal(0.0005, 0.02, n_days)  # 12.5% annual return, 32% vol
    
    # Create correlated symbols
    symbols = [f'STOCK_{i}' for i in range(n_symbols)]
    price_data = {}
    
    for i, symbol in enumerate(symbols):
        # Add some idiosyncratic noise
        idiosyncratic = np.random.normal(0, 0.01, n_days)
        correlation = 0.7 + 0.2 * np.random.random()  # 70-90% correlation
        
        symbol_returns = base_returns * correlation + idiosyncratic * (1 - correlation)
        
        # Convert to price levels
        prices = 100 * np.exp(np.cumsum(symbol_returns))
        volumes = np.random.lognormal(15, 0.5, n_days)  # Log-normal volume
        
        price_data[symbol] = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Volume': volumes,
            'Returns': symbol_returns
        }).set_index('Date')
    
    return price_data


def demonstrate_spread_modeling():
    """Demonstrate OU process spread modeling"""
    logger.info("=== DEMONSTRATING SPREAD MODELING ===")
    
    # Generate synthetic pair data
    price_data = generate_synthetic_data(n_days=500, n_symbols=2)
    
    # Calculate spread
    prices1 = price_data['STOCK_0']['Close']
    prices2 = price_data['STOCK_1']['Close']
    
    # Log price spread with beta estimation
    log_prices1 = np.log(prices1)
    log_prices2 = np.log(prices2)
    
    # OLS regression for hedge ratio
    X = log_prices2.values.reshape(-1, 1)
    y = log_prices1.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
    
    spread = y - beta * log_prices2.values
    
    logger.info(f"Calculated spread with beta = {beta:.4f}")
    
    # Analyze spread properties
    half_life = calculate_half_life(spread)
    hurst = calculate_hurst_exponent(spread)
    theta, mu, sigma = calculate_ou_parameters(spread)
    
    logger.info(f"Spread Statistics:")
    logger.info(f"  Half-life: {half_life:.2f} days")
    logger.info(f"  Hurst exponent: {hurst:.3f}")
    logger.info(f"  OU parameters: θ={theta:.4f}, μ={mu:.4f}, σ={sigma:.4f}")
    
    # Fit OU model
    ou_model = OUProcess()
    fit_result = ou_model.fit(spread, method='mle')
    
    if fit_result.get('success', False):
        logger.info(f"OU Model fitted successfully:")
        logger.info(f"  θ={ou_model.theta:.4f}, μ={ou_model.mu:.4f}, σ={ou_model.sigma:.4f}")
        logger.info(f"  Half-life: {fit_result.get('half_life', 0):.2f} days")
        
        # Generate z-scores
        z_scores = ou_model.calculate_z_score(spread)
        
        # Count signals
        entry_signals = np.sum(np.abs(z_scores) >= config.ENTRY_Z_THRESHOLD)
        exit_signals = np.sum(np.abs(z_scores) <= config.EXIT_Z_THRESHOLD)
        
        logger.info(f"Signal Analysis:")
        logger.info(f"  Entry signals (|z| >= {config.ENTRY_Z_THRESHOLD}): {entry_signals}")
        logger.info(f"  Exit signals (|z| <= {config.EXIT_Z_THRESHOLD}): {exit_signals}")
        logger.info(f"  Max z-score: {np.max(np.abs(z_scores)):.2f}")
        
        return {
            'spread': spread,
            'z_scores': z_scores,
            'ou_model': ou_model,
            'beta': beta,
            'prices': (prices1, prices2)
        }
    
    return None


def demonstrate_feature_engineering():
    """Demonstrate meta-filter feature engineering"""
    logger.info("\n=== DEMONSTRATING FEATURE ENGINEERING ===")
    
    # Generate synthetic data
    price_data = generate_synthetic_data(n_days=200, n_symbols=2)
    
    # Create spread data
    log_prices1 = np.log(price_data['STOCK_0']['Close'])
    log_prices2 = np.log(price_data['STOCK_1']['Close'])
    spread_data = log_prices1 - log_prices2
    
    # Sample signal data
    signal_data = {
        'z_score': 2.5,
        'confidence': 0.75,
        'kelly_fraction': 0.08,
        'ou_theta': 0.15,
        'ou_mu': 0.02,
        'ou_sigma': 0.25,
        'entry_signal': 1,
        'prediction_std': 0.1
    }
    
    # Create feature engineer
    feature_engineer = FeatureEngineering()
    
    # Engineer features
    features_df = feature_engineer.create_features(
        price_data=None,  # Simplified for demo
        spread_data=spread_data,
        signal_data=signal_data,
        market_data={'vix': 18.5, 'risk_free_rate': 0.025}
    )
    
    logger.info(f"Feature Engineering Results:")
    logger.info(f"  Total features generated: {len(features_df.columns)}")
    logger.info(f"  Feature categories:")
    
    # Categorize features
    categories = {
        'spread': [col for col in features_df.columns if 'spread' in col.lower()],
        'technical': [col for col in features_df.columns if any(x in col.lower() for x in ['ma_', 'bb_', 'rsi', 'momentum'])],
        'signal': [col for col in features_df.columns if any(x in col.lower() for x in ['z_score', 'confidence', 'kelly'])],
        'volatility': [col for col in features_df.columns if 'vol' in col.lower()],
        'interaction': [col for col in features_df.columns if 'interaction' in col.lower()]
    }
    
    for category, features in categories.items():
        logger.info(f"    {category.capitalize()}: {len(features)} features")
    
    # Show sample feature values
    logger.info(f"Sample feature values:")
    for col in list(features_df.columns)[:10]:
        value = features_df[col].iloc[0]
        logger.info(f"    {col}: {value:.4f}")
    
    return features_df


def demonstrate_simple_backtest():
    """Demonstrate simplified backtesting"""
    logger.info("\n=== DEMONSTRATING SIMPLE BACKTEST ===")
    
    # Generate longer time series for backtest
    price_data = generate_synthetic_data(n_days=1000, n_symbols=2)
    
    # Calculate spread and signals
    prices1 = price_data['STOCK_0']['Close']
    prices2 = price_data['STOCK_1']['Close']
    
    log_prices1 = np.log(prices1)
    log_prices2 = np.log(prices2)
    
    # Rolling beta estimation (simplified)
    window = 60
    spread_series = []
    dates = []
    
    for i in range(window, len(log_prices1)):
        # Estimate beta over rolling window
        y_window = log_prices1.iloc[i-window:i].values
        x_window = log_prices2.iloc[i-window:i].values.reshape(-1, 1)
        
        beta = np.linalg.lstsq(x_window, y_window, rcond=None)[0][0]
        spread = log_prices1.iloc[i] - beta * log_prices2.iloc[i]
        
        spread_series.append(spread)
        dates.append(prices1.index[i])
    
    spread_df = pd.Series(spread_series, index=dates)
    
    # Simple mean-reversion signals
    rolling_mean = spread_df.rolling(20).mean()
    rolling_std = spread_df.rolling(20).std()
    z_scores = (spread_df - rolling_mean) / rolling_std
    
    # Generate trades
    positions = pd.Series(0, index=z_scores.index)
    entry_threshold = 2.0
    exit_threshold = 0.5
    
    current_position = 0
    trades = []
    
    for date, z_score in z_scores.items():
        if current_position == 0:  # No position
            if z_score > entry_threshold:
                current_position = -1  # Sell spread (expect reversion)
                positions[date] = current_position
            elif z_score < -entry_threshold:
                current_position = 1   # Buy spread
                positions[date] = current_position
        else:  # Have position
            if abs(z_score) < exit_threshold:
                # Exit position
                current_position = 0
                positions[date] = current_position
    
    # Calculate P&L
    position_changes = positions.diff().fillna(0)
    spread_returns = spread_df.diff()
    
    # P&L = -position_change * subsequent_spread_change (simplified)
    pnl_series = pd.Series(0.0, index=spread_df.index)
    
    for i in range(1, len(spread_df)):
        if positions.iloc[i-1] != 0:
            # P&L from holding position
            pnl_series.iloc[i] = -positions.iloc[i-1] * spread_returns.iloc[i] * 1000  # Scale for readability
    
    # Calculate cumulative P&L
    cumulative_pnl = pnl_series.cumsum()
    
    # Calculate metrics
    total_return = cumulative_pnl.iloc[-1]
    winning_days = (pnl_series > 0).sum()
    losing_days = (pnl_series < 0).sum()
    hit_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
    
    # Volatility and Sharpe
    pnl_vol = pnl_series.std() * np.sqrt(252)
    avg_daily_pnl = pnl_series.mean()
    sharpe_approx = avg_daily_pnl * np.sqrt(252) / (pnl_vol + 1e-6)
    
    logger.info(f"Simple Backtest Results:")
    logger.info(f"  Total P&L: {total_return:.2f}")
    logger.info(f"  Hit Rate: {hit_rate:.2%}")
    logger.info(f"  Winning Days: {winning_days}")
    logger.info(f"  Losing Days: {losing_days}")
    logger.info(f"  Approx Sharpe: {sharpe_approx:.2f}")
    logger.info(f"  Max Drawdown: {(cumulative_pnl - cumulative_pnl.expanding().max()).min():.2f}")
    
    # Number of trades
    trade_count = (position_changes != 0).sum()
    logger.info(f"  Number of trades: {trade_count}")
    
    return {
        'pnl_series': pnl_series,
        'cumulative_pnl': cumulative_pnl,
        'positions': positions,
        'z_scores': z_scores,
        'hit_rate': hit_rate,
        'sharpe': sharpe_approx
    }


def create_visualization(spread_data=None, backtest_data=None):
    """Create simple visualizations"""
    try:
        import matplotlib.pyplot as plt
        
        if spread_data is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('DOAC System - Spread Analysis Demo', fontsize=14)
            
            # Plot 1: Price series
            axes[0, 0].plot(spread_data['prices'][0].index, spread_data['prices'][0], label='Stock 1', alpha=0.8)
            axes[0, 0].plot(spread_data['prices'][1].index, spread_data['prices'][1], label='Stock 2', alpha=0.8)
            axes[0, 0].set_title('Synthetic Price Series')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Spread
            axes[0, 1].plot(spread_data['prices'][0].index, spread_data['spread'])
            axes[0, 1].set_title('Price Spread')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Z-scores
            axes[1, 0].plot(spread_data['prices'][0].index, spread_data['z_scores'])
            axes[1, 0].axhline(y=config.ENTRY_Z_THRESHOLD, color='r', linestyle='--', label='Entry threshold')
            axes[1, 0].axhline(y=-config.ENTRY_Z_THRESHOLD, color='r', linestyle='--')
            axes[1, 0].axhline(y=config.EXIT_Z_THRESHOLD, color='g', linestyle='--', label='Exit threshold')
            axes[1, 0].axhline(y=-config.EXIT_Z_THRESHOLD, color='g', linestyle='--')
            axes[1, 0].set_title('Z-Scores and Trading Thresholds')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Signal distribution
            axes[1, 1].hist(spread_data['z_scores'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=config.ENTRY_Z_THRESHOLD, color='r', linestyle='--')
            axes[1, 1].axvline(x=-config.ENTRY_Z_THRESHOLD, color='r', linestyle='--')
            axes[1, 1].set_title('Z-Score Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('doac_spread_analysis.png', dpi=150, bbox_inches='tight')
            logger.info("Spread analysis chart saved as 'doac_spread_analysis.png'")
        
        if backtest_data is not None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('DOAC System - Simple Backtest Results', fontsize=14)
            
            # Plot 1: Cumulative P&L
            axes[0].plot(backtest_data['cumulative_pnl'].index, backtest_data['cumulative_pnl'])
            axes[0].set_title('Cumulative P&L')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylabel('P&L')
            
            # Plot 2: Positions and Z-scores
            ax2 = axes[1]
            ax3 = ax2.twinx()
            
            ax2.plot(backtest_data['z_scores'].index, backtest_data['z_scores'], alpha=0.7, label='Z-Score')
            ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Z-Score')
            ax2.legend(loc='upper left')
            
            ax3.plot(backtest_data['positions'].index, backtest_data['positions'], 'ro-', markersize=2, alpha=0.6, label='Position')
            ax3.set_ylabel('Position')
            ax3.legend(loc='upper right')
            
            axes[1].set_title('Trading Signals and Positions')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('doac_backtest_results.png', dpi=150, bbox_inches='tight')
            logger.info("Backtest results chart saved as 'doac_backtest_results.png'")
            
    except ImportError:
        logger.warning("Matplotlib not available - skipping visualizations")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("DOAC SYSTEM DEMONSTRATION")
    print("Δ-Ω Adaptive Convergence Strategy")
    print("=" * 80)
    
    logger.info("Starting DOAC system demonstration...")
    
    try:
        # Demonstrate spread modeling
        spread_data = demonstrate_spread_modeling()
        
        # Demonstrate feature engineering
        features_df = demonstrate_feature_engineering()
        
        # Demonstrate simple backtesting
        backtest_data = demonstrate_simple_backtest()
        
        # Create visualizations
        logger.info("\nGenerating visualizations...")
        create_visualization(spread_data, backtest_data)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("DEMONSTRATION SUMMARY")
        logger.info("=" * 50)
        
        if spread_data:
            logger.info(f"✅ Spread Modeling: OU process fitted successfully")
            logger.info(f"   Half-life: {spread_data['ou_model'].get_equilibrium_stats().get('half_life', 0):.2f} days")
        
        if features_df is not None:
            logger.info(f"✅ Feature Engineering: {len(features_df.columns)} features generated")
        
        if backtest_data:
            logger.info(f"✅ Simple Backtest: Sharpe ≈ {backtest_data['sharpe']:.2f}, Hit Rate = {backtest_data['hit_rate']:.1%}")
        
        logger.info("\nNext Steps:")
        logger.info("1. Run 'python main.py' for full system demonstration")
        logger.info("2. Configure API keys in env_template.txt")
        logger.info("3. Customize parameters in config/settings.py")
        logger.info("4. Deploy with real market data")
        
        logger.info("\n🎯 DOAC System demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 