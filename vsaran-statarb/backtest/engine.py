"""
Backtesting Engine for the DOAC system
Implements comprehensive backtesting with walk-forward analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

from config.settings import config
from data.fetcher import data_fetcher
from universe.filters import universe_filter
from signals.pair_discovery import pair_discovery
from signals.ou_model import signal_generator
from models.rl_agent import rl_agent
from models.meta_filter import meta_filter
from utils.statistics import calculate_sharpe_ratio, calculate_maximum_drawdown, calculate_var

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Trade record for backtesting"""
    pair: Tuple[str, str]
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_signal: int
    entry_spread: float
    exit_spread: Optional[float]
    position_size: float
    pnl: float
    holding_period: int
    exit_reason: str
    signal_data: Dict
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pair': f"{self.pair[0]}/{self.pair[1]}",
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_signal': self.entry_signal,
            'entry_spread': self.entry_spread,
            'exit_spread': self.exit_spread,
            'position_size': self.position_size,
            'pnl': self.pnl,
            'holding_period': self.holding_period,
            'exit_reason': self.exit_reason,
            'z_score': self.signal_data.get('z_score', 0),
            'confidence': self.signal_data.get('confidence', 0)
        }


class Portfolio:
    """Portfolio state management for backtesting"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # pair -> position info
        self.nav_history = []
        self.trade_history = []
        self.daily_returns = []
        
    def update_nav(self, date: pd.Timestamp, market_values: Dict[Tuple[str, str], float]):
        """Update NAV based on current market values"""
        total_position_value = sum(market_values.get(pair, 0) for pair in self.positions.keys())
        nav = self.cash + total_position_value
        
        self.nav_history.append({
            'date': date,
            'nav': nav,
            'cash': self.cash,
            'positions_value': total_position_value,
            'num_positions': len(self.positions)
        })
        
        # Calculate daily return
        if len(self.nav_history) > 1:
            prev_nav = self.nav_history[-2]['nav']
            daily_return = (nav - prev_nav) / prev_nav
            self.daily_returns.append(daily_return)
        
        return nav
    
    def open_position(self, trade: Trade):
        """Open a new position"""
        self.positions[trade.pair] = {
            'trade': trade,
            'entry_date': trade.entry_date,
            'position_size': trade.position_size,
            'entry_spread': trade.entry_spread
        }
        
        # Reduce cash by position notional (simplified)
        position_notional = abs(trade.position_size) * self.initial_capital
        self.cash -= position_notional * config.DEFAULT_SLIPPAGE  # Transaction costs
    
    def close_position(self, pair: Tuple[str, str], exit_date: pd.Timestamp, 
                      exit_spread: float, exit_reason: str) -> Optional[Trade]:
        """Close an existing position"""
        if pair not in self.positions:
            return None
        
        position = self.positions[pair]
        trade = position['trade']
        
        # Calculate P&L
        spread_change = exit_spread - trade.entry_spread
        pnl = trade.entry_signal * spread_change * abs(trade.position_size) * self.initial_capital
        
        # Apply transaction costs
        pnl -= abs(trade.position_size) * self.initial_capital * config.DEFAULT_SLIPPAGE
        
        # Update trade record
        trade.exit_date = exit_date
        trade.exit_spread = exit_spread
        trade.pnl = pnl
        trade.holding_period = (exit_date - trade.entry_date).days
        trade.exit_reason = exit_reason
        
        # Update cash
        self.cash += pnl
        
        # Remove position
        del self.positions[pair]
        
        # Add to trade history
        self.trade_history.append(trade)
        
        return trade
    
    def get_current_positions(self) -> Dict[Tuple[str, str], Dict]:
        """Get current open positions"""
        return self.positions.copy()
    
    def get_nav_series(self) -> pd.Series:
        """Get NAV time series"""
        if not self.nav_history:
            return pd.Series()
        
        nav_df = pd.DataFrame(self.nav_history)
        nav_df.set_index('date', inplace=True)
        return nav_df['nav']


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self):
        self.results = {}
        self.portfolio = None
        
    def run_backtest(self, 
                    start_date: str,
                    end_date: str,
                    initial_capital: float = 10_000_000,
                    rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
        """
        Run comprehensive backtest as per blueprint example
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date  
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance universe
            
        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize portfolio
        self.portfolio = Portfolio(initial_capital)
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        all_symbols = config.data_config.SP500_TICKERS + config.data_config.LIQUID_ETFS
        
        price_data = data_fetcher.fetch_price_data(
            symbols=all_symbols[:50],  # Limit for demo
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data.empty:
            return {'success': False, 'error': 'No price data available'}
        
        # Create date range for backtesting
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = [d for d in date_range if d in price_data.index]
        
        # Initialize trading state
        active_pairs = {}
        last_rebalance = None
        
        logger.info(f"Running backtest over {len(date_range)} trading days")
        
        for i, current_date in enumerate(date_range):
            try:
                # Check if we need to rebalance universe
                if (last_rebalance is None or 
                    self._should_rebalance(current_date, last_rebalance, rebalance_frequency)):
                    
                    logger.info(f"Rebalancing universe on {current_date.date()}")
                    active_pairs = self._rebalance_universe(price_data, current_date)
                    last_rebalance = current_date
                
                # Update signals for active pairs
                self._update_signals(active_pairs, price_data, current_date)
                
                # Check for exits
                self._process_exits(active_pairs, current_date)
                
                # Check for entries
                self._process_entries(active_pairs, current_date)
                
                # Update portfolio NAV
                market_values = self._calculate_market_values(price_data, current_date)
                nav = self.portfolio.update_nav(current_date, market_values)
                
                # Progress logging
                if i % 50 == 0:
                    logger.info(f"Progress: {current_date.date()}, NAV: ${nav:,.0f}, "
                               f"Positions: {len(self.portfolio.positions)}")
                
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
                continue
        
        # Calculate final results
        results = self._calculate_performance_metrics()
        
        logger.info("Backtest completed")
        return results
    
    def _should_rebalance(self, current_date: pd.Timestamp, 
                         last_rebalance: pd.Timestamp,
                         frequency: str) -> bool:
        """Check if universe should be rebalanced"""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif frequency == 'monthly':
            return (current_date - last_rebalance).days >= 30
        return False
    
    def _rebalance_universe(self, price_data: pd.DataFrame, 
                           current_date: pd.Timestamp) -> Dict[Tuple[str, str], Dict]:
        """Rebalance universe and discover pairs"""
        try:
            # Get data up to current date
            historical_data = price_data.loc[:current_date]
            
            # Extract symbols with sufficient data
            symbols = []
            for symbol in price_data.columns.get_level_values(0).unique():
                symbol_data = historical_data[symbol]['Close'].dropna()
                if len(symbol_data) >= 100:  # Minimum data requirement
                    symbols.append(symbol)
            
            if len(symbols) < 10:
                logger.warning(f"Insufficient symbols for universe: {len(symbols)}")
                return {}
            
            # For demo, use example pairs from blueprint
            example_pairs = [('SPY', 'IVV'), ('KO', 'PEP')]
            
            active_pairs = {}
            for pair in example_pairs:
                if pair[0] in symbols and pair[1] in symbols:
                    # Calculate spread history
                    try:
                        series1 = np.log(historical_data[pair[0]]['Close'].dropna())
                        series2 = np.log(historical_data[pair[1]]['Close'].dropna())
                        
                        aligned_data = pd.concat([series1, series2], axis=1).dropna()
                        if len(aligned_data) > 60:
                            aligned_data.columns = [pair[0], pair[1]]
                            
                            # Simple OLS hedge ratio
                            X = aligned_data[pair[1]].values.reshape(-1, 1)
                            y = aligned_data[pair[0]].values
                            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                            
                            spread = y - beta * aligned_data[pair[1]].values
                            
                            # Initialize OU model
                            if signal_generator.initialize_pair(pair, spread):
                                active_pairs[pair] = {
                                    'beta': beta,
                                    'spread_history': spread,
                                    'last_signal': None,
                                    'signal_data': {}
                                }
                    
                    except Exception as e:
                        logger.warning(f"Failed to initialize pair {pair}: {e}")
            
            logger.info(f"Rebalanced universe: {len(active_pairs)} active pairs")
            return active_pairs
            
        except Exception as e:
            logger.error(f"Universe rebalancing failed: {e}")
            return {}
    
    def _update_signals(self, active_pairs: Dict, price_data: pd.DataFrame, 
                       current_date: pd.Timestamp):
        """Update signals for all active pairs"""
        for pair, pair_data in active_pairs.items():
            try:
                # Get current prices
                if (pair[0] in price_data.columns.get_level_values(0) and 
                    pair[1] in price_data.columns.get_level_values(0)):
                    
                    price1 = price_data.loc[current_date, (pair[0], 'Close')]
                    price2 = price_data.loc[current_date, (pair[1], 'Close')]
                    
                    if pd.notna(price1) and pd.notna(price2):
                        # Calculate current spread
                        current_spread = np.log(price1) - pair_data['beta'] * np.log(price2)
                        
                        # Generate signal
                        signal = signal_generator.generate_signals(pair, current_spread)
                        
                        if signal:
                            active_pairs[pair]['last_signal'] = signal
                            active_pairs[pair]['signal_data'] = signal
                            active_pairs[pair]['current_spread'] = current_spread
                
            except Exception as e:
                logger.warning(f"Signal update failed for {pair}: {e}")
    
    def _process_exits(self, active_pairs: Dict, current_date: pd.Timestamp):
        """Process exit signals"""
        current_positions = self.portfolio.get_current_positions()
        
        for pair in list(current_positions.keys()):
            try:
                position = current_positions[pair]
                trade = position['trade']
                
                should_exit = False
                exit_reason = 'unknown'
                
                # Check exit conditions
                if pair in active_pairs:
                    signal_data = active_pairs[pair].get('signal_data', {})
                    current_spread = active_pairs[pair].get('current_spread', 0)
                    
                    # Signal-based exit
                    if signal_data.get('exit_signal', 0) != 0:
                        should_exit = True
                        exit_reason = 'signal_exit'
                    
                    # Time-based exit
                    holding_period = (current_date - trade.entry_date).days
                    max_holding = signal_data.get('max_holding_period', config.MAX_HOLDING_PERIOD)
                    if holding_period >= max_holding:
                        should_exit = True
                        exit_reason = 'time_exit'
                    
                    # Risk-based exit (large adverse move)
                    z_score = signal_data.get('z_score', 0)
                    if abs(z_score) > config.KILL_SWITCH_Z_THRESHOLD:
                        should_exit = True
                        exit_reason = 'risk_exit'
                    
                    if should_exit:
                        self.portfolio.close_position(pair, current_date, current_spread, exit_reason)
                        logger.debug(f"Closed position {pair} on {current_date.date()}: {exit_reason}")
                
            except Exception as e:
                logger.warning(f"Exit processing failed for {pair}: {e}")
    
    def _process_entries(self, active_pairs: Dict, current_date: pd.Timestamp):
        """Process entry signals"""
        for pair, pair_data in active_pairs.items():
            try:
                # Skip if already have position
                if pair in self.portfolio.positions:
                    continue
                
                signal_data = pair_data.get('signal_data', {})
                entry_signal = signal_data.get('entry_signal', 0)
                
                if entry_signal != 0:
                    # Check meta-filter if trained
                    if meta_filter.is_trained:
                        # Create dummy data for meta-filter (would use real data in production)
                        spread_series = pd.Series(pair_data.get('spread_history', []))
                        
                        viability = meta_filter.predict_viability(
                            price_data=pd.DataFrame(),  # Simplified
                            spread_data=spread_series,
                            signal_data=signal_data
                        )
                        
                        if viability['win_probability'] < config.META_FILTER_THRESHOLD:
                            logger.debug(f"Meta-filter rejected entry for {pair}: "
                                       f"prob={viability['win_probability']:.3f}")
                            continue
                    
                    # Calculate position size
                    kelly_fraction = signal_data.get('kelly_fraction', 0.05)
                    confidence = signal_data.get('confidence', 0.5)
                    
                    # Risk-adjusted position size
                    position_size = kelly_fraction * confidence
                    position_size = min(position_size, config.MAX_POSITION_SIZE)
                    
                    # Create trade
                    trade = Trade(
                        pair=pair,
                        entry_date=current_date,
                        exit_date=None,
                        entry_signal=entry_signal,
                        entry_spread=pair_data.get('current_spread', 0),
                        exit_spread=None,
                        position_size=position_size,
                        pnl=0.0,
                        holding_period=0,
                        exit_reason='',
                        signal_data=signal_data
                    )
                    
                    # Open position
                    self.portfolio.open_position(trade)
                    logger.debug(f"Opened position {pair} on {current_date.date()}: "
                               f"signal={entry_signal}, size={position_size:.3f}")
                
            except Exception as e:
                logger.warning(f"Entry processing failed for {pair}: {e}")
    
    def _calculate_market_values(self, price_data: pd.DataFrame, 
                                current_date: pd.Timestamp) -> Dict[Tuple[str, str], float]:
        """Calculate current market value of positions"""
        market_values = {}
        
        for pair in self.portfolio.positions.keys():
            try:
                # Simplified: assume position value tracks spread changes
                # In practice, would calculate actual P&L
                market_values[pair] = 0.0  # Placeholder
            except Exception:
                market_values[pair] = 0.0
        
        return market_values
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        logger.info("Calculating performance metrics...")
        
        # Get NAV series
        nav_series = self.portfolio.get_nav_series()
        
        if len(nav_series) == 0:
            return {'success': False, 'error': 'No NAV data'}
        
        # Calculate returns
        returns = nav_series.pct_change().dropna()
        
        # Core metrics
        total_return = (nav_series.iloc[-1] - nav_series.iloc[0]) / nav_series.iloc[0]
        annual_return = (1 + total_return) ** (252 / len(nav_series)) - 1
        
        # Risk metrics
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_maximum_drawdown(nav_series)
        var_95 = calculate_var(returns, 0.95)
        
        # Trade statistics
        trades_df = pd.DataFrame([t.to_dict() for t in self.portfolio.trade_history])
        
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            hit_rate = len(winning_trades) / len(trades_df)
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean()
            avg_holding_period = trades_df['holding_period'].mean()
            
            # Profit factor
            total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = total_wins / (total_losses + 1e-6)
        else:
            hit_rate = 0
            avg_win = avg_loss = 0
            avg_holding_period = 0
            profit_factor = 0
        
        # Position statistics
        nav_df = pd.DataFrame(self.portfolio.nav_history)
        avg_num_positions = nav_df['num_positions'].mean()
        max_num_positions = nav_df['num_positions'].max()
        
        # Capacity estimate
        avg_daily_volume = nav_series.iloc[-1] * 0.01  # Simplified estimate
        estimated_capacity = avg_daily_volume * 100  # Rule of thumb
        
        results = {
            'success': True,
            'period': {
                'start_date': nav_series.index[0],
                'end_date': nav_series.index[-1],
                'trading_days': len(nav_series)
            },
            'returns': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95
            },
            'trades': {
                'total_trades': len(self.portfolio.trade_history),
                'hit_rate': hit_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_holding_period': avg_holding_period
            },
            'positions': {
                'avg_num_positions': avg_num_positions,
                'max_num_positions': max_num_positions
            },
            'capacity': {
                'estimated_capacity': estimated_capacity
            },
            'nav_series': nav_series,
            'trades_df': trades_df,
            'daily_returns': pd.Series(self.portfolio.daily_returns, index=nav_series.index[1:])
        }
        
        # Log key metrics
        logger.info(f"Backtest Results:")
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"  Hit Rate: {hit_rate:.2%}")
        logger.info(f"  Total Trades: {len(self.portfolio.trade_history)}")
        
        return results
    
    def run_walk_forward_analysis(self, 
                                 start_date: str,
                                 end_date: str,
                                 train_window: int = 252,
                                 test_window: int = 63) -> Dict[str, Any]:
        """
        Run walk-forward analysis for model validation
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            train_window: Training window size (days)
            test_window: Testing window size (days)
            
        Returns:
            Walk-forward analysis results
        """
        logger.info("Starting walk-forward analysis")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        results = []
        current_start = 0
        
        while current_start + train_window + test_window < len(date_range):
            train_start = date_range[current_start]
            train_end = date_range[current_start + train_window]
            test_start = date_range[current_start + train_window + 1]
            test_end = date_range[current_start + train_window + test_window]
            
            logger.info(f"Walk-forward window: train {train_start.date()} to {train_end.date()}, "
                       f"test {test_start.date()} to {test_end.date()}")
            
            try:
                # Run backtest for this window
                window_results = self.run_backtest(
                    start_date=test_start.strftime('%Y-%m-%d'),
                    end_date=test_end.strftime('%Y-%m-%d')
                )
                
                if window_results.get('success', False):
                    window_results['train_start'] = train_start
                    window_results['train_end'] = train_end
                    window_results['test_start'] = test_start
                    window_results['test_end'] = test_end
                    results.append(window_results)
                
            except Exception as e:
                logger.warning(f"Walk-forward window failed: {e}")
            
            current_start += test_window
        
        # Aggregate results
        if results:
            combined_metrics = self._aggregate_walk_forward_results(results)
            return {
                'success': True,
                'individual_windows': results,
                'aggregated_metrics': combined_metrics
            }
        else:
            return {'success': False, 'error': 'No successful walk-forward windows'}
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward results"""
        sharpe_ratios = [r['returns']['sharpe_ratio'] for r in results if r.get('success')]
        annual_returns = [r['returns']['annual_return'] for r in results if r.get('success')]
        max_drawdowns = [r['returns']['max_drawdown'] for r in results if r.get('success')]
        hit_rates = [r['trades']['hit_rate'] for r in results if r.get('success')]
        
        return {
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'avg_annual_return': np.mean(annual_returns),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_hit_rate': np.mean(hit_rates),
            'num_windows': len(results),
            'consistency_score': len([s for s in sharpe_ratios if s > 1.0]) / len(sharpe_ratios) if sharpe_ratios else 0
        }


# Global instance
backtest_engine = BacktestEngine() 