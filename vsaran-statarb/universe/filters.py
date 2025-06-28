"""
Universe Engineering for the DOAC system
Implements the three-layer filtering system:
1. Liquidity/Borrow constraints
2. Fundamental homogeneity 
3. Microstructure sanity checks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from config.settings import config, data_config
from data.fetcher import data_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniverseFilter:
    """Main universe filtering class implementing the three-layer system"""
    
    def __init__(self):
        self.sector_industry_map = data_fetcher.get_sector_industry_mapping()
        
    def apply_liquidity_filters(self, 
                               price_data: pd.DataFrame,
                               fundamental_data: pd.DataFrame,
                               liquidity_data: pd.DataFrame) -> List[str]:
        """
        Layer 1: Apply liquidity and borrowability filters
        
        Criteria:
        - 30-day ADV >= $25M
        - Market cap >= $2B  
        - Easy-to-Borrow flag (approximated)
        
        Args:
            price_data: Historical price data
            fundamental_data: Fundamental metrics
            liquidity_data: Liquidity metrics
            
        Returns:
            List of symbols passing liquidity filters
        """
        logger.info("Applying liquidity filters (Layer 1)")
        
        passed_symbols = []
        
        for _, row in fundamental_data.iterrows():
            symbol = row['symbol']
            
            # Check if we have liquidity data for this symbol
            liquidity_row = liquidity_data[liquidity_data['symbol'] == symbol]
            if liquidity_row.empty:
                continue
                
            liquidity_metrics = liquidity_row.iloc[0]
            
            # Filter 1: Average Daily Volume >= $25M
            adv_check = liquidity_metrics.get('adv_30d', 0) >= config.MIN_ADV
            
            # Filter 2: Market cap >= $2B
            market_cap_check = row.get('market_cap', 0) >= config.MIN_MARKET_CAP
            
            # Filter 3: Easy-to-Borrow approximation
            # Use combination of market cap, volume, and institutional ownership proxy
            float_ratio = row.get('float_shares', 0) / max(row.get('shares_outstanding', 1), 1)
            etb_check = (
                row.get('market_cap', 0) >= 1_000_000_000 and  # $1B+ generally ETB
                float_ratio > 0.7 and  # High float ratio
                liquidity_metrics.get('adv_30d', 0) >= 10_000_000  # $10M+ volume
            )
            
            # Combined liquidity check
            if adv_check and market_cap_check and etb_check:
                passed_symbols.append(symbol)
                logger.debug(f"{symbol} passed liquidity filters")
            else:
                logger.debug(f"{symbol} failed liquidity filters: ADV={adv_check}, "
                           f"MarketCap={market_cap_check}, ETB={etb_check}")
        
        logger.info(f"Liquidity filters: {len(passed_symbols)} symbols passed")
        return passed_symbols
    
    def apply_fundamental_homogeneity_filters(self, 
                                            symbols: List[str],
                                            fundamental_data: pd.DataFrame) -> List[str]:
        """
        Layer 2: Apply fundamental homogeneity filters
        
        Criteria:
        - Same GICS industry
        - < 15% divergence in ROIC and gross margin
        
        Args:
            symbols: Symbols that passed liquidity filters
            fundamental_data: Fundamental metrics
            
        Returns:
            List of symbols passing homogeneity filters
        """
        logger.info("Applying fundamental homogeneity filters (Layer 2)")
        
        # Filter fundamental data to only include symbols that passed Layer 1
        filtered_fundamentals = fundamental_data[
            fundamental_data['symbol'].isin(symbols)
        ].copy()
        
        if filtered_fundamentals.empty:
            return []
        
        # Group by industry for homogeneity analysis
        industry_groups = filtered_fundamentals.groupby('industry')
        
        passed_symbols = []
        
        for industry, group in industry_groups:
            if len(group) < 2:  # Need at least 2 stocks for pairs
                continue
                
            # Calculate ROIC and gross margin statistics for the industry
            roic_values = group['roic'].dropna()
            margin_values = group['gross_margin'].dropna()
            
            if len(roic_values) < 2 or len(margin_values) < 2:
                continue
            
            # Calculate industry medians
            industry_roic_median = roic_values.median()
            industry_margin_median = margin_values.median()
            
            # Filter stocks within acceptable divergence
            for _, row in group.iterrows():
                symbol = row['symbol']
                stock_roic = row.get('roic', np.nan)
                stock_margin = row.get('gross_margin', np.nan)
                
                if pd.isna(stock_roic) or pd.isna(stock_margin):
                    continue
                
                # Check ROIC divergence
                roic_divergence = abs(stock_roic - industry_roic_median) / (
                    abs(industry_roic_median) + 1e-6
                )
                
                # Check gross margin divergence  
                margin_divergence = abs(stock_margin - industry_margin_median) / (
                    abs(industry_margin_median) + 1e-6
                )
                
                # Apply divergence thresholds
                roic_check = roic_divergence <= config.MAX_ROIC_DIVERGENCE
                margin_check = margin_divergence <= config.MAX_GROSS_MARGIN_DIVERGENCE
                
                if roic_check and margin_check:
                    passed_symbols.append(symbol)
                    logger.debug(f"{symbol} passed homogeneity filters "
                               f"(Industry: {industry})")
                else:
                    logger.debug(f"{symbol} failed homogeneity filters: "
                               f"ROIC_div={roic_divergence:.3f}, "
                               f"Margin_div={margin_divergence:.3f}")
        
        logger.info(f"Homogeneity filters: {len(passed_symbols)} symbols passed")
        return passed_symbols
    
    def apply_microstructure_filters(self, 
                                   symbols: List[str],
                                   liquidity_data: pd.DataFrame) -> List[str]:
        """
        Layer 3: Apply microstructure sanity checks
        
        Criteria:
        - Median bid-ask spread < 5 bp
        - Quote update rate > 500 ms (approximated)
        
        Args:
            symbols: Symbols that passed homogeneity filters
            liquidity_data: Liquidity metrics
            
        Returns:
            List of symbols passing microstructure filters
        """
        logger.info("Applying microstructure filters (Layer 3)")
        
        # Filter liquidity data to only include symbols that passed Layer 2
        filtered_liquidity = liquidity_data[
            liquidity_data['symbol'].isin(symbols)
        ].copy()
        
        passed_symbols = []
        
        for _, row in filtered_liquidity.iterrows():
            symbol = row['symbol']
            
            # Filter 1: Bid-ask spread < 5 bp
            bid_ask_spread = row.get('avg_bid_ask_spread', 1.0)
            spread_check = bid_ask_spread <= config.MAX_BID_ASK_SPREAD
            
            # Filter 2: Quote update frequency approximation
            # Higher quote update frequency indicates better microstructure
            quote_freq = row.get('quote_update_freq', 0)
            # Convert to implied milliseconds between updates
            update_interval_ms = 1000 / max(quote_freq, 1e-6)
            freq_check = update_interval_ms <= config.MIN_QUOTE_UPDATE_RATE
            
            # Additional microstructure checks
            # Low price impact (Amihud illiquidity measure)
            amihud_illiq = row.get('amihud_illiquidity', float('inf'))
            impact_check = amihud_illiq < 1e-6  # Low impact threshold
            
            if spread_check and freq_check and impact_check:
                passed_symbols.append(symbol)
                logger.debug(f"{symbol} passed microstructure filters")
            else:
                logger.debug(f"{symbol} failed microstructure filters: "
                           f"Spread={spread_check}, Freq={freq_check}, "
                           f"Impact={impact_check}")
        
        logger.info(f"Microstructure filters: {len(passed_symbols)} symbols passed")
        return passed_symbols
    
    def create_filtered_universe(self, 
                               price_data: pd.DataFrame,
                               fundamental_data: pd.DataFrame,
                               liquidity_data: pd.DataFrame) -> Dict:
        """
        Apply all three layers of filtering to create the final universe
        
        Args:
            price_data: Historical price data
            fundamental_data: Fundamental metrics  
            liquidity_data: Liquidity metrics
            
        Returns:
            Dictionary with filtered universe and metadata
        """
        logger.info("Creating filtered universe (applying all three layers)")
        
        # Start with all symbols that have data
        initial_symbols = set(fundamental_data['symbol'].tolist())
        logger.info(f"Starting with {len(initial_symbols)} initial symbols")
        
        # Layer 1: Liquidity filters
        layer1_symbols = self.apply_liquidity_filters(
            price_data, fundamental_data, liquidity_data
        )
        
        # Layer 2: Fundamental homogeneity
        layer2_symbols = self.apply_fundamental_homogeneity_filters(
            layer1_symbols, fundamental_data
        )
        
        # Layer 3: Microstructure filters
        final_symbols = self.apply_microstructure_filters(
            layer2_symbols, liquidity_data
        )
        
        # Create universe metadata
        universe_metadata = {
            'initial_count': len(initial_symbols),
            'post_liquidity_count': len(layer1_symbols),
            'post_homogeneity_count': len(layer2_symbols), 
            'final_count': len(final_symbols),
            'final_symbols': final_symbols,
            'liquidity_retention': len(layer1_symbols) / len(initial_symbols),
            'homogeneity_retention': len(layer2_symbols) / len(layer1_symbols) if layer1_symbols else 0,
            'microstructure_retention': len(final_symbols) / len(layer2_symbols) if layer2_symbols else 0,
            'overall_retention': len(final_symbols) / len(initial_symbols)
        }
        
        # Group final symbols by industry for pair discovery
        final_fundamentals = fundamental_data[
            fundamental_data['symbol'].isin(final_symbols)
        ]
        
        industry_groups = {}
        for industry, group in final_fundamentals.groupby('industry'):
            industry_symbols = group['symbol'].tolist()
            if len(industry_symbols) >= 2:  # Only include industries with 2+ stocks
                industry_groups[industry] = industry_symbols
        
        universe_data = {
            'symbols': final_symbols,
            'metadata': universe_metadata,
            'industry_groups': industry_groups,
            'fundamentals': final_fundamentals,
            'creation_date': pd.Timestamp.now()
        }
        
        logger.info(f"Final universe created: {len(final_symbols)} symbols "
                   f"across {len(industry_groups)} industries")
        
        return universe_data


class ETFUniverseFilter:
    """Specialized filter for ETF universe"""
    
    def __init__(self):
        pass
        
    def filter_liquid_etfs(self, 
                          price_data: pd.DataFrame,
                          fundamental_data: pd.DataFrame) -> List[str]:
        """
        Filter ETFs based on liquidity and tracking quality
        
        Args:
            price_data: Historical price data
            fundamental_data: Fundamental metrics
            
        Returns:
            List of qualified ETF symbols
        """
        logger.info("Filtering liquid ETFs")
        
        # Start with predefined liquid ETFs
        candidate_etfs = data_config.LIQUID_ETFS
        
        qualified_etfs = []
        
        for etf in candidate_etfs:
            # Check if we have data for this ETF
            if etf not in price_data.columns.get_level_values(0):
                continue
                
            try:
                etf_data = price_data[etf]
                
                # Liquidity checks
                avg_volume_usd = etf_data['Volume_USD'].rolling(30).mean().iloc[-1]
                
                # ETF-specific criteria (more lenient than stocks)
                volume_check = avg_volume_usd >= 50_000_000  # $50M ADV for ETFs
                
                # Tracking quality check (low tracking error)
                returns = etf_data['Returns'].dropna()
                tracking_vol = returns.rolling(60).std().iloc[-1]
                tracking_check = tracking_vol < 0.02  # < 2% daily volatility
                
                if volume_check and tracking_check:
                    qualified_etfs.append(etf)
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate ETF {etf}: {e}")
        
        logger.info(f"Qualified ETFs: {len(qualified_etfs)} out of {len(candidate_etfs)}")
        return qualified_etfs


class UniverseMonitor:
    """Monitor universe composition and trigger rebalancing"""
    
    def __init__(self):
        self.last_universe = None
        self.last_update = None
        
    def check_universe_stability(self, 
                                current_universe: Dict,
                                threshold: float = 0.1) -> Dict:
        """
        Check if current universe has changed significantly
        
        Args:
            current_universe: Current universe data
            threshold: Maximum allowed change ratio
            
        Returns:
            Dictionary with stability metrics
        """
        if self.last_universe is None:
            self.last_universe = current_universe
            return {
                'is_stable': True,
                'change_ratio': 0.0,
                'added_symbols': [],
                'removed_symbols': [],
                'requires_rebalance': False
            }
        
        current_symbols = set(current_universe['symbols'])
        last_symbols = set(self.last_universe['symbols'])
        
        # Calculate changes
        added_symbols = list(current_symbols - last_symbols)
        removed_symbols = list(last_symbols - current_symbols)
        
        total_change = len(added_symbols) + len(removed_symbols)
        change_ratio = total_change / len(last_symbols) if last_symbols else 1.0
        
        is_stable = change_ratio <= threshold
        requires_rebalance = not is_stable
        
        stability_metrics = {
            'is_stable': is_stable,
            'change_ratio': change_ratio,
            'added_symbols': added_symbols,
            'removed_symbols': removed_symbols,
            'requires_rebalance': requires_rebalance,
            'current_count': len(current_symbols),
            'previous_count': len(last_symbols)
        }
        
        if requires_rebalance:
            logger.warning(f"Universe instability detected: {change_ratio:.2%} change")
        
        # Update stored universe
        self.last_universe = current_universe
        self.last_update = pd.Timestamp.now()
        
        return stability_metrics


# Global instances
universe_filter = UniverseFilter()
etf_filter = ETFUniverseFilter()
universe_monitor = UniverseMonitor() 