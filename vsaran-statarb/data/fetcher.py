"""
Data fetching and preprocessing for the DOAC system
Handles price data, volume, and fundamental metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

from config.settings import config, data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Main data fetching class for price and fundamental data"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        
    def fetch_price_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        max_workers: int = 10) -> pd.DataFrame:
        """
        Fetch historical price data for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Number of concurrent threads
            
        Returns:
            DataFrame with multi-level columns (symbol, price_type)
        """
        logger.info(f"Fetching price data for {len(symbols)} symbols")
        
        def fetch_single_symbol(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                # Calculate additional metrics
                data['Returns'] = data['Close'].pct_change()
                data['Volume_USD'] = data['Close'] * data['Volume']
                data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
                
                return symbol, data
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        price_data = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(fetch_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if not data.empty:
                    price_data[symbol] = data
        
        if not price_data:
            return pd.DataFrame()
        
        # Combine into multi-level DataFrame
        combined_data = pd.concat(price_data, axis=1)
        combined_data.columns.names = ['Symbol', 'Metric']
        
        return combined_data
    
    def fetch_fundamental_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for universe filtering
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            DataFrame with fundamental metrics
        """
        logger.info(f"Fetching fundamental data for {len(symbols)} symbols")
        
        def fetch_single_fundamental(symbol: str) -> Dict:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract key metrics for filtering
                fundamentals = {
                    'symbol': symbol,
                    'market_cap': info.get('marketCap', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'beta': info.get('beta', 1.0),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'roe': info.get('returnOnEquity', np.nan),
                    'roa': info.get('returnOnAssets', np.nan),
                    'debt_to_equity': info.get('debtToEquity', np.nan),
                    'current_ratio': info.get('currentRatio', np.nan),
                    'gross_margin': info.get('grossMargins', np.nan),
                    'operating_margin': info.get('operatingMargins', np.nan),
                    'net_margin': info.get('profitMargins', np.nan),
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'earnings_growth': info.get('earningsGrowth', np.nan),
                    'float_shares': info.get('floatShares', 0),
                    'shares_outstanding': info.get('sharesOutstanding', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'avg_volume_10days': info.get('averageVolume10days', 0)
                }
                
                # Calculate ROIC approximation
                roic = fundamentals.get('roa', 0) * (1 + fundamentals.get('debt_to_equity', 0) / 100)
                fundamentals['roic'] = roic
                
                return fundamentals
                
            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
                return {'symbol': symbol}
        
        fundamental_data = []
        with ThreadPoolExecutor(max_workers=5) as executor:  # Be nice to Yahoo Finance
            futures = [
                executor.submit(fetch_single_fundamental, symbol) 
                for symbol in symbols
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    fundamental_data.append(result)
                # Small delay to avoid rate limiting
                time.sleep(0.1)
        
        return pd.DataFrame(fundamental_data)
    
    def calculate_liquidity_metrics(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity metrics for universe filtering
        
        Args:
            price_data: Multi-level price DataFrame
            
        Returns:
            DataFrame with liquidity metrics
        """
        logger.info("Calculating liquidity metrics")
        
        liquidity_metrics = []
        
        for symbol in price_data.columns.get_level_values(0).unique():
            try:
                symbol_data = price_data[symbol]
                
                # Average Daily Volume (30-day)
                adv_30d = symbol_data['Volume_USD'].rolling(30).mean().iloc[-1]
                
                # Bid-ask spread approximation using high-low
                bid_ask_approx = (symbol_data['High'] - symbol_data['Low']) / symbol_data['Close']
                avg_bid_ask = bid_ask_approx.rolling(20).mean().iloc[-1]
                
                # Price impact approximation (Amihud illiquidity)
                returns = symbol_data['Returns'].abs()
                volume_usd = symbol_data['Volume_USD']
                amihud_illiq = (returns / volume_usd).rolling(20).mean().iloc[-1]
                
                # Quote update frequency approximation (inverse of volatility)
                volatility = symbol_data['Volatility'].iloc[-1]
                quote_update_freq = 1 / (volatility + 1e-6)  # Higher vol = more updates
                
                metrics = {
                    'symbol': symbol,
                    'adv_30d': adv_30d,
                    'avg_bid_ask_spread': avg_bid_ask,
                    'amihud_illiquidity': amihud_illiq,
                    'quote_update_freq': quote_update_freq,
                    'avg_price': symbol_data['Close'].iloc[-1]
                }
                
                liquidity_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to calculate liquidity for {symbol}: {e}")
        
        return pd.DataFrame(liquidity_metrics)
    
    def get_sector_industry_mapping(self) -> Dict[str, List[str]]:
        """
        Get GICS sector to industry mapping for homogeneity filtering
        
        Returns:
            Dictionary mapping sectors to list of industries
        """
        # Simplified GICS mapping - in production, use official GICS data
        sector_industry_map = {
            'Technology': [
                'Software—Application', 'Software—Infrastructure', 'Semiconductors',
                'Computer Hardware', 'Electronic Components', 'IT Services'
            ],
            'Healthcare': [
                'Biotechnology', 'Drug Manufacturers—General', 'Drug Manufacturers—Specialty',
                'Medical Devices', 'Healthcare Plans', 'Medical Instruments & Supplies'
            ],
            'Financial Services': [
                'Banks—Regional', 'Banks—Diversified', 'Asset Management',
                'Insurance—Life', 'Insurance—Property & Casualty', 'Capital Markets'
            ],
            'Consumer Discretionary': [
                'Auto Manufacturers', 'Restaurants', 'Specialty Retail',
                'Apparel Retail', 'Leisure', 'Resorts & Casinos'
            ],
            'Consumer Staples': [
                'Beverages—Non-Alcoholic', 'Beverages—Alcoholic', 'Food Distribution',
                'Grocery Stores', 'Household & Personal Products', 'Tobacco'
            ],
            'Industrials': [
                'Aerospace & Defense', 'Airlines', 'Industrial Distribution',
                'Trucking', 'Railroads', 'Engineering & Construction'
            ],
            'Materials': [
                'Steel', 'Chemicals', 'Gold', 'Copper', 'Aluminum',
                'Agricultural Inputs', 'Specialty Chemicals'
            ],
            'Energy': [
                'Oil & Gas E&P', 'Oil & Gas Integrated', 'Oil & Gas Refining & Marketing',
                'Oil & Gas Equipment & Services', 'Oil & Gas Midstream'
            ],
            'Utilities': [
                'Utilities—Regulated Electric', 'Utilities—Renewable',
                'Utilities—Regulated Gas', 'Utilities—Regulated Water'
            ],
            'Real Estate': [
                'REIT—Residential', 'REIT—Retail', 'REIT—Office',
                'REIT—Industrial', 'REIT—Healthcare Facilities'
            ],
            'Communication Services': [
                'Telecom Services', 'Internet Content & Information',
                'Entertainment', 'Broadcasting', 'Advertising Agencies'
            ]
        }
        
        return sector_industry_map
    
    @lru_cache(maxsize=100)
    def get_cached_data(self, 
                       symbols_hash: str, 
                       start_date: str, 
                       end_date: str) -> pd.DataFrame:
        """
        Cached data fetching to avoid repeated API calls
        
        Args:
            symbols_hash: Hash of symbols list
            start_date: Start date
            end_date: End date
            
        Returns:
            Cached DataFrame or fetches new data
        """
        cache_key = f"{symbols_hash}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # This would be implemented with actual caching logic
        return pd.DataFrame()


class DataValidator:
    """Validates data quality and completeness"""
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame, 
                          min_observations: int = 252) -> Dict[str, bool]:
        """
        Validate price data quality
        
        Args:
            data: Price data DataFrame
            min_observations: Minimum required observations
            
        Returns:
            Dictionary with validation results per symbol
        """
        validation_results = {}
        
        for symbol in data.columns.get_level_values(0).unique():
            try:
                symbol_data = data[symbol]
                
                # Check data completeness
                total_obs = len(symbol_data.dropna())
                has_sufficient_data = total_obs >= min_observations
                
                # Check for outliers in returns
                returns = symbol_data['Returns'].dropna()
                has_extreme_returns = (abs(returns) > 0.2).any()  # 20% single-day moves
                
                # Check for zero prices or volumes
                has_zero_prices = (symbol_data['Close'] <= 0).any()
                has_zero_volumes = (symbol_data['Volume'] <= 0).any()
                
                # Overall validation
                is_valid = (has_sufficient_data and 
                           not has_extreme_returns and 
                           not has_zero_prices and 
                           not has_zero_volumes)
                
                validation_results[symbol] = {
                    'is_valid': is_valid,
                    'total_observations': total_obs,
                    'has_sufficient_data': has_sufficient_data,
                    'has_extreme_returns': has_extreme_returns,
                    'has_zero_prices': has_zero_prices,
                    'has_zero_volumes': has_zero_volumes
                }
                
            except Exception as e:
                logger.warning(f"Validation failed for {symbol}: {e}")
                validation_results[symbol] = {'is_valid': False}
        
        return validation_results
    
    @staticmethod
    def validate_fundamental_data(data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate fundamental data completeness
        
        Args:
            data: Fundamental data DataFrame
            
        Returns:
            Dictionary with validation results per symbol
        """
        validation_results = {}
        
        required_fields = ['market_cap', 'sector', 'industry', 'gross_margin', 'roic']
        
        for _, row in data.iterrows():
            symbol = row['symbol']
            
            # Check for required fields
            has_required_data = all(
                pd.notna(row.get(field)) and row.get(field) != 0 
                for field in required_fields
                if field in ['market_cap']
            )
            
            # Check for reasonable values
            market_cap = row.get('market_cap', 0)
            has_reasonable_values = market_cap > 0
            
            is_valid = has_required_data and has_reasonable_values
            
            validation_results[symbol] = {
                'is_valid': is_valid,
                'has_required_data': has_required_data,
                'has_reasonable_values': has_reasonable_values
            }
        
        return validation_results


# Global instances
data_fetcher = DataFetcher()
data_validator = DataValidator() 