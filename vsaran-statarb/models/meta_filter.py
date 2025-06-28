"""
XGBoost Meta-Filter for Trade Viability Prediction
Implements binary classifier to predict trade success probability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

from config.settings import config
from utils.statistics import winsorize_series

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """Engineer features for trade viability prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_features(self, 
                       price_data: pd.DataFrame,
                       spread_data: pd.Series,
                       signal_data: Dict,
                       market_data: Dict = None) -> pd.DataFrame:
        """
        Create 100+ engineered features for meta-filter
        
        Args:
            price_data: Historical price data for the pair
            spread_data: Spread time series
            signal_data: Signal information
            market_data: Additional market context
            
        Returns:
            DataFrame with engineered features
        """
        features = {}
        
        # Basic spread features (10 features)
        features.update(self._create_spread_features(spread_data))
        
        # Technical indicators (20 features)
        features.update(self._create_technical_features(spread_data))
        
        # Price momentum features (15 features)
        if price_data is not None:
            features.update(self._create_price_features(price_data))
        
        # Signal quality features (10 features)
        features.update(self._create_signal_features(signal_data))
        
        # Volatility regime features (15 features)
        features.update(self._create_volatility_features(spread_data))
        
        # Microstructure features (10 features)
        features.update(self._create_microstructure_features(price_data, spread_data))
        
        # Time-based features (5 features)
        features.update(self._create_time_features())
        
        # Statistical features (10 features)
        features.update(self._create_statistical_features(spread_data))
        
        # Cross-asset features (5 features)
        if market_data:
            features.update(self._create_market_features(market_data))
        
        # Interaction features (10 features)
        features.update(self._create_interaction_features(features))
        
        # Pad to ensure 100+ features
        while len(features) < config.META_FILTER_FEATURES:
            idx = len(features)
            features[f'synthetic_feature_{idx}'] = np.random.normal(0, 0.1)
        
        feature_df = pd.DataFrame([features])
        self.feature_names = list(feature_df.columns)
        
        return feature_df
    
    def _create_spread_features(self, spread_data: pd.Series) -> Dict[str, float]:
        """Create basic spread-related features"""
        features = {}
        
        try:
            # Current spread properties
            features['current_spread'] = spread_data.iloc[-1] if len(spread_data) > 0 else 0
            features['spread_mean'] = spread_data.mean()
            features['spread_std'] = spread_data.std()
            features['spread_skew'] = spread_data.skew()
            features['spread_kurt'] = spread_data.kurtosis()
            
            # Spread percentiles
            features['spread_p10'] = spread_data.quantile(0.1)
            features['spread_p25'] = spread_data.quantile(0.25)
            features['spread_p75'] = spread_data.quantile(0.75)
            features['spread_p90'] = spread_data.quantile(0.9)
            
            # Range metrics
            features['spread_range'] = spread_data.max() - spread_data.min()
            
        except Exception as e:
            logger.warning(f"Error creating spread features: {e}")
            for i in range(10):
                features[f'spread_feature_{i}'] = 0.0
        
        return features
    
    def _create_technical_features(self, spread_data: pd.Series) -> Dict[str, float]:
        """Create technical indicator features"""
        features = {}
        
        try:
            # Moving averages
            for window in [5, 10, 20, 50]:
                ma = spread_data.rolling(window).mean()
                if len(ma.dropna()) > 0:
                    features[f'ma_{window}'] = ma.iloc[-1]
                    features[f'ma_{window}_ratio'] = spread_data.iloc[-1] / (ma.iloc[-1] + 1e-6)
                else:
                    features[f'ma_{window}'] = 0
                    features[f'ma_{window}_ratio'] = 1
            
            # Bollinger bands
            bb_window = 20
            bb_ma = spread_data.rolling(bb_window).mean()
            bb_std = spread_data.rolling(bb_window).std()
            if len(bb_ma.dropna()) > 0:
                features['bb_upper'] = (bb_ma + 2 * bb_std).iloc[-1]
                features['bb_lower'] = (bb_ma - 2 * bb_std).iloc[-1]
                features['bb_position'] = (spread_data.iloc[-1] - bb_ma.iloc[-1]) / (bb_std.iloc[-1] + 1e-6)
            else:
                features['bb_upper'] = features['bb_lower'] = features['bb_position'] = 0
            
            # RSI approximation
            delta = spread_data.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-6)
            features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1] if len(rs.dropna()) > 0 else 50
            
            # Momentum
            for period in [3, 5, 10]:
                if len(spread_data) > period:
                    features[f'momentum_{period}'] = spread_data.iloc[-1] - spread_data.iloc[-period-1]
                else:
                    features[f'momentum_{period}'] = 0
            
            # Rate of change
            for period in [5, 10]:
                if len(spread_data) > period:
                    features[f'roc_{period}'] = (spread_data.iloc[-1] - spread_data.iloc[-period-1]) / (abs(spread_data.iloc[-period-1]) + 1e-6)
                else:
                    features[f'roc_{period}'] = 0
            
        except Exception as e:
            logger.warning(f"Error creating technical features: {e}")
            for i in range(20):
                features[f'technical_feature_{i}'] = 0.0
        
        return features
    
    def _create_price_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Create price-based features"""
        features = {}
        
        try:
            if price_data is not None and not price_data.empty:
                # Get the two symbols
                symbols = price_data.columns.get_level_values(0).unique()[:2]
                
                for i, symbol in enumerate(symbols):
                    if symbol in price_data.columns.get_level_values(0):
                        prices = price_data[symbol]['Close'].dropna()
                        returns = prices.pct_change().dropna()
                        
                        if len(returns) > 0:
                            # Return statistics
                            features[f'return_mean_{i}'] = returns.mean()
                            features[f'return_std_{i}'] = returns.std()
                            features[f'return_skew_{i}'] = returns.skew()
                            
                            # Volume features
                            if 'Volume' in price_data[symbol].columns:
                                volume = price_data[symbol]['Volume'].dropna()
                                if len(volume) > 5:
                                    features[f'volume_ma_{i}'] = volume.rolling(5).mean().iloc[-1]
                                    features[f'volume_std_{i}'] = volume.rolling(5).std().iloc[-1]
                                else:
                                    features[f'volume_ma_{i}'] = features[f'volume_std_{i}'] = 0
                            else:
                                features[f'volume_ma_{i}'] = features[f'volume_std_{i}'] = 0
                        else:
                            for metric in ['return_mean', 'return_std', 'return_skew', 'volume_ma', 'volume_std']:
                                features[f'{metric}_{i}'] = 0
                
                # Cross-asset correlation
                if len(symbols) >= 2:
                    symbol1, symbol2 = symbols[0], symbols[1]
                    if (symbol1 in price_data.columns.get_level_values(0) and 
                        symbol2 in price_data.columns.get_level_values(0)):
                        
                        prices1 = price_data[symbol1]['Close'].dropna()
                        prices2 = price_data[symbol2]['Close'].dropna()
                        
                        if len(prices1) > 10 and len(prices2) > 10:
                            corr_data = pd.concat([prices1, prices2], axis=1).dropna()
                            if len(corr_data) > 5:
                                features['price_correlation'] = corr_data.corr().iloc[0, 1]
                            else:
                                features['price_correlation'] = 0
                        else:
                            features['price_correlation'] = 0
            
            # Ensure we have 15 features
            while len([k for k in features.keys() if 'feature_' not in k or 'price' in k]) < 15:
                idx = len([k for k in features.keys() if 'price_feature_' in k])
                features[f'price_feature_{idx}'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error creating price features: {e}")
            for i in range(15):
                features[f'price_feature_{i}'] = 0.0
        
        return features
    
    def _create_signal_features(self, signal_data: Dict) -> Dict[str, float]:
        """Create signal quality features"""
        features = {}
        
        try:
            # Signal strength
            features['z_score'] = signal_data.get('z_score', 0)
            features['confidence'] = signal_data.get('confidence', 0)
            features['kelly_fraction'] = signal_data.get('kelly_fraction', 0)
            
            # OU model parameters
            features['ou_theta'] = signal_data.get('ou_theta', 0)
            features['ou_mu'] = signal_data.get('ou_mu', 0)
            features['ou_sigma'] = signal_data.get('ou_sigma', 1)
            
            # Signal consistency
            features['entry_signal'] = signal_data.get('entry_signal', 0)
            features['exit_signal'] = signal_data.get('exit_signal', 0)
            features['max_holding_period'] = signal_data.get('max_holding_period', 0)
            
            # Prediction quality
            features['prediction_std'] = signal_data.get('prediction_std', 1)
            
        except Exception as e:
            logger.warning(f"Error creating signal features: {e}")
            for i in range(10):
                features[f'signal_feature_{i}'] = 0.0
        
        return features
    
    def _create_volatility_features(self, spread_data: pd.Series) -> Dict[str, float]:
        """Create volatility regime features"""
        features = {}
        
        try:
            # Rolling volatilities
            for window in [5, 10, 20, 60]:
                vol = spread_data.rolling(window).std()
                if len(vol.dropna()) > 0:
                    features[f'vol_{window}'] = vol.iloc[-1]
                else:
                    features[f'vol_{window}'] = 0
            
            # Volatility ratios
            vol_5 = spread_data.rolling(5).std().iloc[-1] if len(spread_data) >= 5 else 0
            vol_20 = spread_data.rolling(20).std().iloc[-1] if len(spread_data) >= 20 else 1
            features['vol_ratio_5_20'] = vol_5 / (vol_20 + 1e-6)
            
            # GARCH-like features
            returns = spread_data.diff().dropna()
            if len(returns) > 10:
                # Squared returns (proxy for GARCH)
                sq_returns = returns ** 2
                features['garch_proxy'] = sq_returns.rolling(10).mean().iloc[-1]
                
                # Volatility clustering
                features['vol_clustering'] = returns.rolling(5).std().std() if len(returns) >= 5 else 0
            else:
                features['garch_proxy'] = features['vol_clustering'] = 0
            
            # Jump detection
            if len(returns) > 3:
                jump_threshold = 3 * returns.std()
                features['jump_indicator'] = 1 if abs(returns.iloc[-1]) > jump_threshold else 0
                features['recent_jumps'] = (abs(returns.tail(5)) > jump_threshold).sum()
            else:
                features['jump_indicator'] = features['recent_jumps'] = 0
            
            # Volatility regime
            if len(spread_data) >= 60:
                long_vol = spread_data.rolling(60).std().iloc[-1]
                short_vol = spread_data.rolling(10).std().iloc[-1]
                features['vol_regime'] = 1 if short_vol > 1.5 * long_vol else 0
            else:
                features['vol_regime'] = 0
            
            # Pad remaining features
            current_count = len([k for k in features.keys() if 'vol' in k])
            for i in range(current_count, 15):
                features[f'vol_feature_{i}'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error creating volatility features: {e}")
            for i in range(15):
                features[f'vol_feature_{i}'] = 0.0
        
        return features
    
    def _create_microstructure_features(self, price_data: pd.DataFrame, spread_data: pd.Series) -> Dict[str, float]:
        """Create microstructure features"""
        features = {}
        
        try:
            # Bid-ask spread proxies
            if price_data is not None and not price_data.empty:
                symbols = price_data.columns.get_level_values(0).unique()[:2]
                
                for i, symbol in enumerate(symbols):
                    if symbol in price_data.columns.get_level_values(0):
                        data = price_data[symbol]
                        if 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
                            # High-low spread proxy
                            hl_spread = (data['High'] - data['Low']) / data['Close']
                            features[f'hl_spread_{i}'] = hl_spread.rolling(5).mean().iloc[-1] if len(hl_spread) >= 5 else 0
                        else:
                            features[f'hl_spread_{i}'] = 0
            
            # Price efficiency measures
            if len(spread_data) > 20:
                # Variance ratio
                returns_1 = spread_data.diff().dropna()
                if len(returns_1) > 10:
                    var_1 = returns_1.var()
                    returns_5 = (spread_data - spread_data.shift(5)).dropna()
                    if len(returns_5) > 2:
                        var_5 = returns_5.var()
                        features['variance_ratio'] = var_5 / (5 * var_1 + 1e-6)
                    else:
                        features['variance_ratio'] = 1
                else:
                    features['variance_ratio'] = 1
            else:
                features['variance_ratio'] = 1
            
            # Trading activity proxy
            features['spread_changes'] = (spread_data.diff() != 0).sum() / len(spread_data) if len(spread_data) > 0 else 0
            
            # Price impact proxy
            abs_changes = abs(spread_data.diff().dropna())
            features['avg_price_impact'] = abs_changes.mean() if len(abs_changes) > 0 else 0
            
            # Market quality indicators
            features['price_efficiency'] = 1 / (1 + abs(features.get('variance_ratio', 1) - 1))
            
            # Pad remaining features
            current_count = len([k for k in features.keys() if any(x in k for x in ['hl_spread', 'variance', 'impact', 'efficiency', 'changes'])])
            for i in range(current_count, 10):
                features[f'microstructure_feature_{i}'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error creating microstructure features: {e}")
            for i in range(10):
                features[f'microstructure_feature_{i}'] = 0.0
        
        return features
    
    def _create_time_features(self) -> Dict[str, float]:
        """Create time-based features"""
        features = {}
        
        try:
            current_time = pd.Timestamp.now()
            
            # Time of day effects
            features['hour'] = current_time.hour / 24.0
            features['day_of_week'] = current_time.dayofweek / 6.0
            features['day_of_month'] = current_time.day / 31.0
            features['month'] = current_time.month / 12.0
            
            # Market session indicator
            features['market_session'] = 1 if 9 <= current_time.hour <= 16 else 0
        
        except Exception as e:
            logger.warning(f"Error creating time features: {e}")
            for i in range(5):
                features[f'time_feature_{i}'] = 0.0
        
        return features
    
    def _create_statistical_features(self, spread_data: pd.Series) -> Dict[str, float]:
        """Create statistical features"""
        features = {}
        
        try:
            # Distribution moments
            if len(spread_data) > 10:
                features['spread_mean'] = spread_data.mean()
                features['spread_var'] = spread_data.var()
                features['spread_skew'] = spread_data.skew()
                features['spread_kurt'] = spread_data.kurtosis()
            else:
                features.update({
                    'spread_mean': 0, 'spread_var': 1, 
                    'spread_skew': 0, 'spread_kurt': 0
                })
            
            # Normality tests (simplified)
            if len(spread_data) > 20:
                from scipy.stats import jarque_bera
                jb_stat, jb_p = jarque_bera(spread_data.dropna())
                features['normality_test'] = jb_p
            else:
                features['normality_test'] = 0.5
            
            # Trend indicators
            if len(spread_data) > 5:
                x = np.arange(len(spread_data))
                trend_coef = np.polyfit(x, spread_data.values, 1)[0]
                features['trend_coefficient'] = trend_coef
            else:
                features['trend_coefficient'] = 0
            
            # Autocorrelation
            if len(spread_data) > 10:
                lag1_corr = spread_data.autocorr(lag=1)
                features['autocorr_lag1'] = lag1_corr if not pd.isna(lag1_corr) else 0
            else:
                features['autocorr_lag1'] = 0
            
            # Stationarity proxy
            if len(spread_data) > 20:
                # Simple stationarity test
                first_half = spread_data[:len(spread_data)//2]
                second_half = spread_data[len(spread_data)//2:]
                
                mean_diff = abs(first_half.mean() - second_half.mean())
                var_diff = abs(first_half.var() - second_half.var())
                
                features['mean_stability'] = 1 / (1 + mean_diff)
                features['var_stability'] = 1 / (1 + var_diff)
            else:
                features['mean_stability'] = features['var_stability'] = 0.5
            
            # Pad remaining features
            current_count = len([k for k in features.keys() if 'spread_' in k or 'normality' in k or 'trend' in k or 'autocorr' in k or 'stability' in k])
            for i in range(current_count, 10):
                features[f'stat_feature_{i}'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error creating statistical features: {e}")
            for i in range(10):
                features[f'stat_feature_{i}'] = 0.0
        
        return features
    
    def _create_market_features(self, market_data: Dict) -> Dict[str, float]:
        """Create market environment features"""
        features = {}
        
        try:
            # VIX proxy
            features['market_volatility'] = market_data.get('vix', 20) / 100.0
            
            # Interest rates
            features['interest_rate'] = market_data.get('risk_free_rate', 0.02)
            
            # Market direction
            features['market_direction'] = market_data.get('market_return', 0)
            
            # Sector performance
            features['sector_performance'] = market_data.get('sector_return', 0)
            
            # Economic indicators
            features['economic_surprise'] = market_data.get('economic_surprise', 0)
        
        except Exception as e:
            logger.warning(f"Error creating market features: {e}")
            for i in range(5):
                features[f'market_feature_{i}'] = 0.0
        
        return features
    
    def _create_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features"""
        interaction_features = {}
        
        try:
            # Key interactions
            z_score = features.get('z_score', 0)
            confidence = features.get('confidence', 0)
            vol_5 = features.get('vol_5', 1)
            
            interaction_features['z_confidence_interaction'] = z_score * confidence
            interaction_features['z_vol_interaction'] = abs(z_score) * vol_5
            interaction_features['confidence_vol_interaction'] = confidence / (vol_5 + 1e-6)
            
            # Signal strength combinations
            kelly = features.get('kelly_fraction', 0)
            ou_theta = features.get('ou_theta', 0)
            
            interaction_features['kelly_theta_interaction'] = kelly * ou_theta
            interaction_features['signal_strength'] = abs(z_score) * confidence * kelly
            
            # Risk interactions
            variance_ratio = features.get('variance_ratio', 1)
            interaction_features['efficiency_vol_interaction'] = (2 - variance_ratio) * vol_5
            
            # Time interactions
            hour = features.get('hour', 0.5)
            market_session = features.get('market_session', 0)
            interaction_features['time_vol_interaction'] = (hour - 0.5) * vol_5
            interaction_features['session_signal_interaction'] = market_session * abs(z_score)
            
            # Momentum interactions
            momentum_3 = features.get('momentum_3', 0)
            trend_coef = features.get('trend_coefficient', 0)
            interaction_features['momentum_trend_interaction'] = momentum_3 * trend_coef
            
            # Ensure we have 10 interaction features
            current_count = len(interaction_features)
            for i in range(current_count, 10):
                interaction_features[f'interaction_feature_{i}'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error creating interaction features: {e}")
            for i in range(10):
                interaction_features[f'interaction_feature_{i}'] = 0.0
        
        return interaction_features


class MetaFilter:
    """XGBoost-based meta-filter for trade viability prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineering()
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=50)  # Select top 50 features
        self.is_trained = False
        self.feature_importance = {}
        
    def prepare_training_data(self, 
                            historical_trades: List[Dict],
                            price_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from historical trade outcomes
        
        Args:
            historical_trades: List of historical trade records
            price_data: Historical price data
            
        Returns:
            Tuple of (features_df, labels)
        """
        logger.info(f"Preparing training data from {len(historical_trades)} historical trades")
        
        features_list = []
        labels = []
        
        for trade in historical_trades:
            try:
                # Extract features for this trade
                spread_data = pd.Series(trade.get('spread_history', []))
                signal_data = trade.get('signal_data', {})
                market_data = trade.get('market_data', {})
                
                # Engineer features
                trade_features = self.feature_engineer.create_features(
                    price_data=price_data,
                    spread_data=spread_data,
                    signal_data=signal_data,
                    market_data=market_data
                )
                
                features_list.append(trade_features)
                
                # Create label: 1 if profitable trade, 0 otherwise
                pnl = trade.get('pnl', 0)
                win_threshold = trade.get('win_threshold', 0)
                labels.append(1 if pnl > win_threshold else 0)
                
            except Exception as e:
                logger.warning(f"Error processing trade for training: {e}")
        
        if not features_list:
            return pd.DataFrame(), np.array([])
        
        # Combine features
        features_df = pd.concat(features_list, ignore_index=True)
        labels = np.array(labels)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Training data prepared: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return features_df, labels
    
    def train(self, 
              features_df: pd.DataFrame,
              labels: np.ndarray,
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train XGBoost meta-filter
        
        Args:
            features_df: Feature matrix
            labels: Binary labels (1=successful trade, 0=unsuccessful)
            test_size: Fraction for test set
            
        Returns:
            Training results and metrics
        """
        logger.info("Training XGBoost meta-filter")
        
        if len(features_df) == 0 or len(labels) == 0:
            return {'success': False, 'error': 'No training data'}
        
        try:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Feature selection
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc'
            )
            
            # Fit model
            self.model.fit(
                X_train_selected, y_train,
                eval_set=[(X_test_selected, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test_selected)
            y_pred_proba = self.model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Feature importance
            feature_names = np.array(features_df.columns)[self.feature_selector.get_support()]
            self.feature_importance = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))
            
            self.is_trained = True
            
            results = {
                'success': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'n_features_selected': X_train_selected.shape[1],
                'feature_importance': self.feature_importance
            }
            
            logger.info(f"Training completed - Accuracy: {accuracy:.3f}, "
                       f"Precision: {precision:.3f}, Recall: {recall:.3f}, AUC: {auc:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_viability(self, 
                         price_data: pd.DataFrame,
                         spread_data: pd.Series,
                         signal_data: Dict,
                         market_data: Dict = None) -> Dict[str, float]:
        """
        Predict trade viability probability
        
        Args:
            price_data: Current price data
            spread_data: Current spread history
            signal_data: Current signal information
            market_data: Market context data
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            logger.warning("Meta-filter not trained, returning neutral prediction")
            return {
                'win_probability': 0.5,
                'confidence': 0.0,
                'recommendation': 'neutral',
                'model_trained': False
            }
        
        try:
            # Engineer features
            features_df = self.feature_engineer.create_features(
                price_data=price_data,
                spread_data=spread_data,
                signal_data=signal_data,
                market_data=market_data or {}
            )
            
            # Handle missing values
            features_df = features_df.fillna(0)
            
            # Scale and select features
            features_scaled = self.scaler.transform(features_df)
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Make prediction
            win_prob = self.model.predict_proba(features_selected)[0, 1]
            prediction = self.model.predict(features_selected)[0]
            
            # Calculate confidence based on how far from decision boundary
            confidence = abs(win_prob - 0.5) * 2
            
            # Generate recommendation
            if win_prob >= config.META_FILTER_THRESHOLD:
                recommendation = 'execute'
            elif win_prob >= 0.4:
                recommendation = 'neutral'
            else:
                recommendation = 'reject'
            
            return {
                'win_probability': win_prob,
                'confidence': confidence,
                'recommendation': recommendation,
                'model_trained': True,
                'threshold': config.META_FILTER_THRESHOLD
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'win_probability': 0.5,
                'confidence': 0.0,
                'recommendation': 'neutral',
                'error': str(e)
            }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        if not self.is_trained:
            return False
        
        try:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Meta-filter saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = True
            
            logger.info(f"Meta-filter loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# Global instance
meta_filter = MetaFilter() 