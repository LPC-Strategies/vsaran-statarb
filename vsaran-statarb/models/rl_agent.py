"""
Reinforcement Learning Agent for Signal Enhancement
Implements Actor-Critic PPO agent for trading signal optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import warnings

from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class PairTradingEnv(gym.Env):
    """
    Custom environment for pair trading with RL
    """
    
    def __init__(self, 
                 price_data: pd.DataFrame = None,
                 pair_data: Dict = None,
                 initial_capital: float = 100_000):
        super(PairTradingEnv, self).__init__()
        
        # Environment parameters
        self.price_data = price_data
        self.pair_data = pair_data or {}
        self.initial_capital = initial_capital
        self.lookback = config.RL_LOOKBACK
        
        # State space: 15-feature vector as per blueprint
        # z-score, Δz, realized vol, order-book imbalance (L1 & L2), 
        # bid-ask trend, macro dummy, position, cash
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.RL_STATE_DIM,), dtype=np.float32
        )
        
        # Action space: {-1: sell spread, 0: hold, 1: buy spread}
        self.action_space = spaces.Discrete(3)
        
        # Trading state
        self.current_step = 0
        self.position = 0.0  # Current position (-1 to 1)
        self.cash = initial_capital
        self.total_value = initial_capital
        self.trade_history = []
        
        # Simulation parameters
        self.transaction_cost = config.DEFAULT_SLIPPAGE + config.TAKER_FEE
        self.max_position = 1.0
        
        # State tracking
        self.state_history = []
        self.returns_history = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback
        self.position = 0.0
        self.cash = self.initial_capital
        self.total_value = self.initial_capital
        self.trade_history = []
        self.state_history = []
        self.returns_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one time step"""
        # Decode action: 0=sell, 1=hold, 2=buy
        action_map = {0: -1, 1: 0, 2: 1}
        target_position = action_map[action] * self.max_position
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Execute trade if position changes
        reward = 0.0
        if abs(position_change) > 1e-6:
            trade_cost = abs(position_change) * self.transaction_cost
            reward -= trade_cost  # Transaction cost penalty
            
            # Update position
            self.position = target_position
        
        # Calculate P&L from spread movement
        if hasattr(self, '_current_spread') and hasattr(self, '_previous_spread'):
            spread_return = (self._current_spread - self._previous_spread) / abs(self._previous_spread + 1e-6)
            position_pnl = self.position * spread_return * self.initial_capital
            reward += position_pnl / self.initial_capital  # Normalize by capital
        
        # Inventory penalty (as per blueprint: λ = 0.01)
        inventory_penalty = config.RL_REWARD_LAMBDA * abs(self.position)
        reward -= inventory_penalty
        
        # Risk-adjusted reward
        if len(self.returns_history) > 10:
            volatility = np.std(self.returns_history[-10:])
            if volatility > 0:
                reward = reward / (volatility + 1e-6)
        
        # Update state
        self.current_step += 1
        self.returns_history.append(reward)
        
        # Check if episode is done
        done = (self.current_step >= len(self.pair_data.get('spread_history', [])) - 1 or
                abs(self.position) > 2.0 or  # Risk limit
                len(self.returns_history) > 1000)  # Episode length limit
        
        # Truncation (optional)
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct 15-feature state vector:
        1. z-score
        2. Δz (change in z-score)
        3. Realized volatility
        4. Order book imbalance L1 (approximated)
        5. Order book imbalance L2 (approximated)
        6. Bid-ask trend (approximated)
        7. Macro dummy (CPI day, etc.)
        8-10. Position and cash metrics
        11-15. Technical indicators
        """
        features = np.zeros(config.RL_STATE_DIM, dtype=np.float32)
        
        try:
            # Get current spread data
            spread_history = self.pair_data.get('spread_history', [])
            if self.current_step < len(spread_history):
                current_spread = spread_history[self.current_step]
                self._current_spread = current_spread
                
                if self.current_step > 0:
                    self._previous_spread = spread_history[self.current_step - 1]
                
                # Calculate z-score (feature 1)
                if self.current_step >= self.lookback:
                    recent_spreads = spread_history[self.current_step - self.lookback:self.current_step]
                    if len(recent_spreads) > 1:
                        mean_spread = np.mean(recent_spreads)
                        std_spread = np.std(recent_spreads)
                        features[0] = (current_spread - mean_spread) / (std_spread + 1e-6)
                
                # Δz (feature 2)
                if len(self.state_history) > 0:
                    features[1] = features[0] - self.state_history[-1][0]
                
                # Realized volatility (feature 3)
                if self.current_step >= self.lookback:
                    recent_returns = np.diff(spread_history[self.current_step - self.lookback:self.current_step + 1])
                    if len(recent_returns) > 1:
                        features[2] = np.std(recent_returns) * np.sqrt(252)  # Annualized
                
                # Order book imbalance proxies (features 4-5)
                # Approximated using price volatility and volume patterns
                if hasattr(self, 'price_data') and self.price_data is not None:
                    # This would use real order book data in production
                    features[3] = np.random.normal(0, 0.1)  # L1 imbalance proxy
                    features[4] = np.random.normal(0, 0.05)  # L2 imbalance proxy
                
                # Bid-ask trend (feature 6)
                # Approximated using high-low spread
                features[5] = np.random.normal(0, 0.02)  # Bid-ask trend proxy
                
                # Macro dummy (feature 7)
                # Would check economic calendar in production
                features[6] = 0.0  # Placeholder for macro events
                
                # Position and cash metrics (features 8-10)
                features[7] = self.position
                features[8] = self.cash / self.initial_capital
                features[9] = len(self.trade_history) / 100.0  # Normalized trade count
                
                # Technical indicators (features 10-14)
                if self.current_step >= 20:
                    recent_spreads = spread_history[max(0, self.current_step - 20):self.current_step]
                    if len(recent_spreads) > 10:
                        # Moving average deviation
                        ma = np.mean(recent_spreads[-10:])
                        features[10] = (current_spread - ma) / (np.std(recent_spreads) + 1e-6)
                        
                        # Momentum
                        if len(recent_spreads) >= 5:
                            features[11] = (np.mean(recent_spreads[-3:]) - np.mean(recent_spreads[-8:-5])) / (np.std(recent_spreads) + 1e-6)
                        
                        # Range position
                        spread_range = np.max(recent_spreads) - np.min(recent_spreads)
                        if spread_range > 0:
                            features[12] = (current_spread - np.min(recent_spreads)) / spread_range
                        
                        # Recent volatility change
                        if len(recent_spreads) >= 10:
                            vol1 = np.std(recent_spreads[-5:])
                            vol2 = np.std(recent_spreads[-10:-5])
                            features[13] = (vol1 - vol2) / (vol2 + 1e-6)
                
                # Risk metric (feature 14)
                if len(self.returns_history) > 5:
                    features[14] = np.mean(self.returns_history[-5:]) / (np.std(self.returns_history[-5:]) + 1e-6)
                
        except Exception as e:
            logger.warning(f"Error constructing observation: {e}")
        
        # Store state history
        self.state_history.append(features.copy())
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        return features
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Position: {self.position:.3f}, "
                  f"Cash: ${self.cash:.2f}, Total Value: ${self.total_value:.2f}")


class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            
            self.episode_rewards.append(mean_reward)
            self.episode_lengths.append(mean_length)
            
            if self.verbose > 0:
                logger.info(f"Mean episode reward: {mean_reward:.4f}, Mean length: {mean_length:.1f}")


class RLAgent:
    """Reinforcement Learning Agent for trading signal enhancement"""
    
    def __init__(self):
        self.model = None
        self.env = None
        self.is_trained = False
        self.training_history = {}
        
    def create_environment(self, 
                          price_data: pd.DataFrame,
                          pair_data: Dict) -> gym.Env:
        """
        Create trading environment for training/inference
        
        Args:
            price_data: Historical price data
            pair_data: Pair-specific data including spreads
            
        Returns:
            Gym environment
        """
        env = PairTradingEnv(price_data=price_data, pair_data=pair_data)
        return env
    
    def train_agent(self, 
                   price_data: pd.DataFrame,
                   pair_data: Dict,
                   total_timesteps: int = 50000,
                   learning_rate: float = 3e-4) -> Dict[str, Any]:
        """
        Train the PPO agent
        
        Args:
            price_data: Historical price data
            pair_data: Pair-specific training data
            total_timesteps: Number of training timesteps
            learning_rate: Learning rate for training
            
        Returns:
            Training statistics
        """
        logger.info("Starting RL agent training")
        
        try:
            # Create environment
            env = self.create_environment(price_data, pair_data)
            self.env = DummyVecEnv([lambda: env])
            
            # Create PPO model
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device='auto',
                tensorboard_log="./tensorboard_logs/"
            )
            
            # Training callback
            callback = TradingCallback(verbose=1)
            
            # Train the model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            self.is_trained = True
            
            # Store training history
            self.training_history = {
                'episode_rewards': callback.episode_rewards,
                'episode_lengths': callback.episode_lengths,
                'total_timesteps': total_timesteps,
                'final_mean_reward': np.mean(callback.episode_rewards[-10:]) if callback.episode_rewards else 0
            }
            
            logger.info(f"Training completed. Final mean reward: {self.training_history['final_mean_reward']:.4f}")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_action(self, 
                      state_features: np.ndarray,
                      deterministic: bool = True) -> Tuple[int, float]:
        """
        Predict trading action given state features
        
        Args:
            state_features: 15-dimensional state vector
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, confidence)
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, returning neutral action")
            return 1, 0.0  # Hold action
        
        try:
            # Ensure correct shape
            if state_features.ndim == 1:
                state_features = state_features.reshape(1, -1)
            
            # Get action prediction
            action, _states = self.model.predict(state_features, deterministic=deterministic)
            
            # Get action probabilities for confidence
            obs_tensor = torch.FloatTensor(state_features)
            if hasattr(self.model.policy, 'get_distribution'):
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = torch.softmax(distribution.logits, dim=-1)
                confidence = float(torch.max(probs))
            else:
                confidence = 0.5  # Default confidence
            
            return int(action[0]), confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 1, 0.0  # Hold action with low confidence
    
    def enhance_signal(self, 
                      base_signal: Dict,
                      state_features: np.ndarray) -> Dict[str, Any]:
        """
        Enhance base trading signal with RL predictions
        
        Args:
            base_signal: Base signal from OU model
            state_features: Current state features
            
        Returns:
            Enhanced signal with RL adjustments
        """
        if not self.is_trained:
            # Return base signal if not trained
            enhanced_signal = base_signal.copy()
            enhanced_signal['rl_enhancement'] = 'not_trained'
            return enhanced_signal
        
        try:
            # Get RL action
            rl_action, rl_confidence = self.predict_action(state_features)
            
            # Map action to signal: 0=sell (-1), 1=hold (0), 2=buy (1)
            rl_signal_map = {0: -1, 1: 0, 2: 1}
            rl_signal = rl_signal_map[rl_action]
            
            # Combine with base signal
            base_entry = base_signal.get('entry_signal', 0)
            base_confidence = base_signal.get('confidence', 0)
            
            # Signal combination logic
            if rl_confidence > 0.6:  # High RL confidence
                if rl_signal == 0:  # RL says hold
                    enhanced_entry = 0
                    enhanced_confidence = rl_confidence
                elif rl_signal == base_entry:  # RL agrees with base
                    enhanced_entry = base_entry
                    enhanced_confidence = min(base_confidence + rl_confidence * 0.5, 1.0)
                else:  # RL disagrees with base
                    if rl_confidence > base_confidence:
                        enhanced_entry = rl_signal
                        enhanced_confidence = rl_confidence
                    else:
                        enhanced_entry = base_entry
                        enhanced_confidence = base_confidence * 0.8  # Reduce confidence due to disagreement
            else:  # Low RL confidence, trust base more
                enhanced_entry = base_entry
                enhanced_confidence = base_confidence
            
            # Create enhanced signal
            enhanced_signal = base_signal.copy()
            enhanced_signal.update({
                'entry_signal': enhanced_entry,
                'confidence': enhanced_confidence,
                'rl_action': rl_action,
                'rl_signal': rl_signal,
                'rl_confidence': rl_confidence,
                'base_signal': base_entry,
                'base_confidence': base_confidence,
                'enhancement_type': 'rl_enhanced'
            })
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Signal enhancement failed: {e}")
            enhanced_signal = base_signal.copy()
            enhanced_signal['rl_enhancement'] = 'failed'
            return enhanced_signal
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to disk"""
        if self.model is None:
            return False
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk"""
        try:
            self.model = PPO.load(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of training progress"""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        rewards = self.training_history.get('episode_rewards', [])
        lengths = self.training_history.get('episode_lengths', [])
        
        for i, (reward, length) in enumerate(zip(rewards, lengths)):
            summary_data.append({
                'episode': i,
                'reward': reward,
                'length': length,
                'rolling_mean_reward': np.mean(rewards[max(0, i-10):i+1])
            })
        
        return pd.DataFrame(summary_data)


# Global instance
rl_agent = RLAgent() 