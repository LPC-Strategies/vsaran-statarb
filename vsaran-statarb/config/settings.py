"""
Configuration settings for the Δ-Ω Adaptive Convergence Strategy (DOAC)
Institutional-grade statistical arbitrage system
"""

import os
from typing import Dict, List, Any
from pydantic import BaseSettings, Field
from datetime import timedelta


class TradingConfig(BaseSettings):
    """Main trading configuration"""
    
    # Universe Engineering Parameters
    MIN_ADV: float = 25_000_000  # $25M minimum average daily volume
    MIN_MARKET_CAP: float = 2_000_000_000  # $2B minimum market cap
    MAX_BID_ASK_SPREAD: float = 0.0005  # 5 basis points
    MIN_QUOTE_UPDATE_RATE: int = 500  # milliseconds
    MAX_ROIC_DIVERGENCE: float = 0.15  # 15% max ROIC divergence
    MAX_GROSS_MARGIN_DIVERGENCE: float = 0.15  # 15% max gross margin divergence
    
    # Pair Discovery Parameters
    MIN_CORRELATION: float = 0.6  # 60-day Spearman correlation threshold
    CORRELATION_WINDOW: int = 60  # days
    JOHANSEN_P_VALUE: float = 0.01  # cointegration test threshold
    VARIANCE_RATIO_P_VALUE: float = 0.05  # random walk rejection threshold
    MAX_HALF_LIFE: int = 15  # maximum half-life in days
    MAX_HURST: float = 0.45  # maximum Hurst exponent
    MAX_IMPACT_TURNOVER: float = 0.0025  # 0.25% of NAV per day
    
    # Signal Generation Parameters
    ENTRY_Z_THRESHOLD: float = 2.2  # z-score entry threshold
    EXIT_Z_THRESHOLD: float = 0.4  # z-score exit threshold
    KALMAN_TRANSITION_COV: float = 1e-5
    KALMAN_OBSERVATION_COV: float = 1e-3
    OU_FITTING_WINDOW: int = 252  # days for OU parameter estimation
    
    # Reinforcement Learning Parameters
    RL_STATE_DIM: int = 15  # state vector dimension
    RL_LOOKBACK: int = 20  # days of historical data for state
    RL_REWARD_LAMBDA: float = 0.01  # inventory penalty coefficient
    RL_UPDATE_FREQ: int = 1000  # episodes between model updates
    
    # Meta-Filter Parameters
    META_FILTER_THRESHOLD: float = 0.58  # minimum win probability
    META_FILTER_FEATURES: int = 100  # number of engineered features
    
    # Risk Management Parameters
    MAX_POSITION_SIZE: float = 0.20  # 20% of NAV max position size
    MAX_GROSS_LEVERAGE: float = 4.0  # maximum gross leverage
    VAR_CONFIDENCE: float = 0.99  # 99% VaR confidence level
    VAR_HORIZON: int = 1  # 1-day VaR horizon
    TARGET_VAR_UTILIZATION: float = 0.75  # target 75% of VaR limit
    MAX_DRAWDOWN_LIMIT: float = 0.08  # 8% maximum drawdown
    
    # Execution Parameters
    TICK_UPDATE_FREQ: int = 250  # milliseconds between signal updates
    MAX_ORDER_LATENCY: int = 1  # maximum pre-trade check latency (ms)
    DEFAULT_SLIPPAGE: float = 0.000004  # 0.4 basis points
    TAKER_FEE: float = 0.0000025  # 0.25 basis points
    BORROW_COST: float = 0.0020  # 20 basis points per year
    KILL_SWITCH_Z_THRESHOLD: float = 4.0  # z-score kill switch threshold
    KILL_SWITCH_PNL_THRESHOLD: float = -0.0075  # -0.75% NAV 5-min PnL threshold
    
    # Data Parameters
    DATA_START_DATE: str = "2018-01-01"
    DATA_END_DATE: str = "2024-12-31"
    BACKTEST_START_DATE: str = "2020-01-01"
    REBALANCE_FREQUENCY: str = "weekly"  # universe rebalancing frequency
    
    # Performance Targets
    TARGET_SHARPE: float = 2.5
    TARGET_HIT_RATE: float = 0.55
    MAX_HOLDING_PERIOD: int = 4  # days
    TARGET_CAPACITY: float = 250_000_000  # $250M target capacity
    
    # API Configuration
    POLYGON_API_KEY: str = Field(default="", env="POLYGON_API_KEY")
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class DataConfig:
    """Data source configuration"""
    
    # Stock universes
    SP500_TICKERS: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
        "XOM", "JPM", "V", "PG", "MA", "CVX", "HD", "ABBV", "PFE", "BAC", "KO", "AVGO",
        "WMT", "DIS", "PEP", "TMO", "COST", "DHR", "VZ", "ABT", "NFLX", "ADBE", "CRM",
        "ACN", "CSCO", "MCD", "LIN", "PM", "WFC", "NEE", "BMW", "UPS", "QCOM", "ORCL",
        "LOW", "TXN", "BMY", "HON", "RTX", "MDT", "SPGI", "UNP", "INTU", "GS", "AMGN",
        "CAT", "IBM", "COP", "SCHW", "DE", "LMT", "MU", "AXP", "BLK", "BKNG", "ADI",
        "CVS", "SBUX", "GILD", "ADP", "PYPL", "SYK", "TJX", "VRTX", "ISRG", "AMT",
        "REGN", "FISV", "MMC", "ZTS", "C", "NOW", "PGR", "AON", "TGT", "BSX", "MO",
        "EQIX", "DUK", "ITW", "CCI", "SO", "ICE", "PLD", "KLAC", "NSC", "FCX", "EOG",
        "WM", "GM", "USB", "ATVI", "MCO", "APD", "SHW", "CME", "EMR", "GD", "BDX",
        "LRCX", "TFC", "NOC", "CL", "F", "PSA", "MAR", "ANET", "ORLY", "ECL", "HUM",
        "ADSK", "NXPI", "AEP", "MCHP", "O", "EXC", "BIIB", "ROP", "SRE", "ROST",
        "PAYX", "IDXX", "FAST", "KMB", "VRSK", "EA", "CTSH", "XEL", "CTAS", "PCAR",
        "KHC", "MSI", "IQV", "WELL", "OTIS", "DOW", "CMG", "CARR", "BK", "HPQ",
        "GIS", "MNST", "A", "WBA", "VIAC", "ADM", "PRU", "GLW", "DD", "YUM", "EBAY",
        "ILMN", "DLTR", "ED", "ETN", "EW", "APH", "ZBH", "DG", "ABC", "MTB", "WEC",
        "ALGN", "HRL", "DXCM", "EQR", "ES", "AME", "KEYS", "AVB", "PPG", "FTV",
        "ANSS", "CPRT", "ROK", "FITB", "HBAN", "STZ", "CLX", "WY", "TER", "VMC",
        "AWK", "MTD", "EFX", "RMD", "CERN", "TSCO", "EXPD", "K", "NUE", "FE", "WAB",
        "EXR", "BR", "NTRS", "TTWO", "TRMB", "ULTA", "MPWR", "CAH", "TDY", "ZBRA"
    ]
    
    LIQUID_ETFS: List[str] = [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "VTI", "VOO", "IVV", "VEA", "VWO",
        "GLD", "SLV", "TLT", "HYG", "LQD", "XLF", "XLE", "XLK", "XLV", "XLI"
    ]
    
    # Example pairs from the blueprint
    EXAMPLE_PAIRS: List[tuple] = [
        ("SPY", "IVV"),
        ("KO", "PEP"),
        ("NEM", "GDX"),
        ("MSFT", "MSFT")  # Dual-share example (placeholder)
    ]


# Global configuration instance
config = TradingConfig()
data_config = DataConfig() 