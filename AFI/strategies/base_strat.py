# strategies/base_strategy.py
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, config: Dict = None):
        """Initialize base strategy"""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.approved_tokens = []
        self.current_regime = None
        self.timeframe = '15m'  # Default timeframe

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data"""
        if df is None or df.empty:
            return False
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_columns)

    def _calculate_common_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators"""
        try:
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Volume metrics
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating common metrics: {e}")
            return df

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        pass

    @abstractmethod
    def generate_signal(self, token: str, df: pd.DataFrame) -> Dict:
        """Generate trading signals"""
        pass

    @abstractmethod
    def select_token(self) -> str:
        """Select token to trade"""
        pass

    @abstractmethod
    def get_required_data_size(self) -> int:
        """Return required number of data points"""
        pass

    @abstractmethod
    def set_tokens(self, tokens: Dict):
        """Set approved tokens"""
        pass

    @abstractmethod
    def get_preferred_regime(self) -> str:
        """Return preferred market regime"""
        pass

    def set_timeframe(self, timeframe: str):
        """Update strategy timeframe"""
        self.timeframe = timeframe