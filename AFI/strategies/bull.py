import ccxt
import pandas as pd
import numpy as np
import talib as ta
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from .base_strat import BaseStrategy

logger = logging.getLogger(__name__)

class BullStrategy(BaseStrategy):
    def __init__(self, config=None):
        super().__init__(config)
        # Strategy Parameters from second GA optimization
        self.params = {
            'ema_short': 5,
            'ema_long': 15,
            'macd_fast': 8,
            'macd_slow': 25,
            'macd_signal': 7,
            'rsi_period': 21,
            'rsi_oversold': 36,
            'rsi_overbought': 80,
            'bb_period': 14,
            'bb_std_dev': 1.55,
            'volume_threshold': 1.83,
            'signal_threshold': 0.66,
            'atr_period': 8,
            'trailing_stop_multiplier': 2.16,
            'max_hold_period': 11,
            'rsi_threshold': 60,
            'trend_threshold': 0.02,
            'min_volume': 1000000
        }
        self.approved_tokens = {}
        self.tokens = {}

    def set_tokens(self, approved_tokens: Dict):
        """Required by orchestrator to set approved tokens"""
        self.approved_tokens = approved_tokens
        self.tokens = approved_tokens

    def get_preferred_regime(self) -> str:
        """Required by orchestrator to identify strategy's preferred market regime"""
        return 'Bullish'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implement abstract method from BaseStrategy"""
        if not self._validate_data(df):
            return None
            
        try:
            # Calculate common metrics from base strategy
            df = self._calculate_common_metrics(df)
            
            # Trend Indicators
            df['ema_short'] = ta.EMA(df['close'], timeperiod=self.params['ema_short'])
            df['ema_long'] = ta.EMA(df['close'], timeperiod=self.params['ema_long'])
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
                df['close'], 
                fastperiod=self.params['macd_fast'], 
                slowperiod=self.params['macd_slow'], 
                signalperiod=self.params['macd_signal']
            )
            
            # Momentum Indicators
            df['rsi'] = ta.RSI(df['close'], timeperiod=self.params['rsi_period'])
            df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            
            # Volatility Indicators
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=self.params['atr_period'])
            upper, middle, lower = ta.BBANDS(
                df['close'], 
                timeperiod=self.params['bb_period'], 
                nbdevup=self.params['bb_std_dev'], 
                nbdevdn=self.params['bb_std_dev']
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # Volume Analysis
            df['volume_ma'] = ta.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = ta.STOCH(
                df['high'], 
                df['low'], 
                df['close'], 
                fastk_period=14, 
                slowk_period=3, 
                slowd_period=3
            )
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def generate_signal(self, token: str, df: pd.DataFrame) -> dict:
        """Implement abstract method from BaseStrategy"""
        try:
            market_data = df.iloc[-1].to_dict() if not df.empty else {}
            
            entry_signal = self.generate_entry_signal(token, {'market_data': market_data})
            exit_signal = self.generate_exit_signal(token, {'market_data': market_data})
            stop_loss, take_profit = self.get_stop_levels(token, {'market_data': market_data})
            
            return {
                'entry': entry_signal,
                'exit': exit_signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {}

    def select_token(self) -> str:
        """Implement abstract method from BaseStrategy"""
        try:
            if not self.approved_tokens:
                return None
            return max(self.approved_tokens.items(), key=lambda x: x[1])[0]
        except Exception as e:
            self.logger.error(f"Error selecting token: {e}")
            return None

    def get_required_data_size(self) -> int:
        """Override base method to specify required data points"""
        return max(
            self.params['ema_long'],
            self.params['macd_slow'],
            self.params['rsi_period'],
            self.params['bb_period'],
            20  # For volume MA
        ) + 10  # Add buffer

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[str]:
        """
        Generate entry signal for bullish market conditions
        Returns: 'buy', 'sell', or None
        """
        try:
            if token not in self.tokens:
                return None

            market_data = analysis.get('market_data', {})
            if not market_data:
                return None

            # Get key indicators
            rsi = market_data.get('rsi', 50)
            sma_20 = market_data.get('sma_20')
            sma_50 = market_data.get('sma_50')
            volume = market_data.get('volume', 0)
            macd = market_data.get('macd', 0)

            if not all([sma_20, sma_50]):
                return None

            # Check volume requirement
            if volume < self.params['min_volume']:
                return None

            # Calculate trend
            trend = (sma_20 - sma_50) / sma_50

            # Generate buy signal
            if (trend > self.params['trend_threshold'] and 
                rsi > self.params['rsi_threshold'] and 
                macd > 0):
                return 'buy'

            return None

        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict) -> Optional[bool]:
        """
        Generate exit signal for bullish market conditions
        Returns: True (exit), False (hold), or None (error)
        """
        try:
            market_data = analysis.get('market_data', {})
            if not market_data:
                return None

            # Get key indicators
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            sma_20 = market_data.get('sma_20')
            sma_50 = market_data.get('sma_50')

            if not all([sma_20, sma_50]):
                return None

            # Calculate trend
            trend = (sma_20 - sma_50) / sma_50

            # Exit conditions
            if trend < 0 or rsi < 40 or macd < 0:
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error generating exit signal: {e}")
            return None

    def get_stop_levels(self, token: str, analysis: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            market_data = analysis.get('market_data', {})
            close = market_data.get('close', 0)
            atr = market_data.get('atr', 0)

            # Default levels if can't calculate
            if not close or not atr:
                return (close * 0.95, close * 1.05)

            # Calculate levels based on ATR
            stop_loss = close * (1 - 1.5 * atr)
            take_profit = close * (1 + 3 * atr)  # Higher reward ratio for trend following

            return (stop_loss, take_profit)

        except Exception as e:
            self.logger.error(f"Error calculating stop levels: {e}")
            return (0, 0)