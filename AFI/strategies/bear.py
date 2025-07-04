# strategies/bear.py
import pandas as pd
import talib as ta
import logging
from typing import Dict, Optional, Tuple
from .base_strat import BaseStrategy

logger = logging.getLogger(__name__)

class BearStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        """Initialize bear strategy with configuration"""
        super().__init__(config)
        self.config = config or {}
        default_params = {
            'rsi_threshold': 40, # RSI level to enter a trade
            'rsi_exit_threshold': 50, # RSI level to exit a trade
            'min_volume': 1000000,
            'stop_loss_multiplier': 2.0,  # ATR multiplier for stop loss
            'take_profit_multiplier': 3.0  # ATR multiplier for take profit
        }
        # Overwrite default params with any provided in config
        self.params = {**default_params, **self.config.get('params', {})}
        self.tokens = {}

    def set_tokens(self, tokens: Dict):
        """Set the tokens being monitored"""
        self.tokens = tokens
        self.logger.info(f"Received {len(tokens)} approved tokens")

    def get_preferred_regime(self) -> str:
        """Return the preferred market regime for this strategy"""
        return 'Bearish'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for bearish strategy"""
        if not self._validate_data(df):
            return None
            
        try:
            # Calculate common metrics
            df = self._calculate_common_metrics(df)
            
            # Trend indicators
            df['ema_short'] = ta.EMA(df['close'], timeperiod=8)
            df['ema_long'] = ta.EMA(df['close'], timeperiod=21)
            df['long_term_trend'] = ta.SMA(df['close'], timeperiod=100)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
                df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            
            # RSI
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            
            # ATR for volatility
            df['atr'] = ta.ATR(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=14
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def generate_signal(self, token: str, df: pd.DataFrame) -> Dict:
        """Generate trading signals for bearish strategy"""
        try:
            if df.empty or len(df) < 2:
                return {}

            entry_signal = self.generate_entry_signal(token, df)
            exit_signal = self.generate_exit_signal(token, df)
            stop_loss, take_profit = self.get_stop_levels(token, df)
            
            return {
                'entry': entry_signal,
                'exit': exit_signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {}

    def generate_entry_signal(self, token: str, df: pd.DataFrame) -> Optional[str]:
        """Generate entry signal for bearish conditions"""
        try:
            if token not in self.tokens:
                return None

            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            # Get indicators from the last row
            volume = last_row.get('volume', 0)
            volume_ma = last_row.get('volume_ma', 1)
            long_term_trend = last_row.get('long_term_trend')
            close = last_row.get('close')

            if not all([long_term_trend, close, volume_ma]):
                return None

            # Check volume requirement
            if volume < self.params['min_volume']:
                return None

            # Entry condition for short position
            is_long_term_bearish = close < long_term_trend
            rsi_cross_down = prev_row['rsi'] >= self.params['rsi_threshold'] and last_row['rsi'] < self.params['rsi_threshold']

            # Simplified entry signal: Long-term trend, RSI crossover, and Volume spike
            if is_long_term_bearish and rsi_cross_down and (volume > volume_ma):
                return 'sell'

            return None

        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return None

    def generate_exit_signal(self, token: str, df: pd.DataFrame) -> Optional[bool]:
        """Generate exit signal for bearish positions"""
        try:
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            # Exit condition: RSI crosses above the exit threshold
            rsi_cross_up = prev_row['rsi'] <= self.params['rsi_exit_threshold'] and last_row['rsi'] > self.params['rsi_exit_threshold']
            
            if rsi_cross_up:
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error generating exit signal: {e}")
            return None

    def get_stop_levels(self, token: str, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels for bearish positions"""
        try:
            last_row = df.iloc[-1]
            close = last_row.get('close', 0)
            atr = last_row.get('atr', 0)

            if not close or not atr:
                return (close * 1.05, close * 0.95)  # Default levels

            # For short positions:
            stop_loss = close + (self.params['stop_loss_multiplier'] * atr)
            take_profit = close - (self.params['take_profit_multiplier'] * atr)

            return (stop_loss, take_profit)

        except Exception as e:
            self.logger.error(f"Error calculating stop levels: {e}")
            return (0, 0)

    def select_token(self) -> str:
        """Select highest priority token"""
        try:
            if not self.tokens:
                return None
            return max(self.tokens.items(), key=lambda x: x[1])[0]
        except Exception as e:
            self.logger.error(f"Error selecting token: {e}")
            return None

    def get_required_data_size(self) -> int:
        """Return required number of data points"""
        return max(
            21, # ema_long
            26, # macd_slow
            14, # rsi_period
            14  # atr_period
        ) + 10  # Add buffer