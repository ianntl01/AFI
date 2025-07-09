# strategies/bear.py
import pandas as pd
import talib as ta
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from .base_strat import BaseStrategy

logger = logging.getLogger(__name__)

class BearStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        """Initialize bear strategy with configuration"""
        super().__init__(config)
        self.name = "BearStrategy"
        self.config = config or {}
        # Parameters for signal generation
        default_params = {
            'rsi_threshold': 40,
            'rsi_exit_threshold': 50,
            'min_volume': 100000,
            'stop_loss_atr_mult': 2.0,
            'take_profit_atr_mult': 3.0,
            'adx_strong_trend': 25,
            'min_confirmations': 4
        }
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
            self.logger.error("Data validation failed.")
            return None
            
        try:
            df = self._calculate_common_metrics(df)
            df['ema_short'] = ta.EMA(df['close'], timeperiod=8)
            df['ema_long'] = ta.EMA(df['close'], timeperiod=21)
            df['long_term_trend'] = ta.SMA(df['close'], timeperiod=100)
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'], fastperiod=8, slowperiod=17, signalperiod=9)
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=25)
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = np.where(df['volume_ma'] > 0, df['volume'] / df['volume_ma'], 1)
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[Dict]:
        """Generate a rich entry signal for bearish conditions."""
        try:
            market_data = analysis.get('market_data')
            prev_market_data = analysis.get('prev_market_data')
            if not market_data or not prev_market_data or analysis.get('regime') != 'Bearish':
                return None

            volume = market_data.get('volume', 0)
            long_term_trend = market_data.get('long_term_trend')
            close = market_data.get('close')

            if not all([long_term_trend, close]) or volume < self.params['min_volume']:
                return None

            # Entry conditions
            is_long_term_bearish = close < long_term_trend
            rsi_cross_down = prev_market_data.get('rsi', 50) >= self.params['rsi_threshold'] and market_data.get('rsi', 50) < self.params['rsi_threshold']
            strong_trend = market_data.get('adx', 0) > self.params['adx_strong_trend']
            macd_bearish = market_data.get('macd_hist', 0) < 0
            bb_range = market_data.get('bb_upper', 0) - market_data.get('bb_lower', 0)
            bb_position = (close - market_data.get('bb_lower', 0)) / bb_range if bb_range > 0 else 0.5
            bb_bearish = bb_position > 0.7
            volume_spike = market_data.get('volume_ratio', 0) > 1.5

            confirmations = [is_long_term_bearish, rsi_cross_down, strong_trend, macd_bearish, bb_bearish, volume_spike]
            confidence = sum(confirmations) / len(confirmations)

            if confidence * len(confirmations) >= self.params['min_confirmations']:
                self.logger.info(f"Strong BEAR signal for {token} with confidence {confidence:.2f}")
                stop_loss, take_profit = self.get_stop_levels(token, analysis, 'sell')
                return {
                    'direction': 'sell',
                    'strategy': self.name,
                    'confidence': confidence,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'oco_enabled': True
                }
            return None
        except Exception as e:
            self.logger.error(f"Error generating entry signal for {token}: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict, position: Dict) -> Optional[Dict]:
        """Generate a rich exit signal for bearish positions."""
        try:
            market_data = analysis.get('market_data')
            prev_market_data = analysis.get('prev_market_data')
            if not market_data or not prev_market_data or not position:
                return None

            exit_triggers = []
            rsi_cross_up = prev_market_data.get('rsi', 50) <= self.params['rsi_exit_threshold'] and market_data.get('rsi', 50) > self.params['rsi_exit_threshold']
            if rsi_cross_up:
                exit_triggers.append("RSI exit threshold reached")

            if market_data.get('adx', 100) < 20:
                exit_triggers.append("Trend weakening (ADX < 20)")

            macd_crossover = prev_market_data.get('macd', 0) < prev_market_data.get('macd_signal', 0) and market_data.get('macd', 0) > market_data.get('macd_signal', 0)
            if macd_crossover:
                exit_triggers.append("MACD bullish crossover")
            
            if exit_triggers:
                reason = ", ".join(exit_triggers)
                self.logger.info(f"Exiting {token} due to: {reason}")
                # For a 'sell' position, the exit direction is 'buy'
                exit_direction = 'buy' if position.get('side') == 'sell' else 'sell'
                return {
                    'direction': exit_direction,
                    'strategy': self.name,
                    'exit_reason': reason
                }
            return None
        except Exception as e:
            self.logger.error(f"Error generating exit signal for {token}: {e}")
            return None

    def get_stop_levels(self, token: str, analysis: Dict, direction: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        try:
            market_data = analysis.get('market_data')
            if not market_data or direction != 'sell':
                return (None, None)

            close = market_data.get('close', 0)
            atr = market_data.get('atr', 0)
            adx = market_data.get('adx', 25)

            if not all([close > 0, atr > 0]):
                return (None, None)

            trend_strength = min(1.5, max(0.8, adx / 25.0))
            stop_mult = self.params['stop_loss_atr_mult'] * trend_strength
            profit_mult = self.params['take_profit_atr_mult'] * trend_strength

            stop_loss = close + (stop_mult * atr)
            take_profit = close - (profit_mult * atr)

            self.logger.info(f"Dynamic stops for {token}: SL={stop_loss:.2f}, TP={take_profit:.2f}")
            return (stop_loss, take_profit)
        except Exception as e:
            self.logger.error(f"Error calculating stop levels for {token}: {e}")
            return (None, None)

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
        # Based on the longest indicator period, which is the 100-period SMA for long_term_trend
        return 100 + 10 # Add buffer
