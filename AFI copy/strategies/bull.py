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
        self.name = "BullStrategy"
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
            'signal_threshold': 0.6,
            'atr_period': 8,
            'trailing_stop_multiplier': 2.16,
            'max_hold_period': 11,
            'rsi_threshold': 60,
            'trend_threshold': 0.02,
            'min_volume': 100000
        }
        self.approved_tokens = {}
        self.tokens = {}
        self.position_info = {}  # Track position entry info (price, peak, bars held)

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
            logger.error(f"Error calculating indicators: {e}")
            return None

    def select_token(self) -> str:
        """Implement abstract method from BaseStrategy"""
        try:
            if not self.approved_tokens:
                return None
            return max(self.approved_tokens.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"Error selecting token: {e}")
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

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[Dict]:
        """
        Generate entry signal for bullish market conditions using pre-computed analysis.
        """
        try:
            market_data = analysis.get('market_data')
            if not market_data:
                return None

            # Get key indicators from provided market_data
            close = market_data.get('close', 0)
            rsi = market_data.get('rsi', 50)
            sma_20 = market_data.get('sma_20')
            sma_50 = market_data.get('sma_50')
            volume = market_data.get('volume', 0)
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            ema_short = market_data.get('ema_short')
            ema_long = market_data.get('ema_long')
            bb_lower = market_data.get('bb_lower')

            if not all([sma_20, sma_50, ema_short, ema_long, bb_lower, close > 0]):
                return None

            # Check volume requirement
            volume_check = (volume > self.params['min_volume']) or (self.params['min_volume'] == 0)

            # Confirmation methods
            confirmations = [
                ((sma_20 - sma_50) / sma_50) > (self.params['trend_threshold'] * 0.8),
                rsi > (self.params['rsi_threshold'] * 0.8),
                macd > macd_signal,
                ema_short > ema_long,
                close > bb_lower
            ]
            
            indicator_confirmations = sum(confirmations)
            confidence = indicator_confirmations / len(confirmations)

            if volume_check and confidence >= self.params['signal_threshold']:
                self.position_info[token] = {
                    'entry_price': close,
                    'peak_price': close,
                    'bars_held': 0,
                    'entry_time': datetime.now()
                }
                stop_loss, take_profit = self.get_stop_levels(token, analysis, 'buy')
                return {
                    'direction': 'buy',
                    'strategy': self.name,
                    'market_data': market_data,
                    'regime': analysis.get('regime'),
                    'strategy_regime': self.get_preferred_regime(),
                    'confidence': confidence,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'oco_enabled': True
                }

            return None
        except Exception as e:
            logger.error(f"Error in generate_entry_signal for {token}: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict, position: Dict) -> Optional[Dict]:
        """
        Generate exit signal for bullish market conditions using pre-computed analysis and position data.
        """
        try:
            market_data = analysis.get('market_data')
            if not market_data or not position:
                return None

            # Get key indicators from provided market_data
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            sma_20 = market_data.get('sma_20')
            sma_50 = market_data.get('sma_50')
            atr = market_data.get('atr', 0)
            close = market_data.get('close', 0)
            
            # Get position data
            entry_price = position.get('entry_price')

            if not all([sma_20, sma_50, atr, close > 0, entry_price]):
                return None

            exit_reasons = []
            
            # Exit conditions
            if ((sma_20 - sma_50) / sma_50) < 0:
                exit_reasons.append("Trend reversal: SMA20 crossed below SMA50")
            if (macd < macd_signal) and (macd > 0):
                exit_reasons.append("MACD sell signal")
            if rsi > self.params['rsi_overbought']:
                exit_reasons.append(f"RSI overbought: {rsi:.2f} > {self.params['rsi_overbought']}")
            if rsi < self.params['rsi_oversold']:
                exit_reasons.append(f"RSI weak: {rsi:.2f} < {self.params['rsi_oversold']}")
            
            # Trailing stop logic using internal state
            if token in self.position_info:
                peak_price = self.position_info[token].get('peak_price', entry_price)
                if close > peak_price:
                    self.position_info[token]['peak_price'] = close
                else:
                    trailing_stop = peak_price - (self.params['trailing_stop_multiplier'] * atr)
                    if close < trailing_stop:
                        exit_reasons.append(f"Trailing stop triggered at {trailing_stop:.2f}")
            
            # Time-based exit
            if token in self.position_info:
                bars_held = self.position_info[token].get('bars_held', 0) + 1
                self.position_info[token]['bars_held'] = bars_held
                if bars_held >= self.params['max_hold_period']:
                    exit_reasons.append(f"Max hold period of {self.params['max_hold_period']} bars reached")

            if exit_reasons:
                # Clean up position info on exit
                if token in self.position_info:
                    del self.position_info[token]
                return {
                    'direction': 'sell',
                    'strategy': self.name,
                    'exit_reason': ", ".join(exit_reasons)
                }

            return None
        except Exception as e:
            logger.error(f"Error in generate_exit_signal for {token}: {e}")
            return None

    def get_stop_levels(self, token: str, analysis: Dict, direction: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels based on ATR and direction."""
        try:
            market_data = analysis.get('market_data', {})
            close = market_data.get('close', 0)
            atr = market_data.get('atr', 0)

            if not close or not atr:
                default_sl = close * 0.95 if direction == 'buy' else close * 1.05
                default_tp = close * 1.05 if direction == 'buy' else close * 0.95
                return (default_sl, default_tp)

            stop_mult = self.params.get('trailing_stop_multiplier', 2.0)
            profit_mult = stop_mult * 1.5

            if direction == 'buy':
                stop_loss = close - (stop_mult * atr)
                take_profit = close + (profit_mult * atr)
            elif direction == 'sell':
                stop_loss = close + (stop_mult * atr)
                take_profit = close - (profit_mult * atr)
            else:
                return (0, 0)

            return (stop_loss, take_profit)

        except Exception as e:
            logger.error(f"Error calculating stop levels for {token}: {e}")
            return (0, 0)
