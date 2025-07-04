# strategies/ranging.py
import pandas as pd
import talib as ta
import logging
from decimal import Decimal
from .base_strat import BaseStrategy
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class RangingStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        """Initialize ranging strategy with configuration"""
        super().__init__(config)
        self.config = config or {}
        self.range_threshold = self.config.get('range_threshold', 0.03)
        self.volume_threshold = self.config.get('volume_threshold', 800000)
        self.tokens = {}
        
        # Strategy parameters
        self.params = {
            'bb_period': 20,
            'bb_std_dev': 2,
            'rsi_period': 14,
            'adx_period': 14,
            'atr_period': 14,
            'ema_short': 50,
            'ema_long': 200,
            'min_confidence': 0.65,
            'stop_loss_multiplier': 1.8,
            'take_profit_ratio': 2.5,
            'volume_threshold': 1.2,
            'adx_trend_threshold': 25
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if not self._validate_data(df):
            return None
            
        try:
            # Calculate common metrics from base strategy
            df = self._calculate_common_metrics(df)
            
            # Volatility indicators
            df['atr'] = ta.ATR(
                df['high'], df['low'], df['close'],
                self.params['atr_period']
            )
            df['atr_ma'] = df['atr'].rolling(5).mean()
            
            # Trend indicators
            df['ema_short'] = ta.EMA(df['close'], self.params['ema_short'])
            df['ema_long'] = ta.EMA(df['close'], self.params['ema_long'])
            
            # Momentum indicators
            df['rsi'] = ta.RSI(df['close'], self.params['rsi_period'])
            df['rsi_smooth'] = df['rsi'].rolling(5).mean()
            df['adx'] = ta.ADX(
                df['high'], df['low'], df['close'],
                self.params['adx_period']
            )
            
            # Support/resistance levels using recent price extremes
            lookback = min(20, len(df))
            df['support'] = df['low'].rolling(lookback).min()
            df['resistance'] = df['high'].rolling(lookback).max()
            
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Bollinger Bands for ranging detection
            upper, middle, lower = ta.BBANDS(
                df['close'],
                timeperiod=self.params['bb_period'],
                nbdevup=self.params['bb_std_dev'],
                nbdevdn=self.params['bb_std_dev']
            )
            df['bb_upper'] = upper
            df['bb_lower'] = lower
            df['bb_middle'] = middle
            df['bb_width'] = (upper - lower) / middle
            
            return df
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {str(e)}")
            return None

    def generate_signal(self, token: str, df: pd.DataFrame) -> Dict:
        """Generate complete trading signal"""
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

    def set_tokens(self, tokens: Dict):
        """Set the tokens being monitored"""
        self.tokens = tokens
        self.logger.info(f"Received {len(self.tokens)} approved tokens")

    def select_token(self) -> str:
        """Select highest priority token from approved list"""
        try:
            if not self.tokens:
                return None
            return max(self.tokens.items(), key=lambda x: x[1])[0]
        except Exception as e:
            self.logger.error(f"Token selection error: {str(e)}")
            return None

    def get_required_data_size(self) -> int:
        """Return number of bars needed for calculations"""
        return max(
            self.params['ema_long'],
            self.params['atr_period'],
            self.params['adx_period'],
            100  # Default safety minimum
        )

    def get_preferred_regime(self) -> str:
        """Return the preferred market regime for this strategy"""
        return 'Ranging'

    def get_stop_levels(self, token: str, analysis: dict) -> tuple:
        """Calculate dynamic stop levels"""
        try:
            market_data = analysis.get('market_data', {})
            if analysis.get('signal') == 'buy':
                return (
                    market_data['support'] - (market_data['atr'] * self.params['stop_loss_multiplier']),
                    market_data['close'] + ((market_data['close'] - market_data['support']) * self.params['take_profit_ratio'])
                )
            else:
                return (
                    market_data['resistance'] + (market_data['atr'] * self.params['stop_loss_multiplier']),
                    market_data['close'] - ((market_data['resistance'] - market_data['close']) * self.params['take_profit_ratio'])
                )
        except Exception as e:
            self.logger.error(f"Stop level calculation failed for {token}: {str(e)}")
            return (None, None)

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[str]:
        """Generate entry signal for ranging market conditions"""
        try:
            if token not in self.tokens:
                return None

            market_data = analysis.get('market_data', {})
            if not market_data:
                return None

            # Get indicators
            close = market_data.get('close', 0)
            rsi = market_data.get('rsi_smooth', 50)
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)
            volume = market_data.get('volume', 0)
            volume_ma = market_data.get('volume_ma', 1)
            adx = market_data.get('adx', 30)

            # Check volume and trend strength
            if volume < volume_ma * self.params['volume_threshold']:
                return None
            if adx > self.params['adx_trend_threshold']:
                return None

            # Generate signals
            if close <= support and rsi < 35:
                return 'buy'
            elif close >= resistance and rsi > 65:
                return 'sell'

            return None

        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict) -> Optional[bool]:
        """Generate exit signal for ranging market conditions"""
        try:
            market_data = analysis.get('market_data', {})
            if not market_data:
                return None

            # Get indicators
            adx = market_data.get('adx', 30)
            rsi = market_data.get('rsi_smooth', 50)
            atr = market_data.get('atr', 0)
            atr_ma = market_data.get('atr_ma', atr)

            # Exit conditions
            if adx > self.params['adx_trend_threshold']:  # Market starts trending
                return True
            if atr > 2 * atr_ma:  # Volatility spike
                return True
            if rsi > 75 or rsi < 25:  # Extreme RSI
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error generating exit signal: {e}")
            return None
