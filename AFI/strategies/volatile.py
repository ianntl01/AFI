# strategies/volatile.py
import pandas as pd
import talib as ta
import logging
from typing import Dict, Optional, Tuple
from .base_strat import BaseStrategy

logger = logging.getLogger(__name__)

class VolatileStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        """Initialize volatile strategy with configuration"""
        super().__init__(config)
        self.config = config or {}
        self.params = {
            'volatility_threshold': 0.04,
            'volume_multiplier': 1.5,
            'atr_period': 14,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'min_volume': 1000000,
            'stop_loss_multiplier': 2.0,
            'take_profit_multiplier': 3.0
        }
        self.tokens = {}

    def set_tokens(self, tokens: Dict):
        """Set the tokens being monitored"""
        self.tokens = tokens
        self.logger.info(f"Received {len(tokens)} approved tokens")

    def get_preferred_regime(self) -> str:
        """Return the preferred market regime for this strategy"""
        return 'Volatile'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for volatile strategy"""
        if not self._validate_data(df):
            return None
            
        try:
            # Calculate common metrics
            df = self._calculate_common_metrics(df)
            
            # ATR for volatility measurement
            df['atr'] = ta.ATR(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.params['atr_period']
            )
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
                df['close'],
                timeperiod=self.params['bb_period'],
                nbdevup=self.params['bb_std'],
                nbdevdn=self.params['bb_std']
            )
            
            # RSI
            df['rsi'] = ta.RSI(df['close'], timeperiod=self.params['rsi_period'])
            
            # Additional volatility metrics
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volatility_ma'] = df['volatility'].rolling(window=10).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def generate_signal(self, token: str, df: pd.DataFrame) -> Dict:
        """Generate trading signals for volatile market conditions"""
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

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[str]:
        """Generate entry signal for volatile conditions"""
        try:
            if token not in self.tokens:
                return None

            market_data = analysis.get('market_data', {})
            if not market_data:
                return None

            # Get indicators
            close = market_data.get('close', 0)
            bb_upper = market_data.get('bb_upper')
            bb_lower = market_data.get('bb_lower')
            volatility = market_data.get('volatility', 0)
            volatility_ma = market_data.get('volatility_ma', 0)
            volume = market_data.get('volume', 0)
            volume_ma = market_data.get('volume_ma', 1)

            if not all([bb_upper, bb_lower]):
                return None

            # Check minimum conditions
            if volume < self.params['min_volume']:
                return None
            if volatility < self.params['volatility_threshold']:
                return None

            # Generate signals based on breakouts
            if (volatility > volatility_ma * self.params['volatility_threshold'] and
                volume > volume_ma * self.params['volume_multiplier']):
                if close > bb_upper:
                    return 'buy'  # Upward breakout
                elif close < bb_lower:
                    return 'sell'  # Downward breakout

            return None

        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict) -> Optional[bool]:
        """Generate exit signal for volatile positions"""
        try:
            market_data = analysis.get('market_data', {})
            if not market_data:
                return None

            # Get indicators
            close = market_data.get('close', 0)
            bb_middle = market_data.get('bb_middle', 0)
            volatility = market_data.get('volatility', 0)
            volatility_ma = market_data.get('volatility_ma', 0)

            # Exit when volatility decreases or price reverts to mean
            if (volatility < volatility_ma or 
                abs(close - bb_middle) / bb_middle < 0.01):
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error generating exit signal: {e}")
            return None

    def get_stop_levels(self, token: str, analysis: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels for volatile conditions"""
        try:
            market_data = analysis.get('market_data', {})
            close = market_data.get('close', 0)
            atr = market_data.get('atr', 0)

            if not close or not atr:
                return (close * 0.95, close * 1.05)  # Default levels

            # Wider stops for volatile conditions
            stop_loss = close * (1 - self.params['stop_loss_multiplier'] * atr)
            take_profit = close * (1 + self.params['take_profit_multiplier'] * atr)

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
            self.params['bb_period'],
            self.params['atr_period'],
            self.params['rsi_period'],
            30  # For volatility calculation
        ) + 10  # Add buffer