# strategies/volatile.py
import pandas as pd
import talib as ta
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from .base_strat import BaseStrategy

logger = logging.getLogger(__name__)

class VolatileStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        """Initialize volatile strategy with configuration"""
        super().__init__(config)
        self.name = "VolatileStrategy"
        self.config = config or {}
        self.params = {
            'volatility_threshold': 0.02,
            'volume_multiplier': 1.2,
            'atr_period': 14,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'min_volume': 100000,
            'stop_loss_atr_mult': 2.0,
            'take_profit_atr_mult': 3.0,
            'volatility_lookback': 20,
            'volume_lookback': 20,
            'bb_squeeze_threshold': 0.1,
            'momentum_threshold': 0.02,
            'min_volume': 100000,
            'signal_threshold': 0.3, # Confidence threshold
        }
        self.tokens = {}
        self.position_info = {}  # Track position entry info

    def set_tokens(self, tokens: Dict):
        """Set the tokens being monitored"""
        self.tokens = tokens
        logger.info(f"Received {len(tokens)} approved tokens")

    def get_preferred_regime(self) -> str:
        """Return the preferred market regime for this strategy"""
        return 'Volatile'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for volatile strategy"""
        if not self._validate_data(df):
            return None
            
        try:
            df = self._calculate_common_metrics(df)
            df['volume'] = df['volume'].fillna(0).replace(0, df['volume'].rolling(20).mean())
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=self.params['atr_period'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'], timeperiod=self.params['bb_period'], nbdevup=self.params['bb_std'], nbdevdn=self.params['bb_std'])
            df['rsi'] = ta.RSI(df['close'], timeperiod=self.params['rsi_period'])
            df['price_volatility'] = df['returns'].rolling(window=self.params['volatility_lookback']).std()
            df['volatility_ma'] = df['price_volatility'].rolling(window=10).mean()
            df['volatility_ratio'] = df['price_volatility'] / df['volatility_ma']
            df['volume_ma'] = df['volume'].rolling(window=self.params['volume_lookback']).mean()
            df['volume_ratio'] = np.where(df['volume_ma'] > 0, df['volume'] / df['volume_ma'], 1)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
            df['momentum'] = df['close'].pct_change(5)
            df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['vol_breakout'] = (df['volatility_ratio'] > 1.5) & (df['volatility_ratio'].shift(1) <= 1.5)
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[Dict]:
        """Generate entry signal for volatile conditions using pre-computed analysis."""
        try:
            market_data = analysis.get('market_data')
            if not market_data or analysis.get('regime') != 'Volatile':
                return None

            close = market_data.get('close', 0)
            if close <= 0 or market_data.get('price_volatility', 0) < self.params['volatility_threshold']:
                return None

            conditions = {
                'volatility': market_data.get('volatility_ratio', 0) > 1.3 or market_data.get('vol_breakout', False),
                'volume': market_data.get('volume_ratio', 1) > self.params['volume_multiplier'],
                'momentum_up': market_data.get('momentum', 0) > self.params['momentum_threshold'],
                'momentum_down': market_data.get('momentum', 0) < -self.params['momentum_threshold'],
                'upper_breakout': close > market_data.get('bb_upper', float('inf')),
                'lower_breakout': close < market_data.get('bb_lower', 0),
                'squeeze_breakout': market_data.get('bb_squeeze', False) and market_data.get('volatility_ratio', 0) > 1.2,
                'price_pos_high': market_data.get('price_position', 0.5) > 0.7,
                'price_pos_low': market_data.get('price_position', 0.5) < 0.3
            }

            direction = None
            buy_confidence_score = 0
            sell_confidence_score = 0

            if conditions['volatility'] and conditions['volume']:
                if conditions['upper_breakout'] and conditions['momentum_up']:
                    direction = 'buy'
                    buy_confidence_score = 0.8
                elif conditions['lower_breakout'] and conditions['momentum_down']:
                    direction = 'sell'
                    sell_confidence_score = 0.8
            
            if not direction and conditions['squeeze_breakout']:
                if conditions['price_pos_high'] and conditions['momentum_up']:
                    direction = 'buy'
                    buy_confidence_score = 0.6
                elif conditions['price_pos_low'] and conditions['momentum_down']:
                    direction = 'sell'
                    sell_confidence_score = 0.6

            confidence = max(buy_confidence_score, sell_confidence_score)

            if direction and confidence >= self.params['signal_threshold']:
                logger.info(f"Volatile {direction.upper()} signal for {token} with confidence {confidence:.2f}")
                stop_loss, take_profit = self.get_stop_levels(token, analysis, direction)
                return {
                    'direction': direction,
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
            logger.error(f"Error generating entry signal for {token}: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict, position: Dict) -> Optional[Dict]:
        """Generate exit signal for volatile positions using pre-computed analysis."""
        try:
            market_data = analysis.get('market_data')
            if not market_data or not position:
                return None

            close = market_data.get('close', 0)
            bb_middle = market_data.get('bb_middle', 0)
            if not all([close > 0, bb_middle > 0]):
                return None

            exit_reasons = []
            if abs(close - bb_middle) / bb_middle < 0.02:
                exit_reasons.append("Mean reversion detected")
            if market_data.get('volatility_ratio', 1) < 0.8:
                exit_reasons.append("Volatility collapse")
            
            rsi = market_data.get('rsi', 50)
            price_position_normal = 0.3 < market_data.get('price_position', 0.5) < 0.7
            if (rsi > 80 or rsi < 20) and price_position_normal:
                exit_reasons.append(f"RSI extreme ({rsi:.2f}) with position normalization")

            if exit_reasons:
                exit_direction = 'sell' if position.get('direction') == 'buy' else 'buy'
                logger.info(f"Volatile EXIT signal for {token}: {', '.join(exit_reasons)}")
                return {
                    'direction': exit_direction,
                    'strategy': self.name,
                    'exit_reason': ', '.join(exit_reasons)
                }

            return None
        except Exception as e:
            logger.error(f"Error generating exit signal for {token}: {e}")
            return None

    def get_stop_levels(self, token: str, analysis: Dict, direction: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels using pre-computed analysis."""
        try:
            market_data = analysis.get('market_data')
            if not market_data:
                return (None, None)

            close = market_data.get('close', 0)
            atr = market_data.get('atr', 0)
            if not close or not atr or close <= 0 or atr <= 0:
                return (None, None)

            vol_multiplier = min(2.5, max(1.0, market_data.get('volatility_ratio', 1)))
            stop_distance = atr * self.params['stop_loss_atr_mult'] * vol_multiplier
            profit_distance = atr * self.params['take_profit_atr_mult'] * vol_multiplier
            
            if direction == 'buy':
                stop_loss = close - stop_distance
                take_profit = close + profit_distance
            elif direction == 'sell':
                stop_loss = close + stop_distance
                take_profit = close - profit_distance
            else:
                return (None, None)

            return (stop_loss, take_profit)

        except Exception as e:
            logger.error(f"Error calculating stop levels for {token}: {e}")
            return (None, None)

    def select_token(self) -> str:
        """Select highest priority token"""
        try:
            if not self.tokens:
                return None
            return max(self.tokens.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"Error selecting token: {e}")
            return None

    def get_required_data_size(self) -> int:
        """Return required number of data points"""
        return max(
            self.params['bb_period'],
            self.params['atr_period'],
            self.params['rsi_period'],
            self.params['volatility_lookback'],
            self.params['volume_lookback']
        ) + 10  # For additional calculations
        + 10  # Add buffer