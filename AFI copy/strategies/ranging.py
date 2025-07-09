# strategies/ranging.py
import pandas as pd
import talib as ta
import numpy as np
import logging
from decimal import Decimal
from datetime import datetime
from .base_strat import BaseStrategy
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class RangingStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        """Initialize ranging strategy with configuration"""
        super().__init__(config)
        self.name = "RangingStrategy"
        self.config = config or {}
        self.position_info = {}  # Track position entry info
        self.params = {
            'bb_period': 20,
            'bb_std_dev': 2,
            'rsi_period': 14,
            'atr_period': 14,
            'support_resistance_periods': 10,  # Reduced from 15 to avoid NaN issues
            'min_confidence': 0.3,  # Reduced from 0.6 to allow more trades
            'stop_loss_atr_mult': 2.0,  # Increased for better risk management
            'rsi_oversold': 35,  # More extreme levels
            'rsi_overbought': 65,  # More extreme levels
            'bb_pos_lower_threshold': 0.15,  # More extreme positions
            'bb_pos_upper_threshold': 0.85,  # More extreme positions
            'dist_to_support_threshold': 0.005,  # Tighter proximity to S/R
            'dist_to_resistance_threshold': 0.005,  # Tighter proximity to S/R
            'min_range_width': 0.02,  # Minimum range width as % of price
            'max_hold_bars': 15,  # Maximum bars to hold position
            'profit_target_ratio': 2.0,  # Risk:reward ratio
            'volume_confirmation': True,  # Use volume for confirmation
            'volume_ma_period': 10  # Volume moving average period
        }
        self.tokens = {}

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ranging strategy."""
        if not self._validate_data(df):
            return None
        try:
            df = self._calculate_common_metrics(df)
            
            # Core indicators
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], self.params['atr_period'])
            df['rsi'] = ta.RSI(df['close'], self.params['rsi_period'])
            
            # Improved support/resistance calculation with multiple methods
            lookback = self.params['support_resistance_periods']
            
            # Method 1: Rolling min/max
            df['support_rolling'] = df['low'].rolling(lookback, center=True).min()
            df['resistance_rolling'] = df['high'].rolling(lookback, center=True).max()
            
            # Method 2: Pivot points
            df['support_pivot'] = df['low'].rolling(lookback*2+1, center=True).min()
            df['resistance_pivot'] = df['high'].rolling(lookback*2+1, center=True).max()
            
            # Combine methods and forward fill NaN values
            df['support'] = df[['support_rolling', 'support_pivot']].min(axis=1)
            df['resistance'] = df[['resistance_rolling', 'resistance_pivot']].max(axis=1)
            
            # Forward fill to handle NaN values
            df['support'] = df['support'].fillna(method='ffill').fillna(method='bfill')
            df['resistance'] = df['resistance'].fillna(method='ffill').fillna(method='bfill')
            
            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(
                df['close'], 
                timeperiod=self.params['bb_period'], 
                nbdevup=self.params['bb_std_dev'], 
                nbdevdn=self.params['bb_std_dev']
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # BB position calculation with safety checks
            bb_range = upper - lower
            df['bb_position'] = np.where(
                bb_range > 0, 
                (df['close'] - lower) / bb_range, 
                0.5
            )
            
            # Distance calculations with safety checks
            df['distance_to_support'] = np.where(
                df['support'] > 0,
                (df['close'] - df['support']) / df['support'],
                1.0
            )
            df['distance_to_resistance'] = np.where(
                df['close'] > 0,
                (df['resistance'] - df['close']) / df['close'],
                1.0
            )
            
            # Range width calculation
            df['range_width'] = np.where(
                df['close'] > 0,
                (df['resistance'] - df['support']) / df['close'],
                0
            )
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(self.params['volume_ma_period']).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
            else:
                df['volume_ratio'] = 1.0
            
            # Price momentum
            df['momentum'] = df['close'].pct_change(5)
            
            # Volatility squeeze indicator
            df['volatility_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {str(e)}")
            return None

    def set_tokens(self, tokens: Dict):
        self.tokens = tokens
        self.logger.info(f"Received {len(self.tokens)} approved tokens")

    def select_token(self) -> str:
        try:
            if not self.tokens:
                return None
            return max(self.tokens.items(), key=lambda x: x[1])[0]
        except Exception as e:
            self.logger.error(f"Token selection error: {str(e)}")
            return None

    def get_required_data_size(self) -> int:
        return max(self.params['bb_period'], self.params['support_resistance_periods'] * 2) + 20

    def get_preferred_regime(self) -> str:
        return 'Ranging'

    def generate_entry_signal(self, token: str, analysis: Dict) -> Optional[Dict]:
        """Generate a rich entry signal for ranging conditions."""
        try:
            market_data = analysis.get('market_data')
            if not market_data:
                return None
            
            # Allow trading in both Ranging and Neutral regimes
            regime = analysis.get('regime', '')
            if regime not in ['Ranging', 'Neutral']:
                return None

            # Check if we have valid support/resistance levels
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)
            close = market_data.get('close', 0)
            range_width = market_data.get('range_width', 0)
            
            if not all([support > 0, resistance > 0, close > 0, range_width > 0]):
                return None
            
            # Only trade if we have a meaningful range
            if range_width < self.params['min_range_width']:
                return None

            direction = None
            confidence = 0
            signal_strength = 0
            
            # Get indicators
            rsi = market_data.get('rsi', 50)
            bb_position = market_data.get('bb_position', 0.5)
            distance_to_support = market_data.get('distance_to_support', 1)
            distance_to_resistance = market_data.get('distance_to_resistance', 1)
            volume_ratio = market_data.get('volume_ratio', 1)
            momentum = market_data.get('momentum', 0)
            
            # Buy signal conditions
            buy_conditions = []
            buy_weights = []
            
            # RSI oversold
            if rsi < self.params['rsi_oversold']:
                buy_conditions.append(True)
                buy_weights.append(0.3)
                signal_strength += (self.params['rsi_oversold'] - rsi) / self.params['rsi_oversold']
            else:
                buy_conditions.append(False)
                buy_weights.append(0.3)
            
            # Near bottom of BB
            if bb_position < self.params['bb_pos_lower_threshold']:
                buy_conditions.append(True)
                buy_weights.append(0.25)
                signal_strength += (self.params['bb_pos_lower_threshold'] - bb_position) / self.params['bb_pos_lower_threshold']
            else:
                buy_conditions.append(False)
                buy_weights.append(0.25)
            
            # Near support
            if distance_to_support < self.params['dist_to_support_threshold']:
                buy_conditions.append(True)
                buy_weights.append(0.25)
                signal_strength += (self.params['dist_to_support_threshold'] - distance_to_support) / self.params['dist_to_support_threshold']
            else:
                buy_conditions.append(False)
                buy_weights.append(0.25)
            
            # Volume confirmation
            if self.params['volume_confirmation']:
                if volume_ratio > 1.1:  # Above average volume
                    buy_conditions.append(True)
                    buy_weights.append(0.1)
                else:
                    buy_conditions.append(False)
                    buy_weights.append(0.1)
            
            # Momentum not too negative
            if momentum > -0.02:  # Not falling too fast
                buy_conditions.append(True)
                buy_weights.append(0.1)
            else:
                buy_conditions.append(False)
                buy_weights.append(0.1)
            
            # Calculate weighted confidence for buy
            buy_confidence = sum(w for c, w in zip(buy_conditions, buy_weights) if c) / sum(buy_weights)
            
            # Sell signal conditions
            sell_conditions = []
            sell_weights = []
            
            # RSI overbought
            if rsi > self.params['rsi_overbought']:
                sell_conditions.append(True)
                sell_weights.append(0.3)
                signal_strength += (rsi - self.params['rsi_overbought']) / (100 - self.params['rsi_overbought'])
            else:
                sell_conditions.append(False)
                sell_weights.append(0.3)
            
            # Near top of BB
            if bb_position > self.params['bb_pos_upper_threshold']:
                sell_conditions.append(True)
                sell_weights.append(0.25)
                signal_strength += (bb_position - self.params['bb_pos_upper_threshold']) / (1 - self.params['bb_pos_upper_threshold'])
            else:
                sell_conditions.append(False)
                sell_weights.append(0.25)
            
            # Near resistance
            if distance_to_resistance < self.params['dist_to_resistance_threshold']:
                sell_conditions.append(True)
                sell_weights.append(0.25)
                signal_strength += (self.params['dist_to_resistance_threshold'] - distance_to_resistance) / self.params['dist_to_resistance_threshold']
            else:
                sell_conditions.append(False)
                sell_weights.append(0.25)
            
            # Volume confirmation
            if self.params['volume_confirmation']:
                if volume_ratio > 1.1:  # Above average volume
                    sell_conditions.append(True)
                    sell_weights.append(0.1)
                else:
                    sell_conditions.append(False)
                    sell_weights.append(0.1)
            
            # Momentum not too positive
            if momentum < 0.02:  # Not rising too fast
                sell_conditions.append(True)
                sell_weights.append(0.1)
            else:
                sell_conditions.append(False)
                sell_weights.append(0.1)
            
            # Calculate weighted confidence for sell
            sell_confidence = sum(w for c, w in zip(sell_conditions, sell_weights) if c) / sum(sell_weights)
            
            # Determine direction based on higher confidence
            if buy_confidence > sell_confidence and buy_confidence >= self.params['min_confidence']:
                direction = 'buy'
                confidence = buy_confidence
            elif sell_confidence > buy_confidence and sell_confidence >= self.params['min_confidence']:
                direction = 'sell'
                confidence = sell_confidence
            
            if direction:
                # Store position info
                self.position_info[token] = {
                    'entry_price': close,
                    'peak_price': close,
                    'trough_price': close,
                    'bars_held': 0,
                    'entry_time': datetime.now(),
                    'direction': direction
                }
                
                stop_loss, take_profit = self.get_stop_levels(token, analysis, direction)
                
                self.logger.info(f"Ranging ENTRY {direction.upper()} for {token}: "
                               f"Price={close:.2f}, RSI={rsi:.1f}, BB_pos={bb_position:.2f}, "
                               f"Conf={confidence:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
                
                return {
                    'direction': direction,
                    'strategy': self.name,
                    'market_data': market_data,
                    'regime': regime,
                    'strategy_regime': self.get_preferred_regime(),
                    'confidence': confidence,
                    'signal_strength': signal_strength,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'oco_enabled': True,
                    'entry_conditions': {
                        'rsi': rsi,
                        'bb_position': bb_position,
                        'distance_to_support': distance_to_support,
                        'distance_to_resistance': distance_to_resistance,
                        'range_width': range_width
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating entry signal for {token}: {e}")
            return None

    def generate_exit_signal(self, token: str, analysis: Dict, position: Dict) -> Optional[Dict]:
        """Generate a rich exit signal for ranging positions."""
        try:
            market_data = analysis.get('market_data')
            if not market_data or not position:
                return None

            reasons = []
            close = market_data.get('close', 0)
            
            # Update position tracking
            if token in self.position_info:
                pos_info = self.position_info[token]
                pos_info['bars_held'] += 1
                
                # Track peak/trough for trailing stops
                if pos_info['direction'] == 'buy':
                    pos_info['peak_price'] = max(pos_info['peak_price'], close)
                else:
                    pos_info['trough_price'] = min(pos_info['trough_price'], close)
            
            # Regime change exit
            regime = analysis.get('regime', '')
            if regime not in ['Ranging', 'Neutral']:
                reasons.append(f"Regime changed to {regime}")
            
            # BB position reversal
            bb_pos = market_data.get('bb_position', 0.5)
            position_side = position.get('side', '')
            
            if position_side == 'buy' and bb_pos > self.params['bb_pos_upper_threshold']:
                reasons.append("Reached upper BB threshold")
            elif position_side == 'sell' and bb_pos < self.params['bb_pos_lower_threshold']:
                reasons.append("Reached lower BB threshold")
            
            # RSI reversal
            rsi = market_data.get('rsi', 50)
            if position_side == 'buy' and rsi > self.params['rsi_overbought']:
                reasons.append("RSI overbought")
            elif position_side == 'sell' and rsi < self.params['rsi_oversold']:
                reasons.append("RSI oversold")
            
            # Time-based exit
            if token in self.position_info:
                bars_held = self.position_info[token]['bars_held']
                if bars_held >= self.params['max_hold_bars']:
                    reasons.append("Max hold period reached")
            
            # Support/resistance breach
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)
            
            if position_side == 'buy' and close < support * 0.995:  # Small buffer
                reasons.append("Support breached")
            elif position_side == 'sell' and close > resistance * 1.005:  # Small buffer
                reasons.append("Resistance breached")
            
            # Momentum reversal
            momentum = market_data.get('momentum', 0)
            if position_side == 'buy' and momentum < -0.03:  # Strong negative momentum
                reasons.append("Strong negative momentum")
            elif position_side == 'sell' and momentum > 0.03:  # Strong positive momentum
                reasons.append("Strong positive momentum")
            
            if reasons:
                exit_reason = ", ".join(reasons)
                exit_direction = 'sell' if position_side == 'buy' else 'buy'
                
                self.logger.info(f"Ranging EXIT {exit_direction.upper()} for {token}: {exit_reason}")
                
                # Clean up position info
                if token in self.position_info:
                    del self.position_info[token]
                
                return {
                    'direction': exit_direction,
                    'strategy': self.name,
                    'exit_reason': exit_reason,
                    'market_data': market_data,
                    'regime': regime,
                    'exit_conditions': {
                        'rsi': rsi,
                        'bb_position': bb_pos,
                        'momentum': momentum,
                        'price': close
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating exit signal for {token}: {e}")
            return None

    def get_stop_levels(self, token: str, analysis: Dict, direction: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate dynamic stop levels based on support/resistance and ATR."""
        try:
            market_data = analysis.get('market_data')
            if not market_data:
                return (None, None)

            close = market_data.get('close', 0)
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)
            atr = market_data.get('atr', 0)
            
            if not all([close > 0, support > 0, resistance > 0, atr > 0]):
                return (None, None)
            
            if direction == 'buy':
                # Stop loss below support with ATR buffer
                stop_loss = support - (atr * self.params['stop_loss_atr_mult'])
                # Take profit below resistance with smaller buffer
                take_profit = resistance - (atr * 0.5)
                
                # Ensure minimum risk:reward ratio
                risk = close - stop_loss
                reward = take_profit - close
                if reward < risk * self.params['profit_target_ratio']:
                    take_profit = close + (risk * self.params['profit_target_ratio'])
                
            elif direction == 'sell':
                # Stop loss above resistance with ATR buffer
                stop_loss = resistance + (atr * self.params['stop_loss_atr_mult'])
                # Take profit above support with smaller buffer
                take_profit = support + (atr * 0.5)
                
                # Ensure minimum risk:reward ratio
                risk = stop_loss - close
                reward = close - take_profit
                if reward < risk * self.params['profit_target_ratio']:
                    take_profit = close - (risk * self.params['profit_target_ratio'])
            else:
                return (None, None)

            # Ensure stop loss and take profit are reasonable
            if direction == 'buy':
                stop_loss = max(stop_loss, close * 0.95)  # Max 5% loss
                take_profit = min(take_profit, close * 1.15)  # Max 15% gain
            else:
                stop_loss = min(stop_loss, close * 1.05)  # Max 5% loss
                take_profit = max(take_profit, close * 0.85)  # Max 15% gain

            return (stop_loss, take_profit)
            
        except Exception as e:
            self.logger.error(f"Stop level calculation failed for {token}: {str(e)}")
            return (None, None)