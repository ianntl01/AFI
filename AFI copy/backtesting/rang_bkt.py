import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta
from typing import Tuple


# Fetch BTC historical data
def fetch_btc_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year timeframe
    
    btc = yf.Ticker("BTC-USD")
    df = btc.history(interval="1h", start=start_date, end=end_date)
    
    # Rename columns to match expected format
    df.index.name = 'Date'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Clean data - fill NaN values and ensure positive volume
    df['Volume'] = df['Volume'].fillna(0).clip(lower=0)
    
    print("\n=== Data Summary ===")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Sample data:\n{df.head(3)}")
    print(f"NaN values:\n{df.isna().sum()}")
    
    return df.astype(float)

data = fetch_btc_data()

# Strategy Implementation
class RangingStrategy(Strategy):
    # Parameters from ranging.py config
    bb_period = 20
    bb_std_dev = 2
    rsi_period = 14
    atr_period = 14
    support_resistance_periods = 15
    min_confidence = 0.6
    stop_loss_atr_mult = 1.5
    rsi_oversold = 40
    rsi_overbought = 60
    bb_pos_lower_threshold = 0.2
    bb_pos_upper_threshold = 0.8
    dist_to_support_threshold = 0.02
    dist_to_resistance_threshold = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Volatility indicators
        self.atr = self.I(ta.ATR, high, low, close, timeperiod=self.atr_period)
        
        # Momentum indicators
        self.rsi = self.I(ta.RSI, close, timeperiod=self.rsi_period)
        
        # Support/resistance levels
        def calc_support(low):
            s = pd.Series(low)
            return s.rolling(self.support_resistance_periods, center=True).min().values
        self.support = self.I(calc_support, low)
        
        def calc_resistance(high):
            s = pd.Series(high)
            return s.rolling(self.support_resistance_periods, center=True).max().values
        self.resistance = self.I(calc_resistance, high)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close,
                             timeperiod=self.bb_period,
                             nbdevup=self.bb_std_dev,
                             nbdevdn=self.bb_std_dev)
        self.bb_upper = self.I(lambda: bb_upper)
        self.bb_middle = self.I(lambda: bb_middle)
        self.bb_lower = self.I(lambda: bb_lower)
        
        # Bollinger Band position
        def calc_bb_pos(close, bb_upper, bb_lower):
            bb_range = bb_upper - bb_lower
            return np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)
        self.bb_position = self.I(calc_bb_pos, close, bb_upper, bb_lower)
        
        # Distance to support/resistance
        def calc_dist_to_support(close, support):
            return (close - support) / support
        self.dist_to_support = self.I(calc_dist_to_support, close, self.support)
        
        def calc_dist_to_resistance(resistance, close):
            return (resistance - close) / close
        self.dist_to_resistance = self.I(calc_dist_to_resistance, self.resistance, close)

    def next(self):
        # Risk management - 10% of capital per trade
        cash_available = self.equity
        btc_price = self.data.Close[-1]
        position_size = max(1, int((cash_available * 0.1) / btc_price))

        # Get current indicators
        close = self.data.Close[-1]
        rsi = self.rsi[-1]
        support = self.support[-1]
        resistance = self.resistance[-1]
        atr = self.atr[-1]

        # Debug output
        print(f"\n=== Signal Debug ===")
        print(f"Close: {close:.2f}, Support: {support:.2f}, Resistance: {resistance:.2f}")
        print(f"RSI: {rsi:.2f}, ATR: {atr:.2f}")
        print(f"Current position: {self.position}")

        # Entry signals with confidence scoring
        if not self.position:
            # Buy Conditions
            is_oversold = self.rsi[-1] < self.rsi_oversold
            is_at_bottom = self.bb_position[-1] < self.bb_pos_lower_threshold
            is_near_support = self.dist_to_support[-1] < self.dist_to_support_threshold
            buy_confirmations = [is_oversold, is_at_bottom, is_near_support]
            
            # Sell Conditions
            is_overbought = self.rsi[-1] > self.rsi_overbought
            is_at_top = self.bb_position[-1] > self.bb_pos_upper_threshold
            is_near_resistance = self.dist_to_resistance[-1] < self.dist_to_resistance_threshold
            sell_confirmations = [is_overbought, is_at_top, is_near_resistance]

            direction = None
            confidence = 0

            if all(buy_confirmations):
                direction = 'buy'
                confidence = sum(buy_confirmations) / len(buy_confirmations)
            elif all(sell_confirmations):
                direction = 'sell'
                confidence = sum(sell_confirmations) / len(sell_confirmations)

            if direction and confidence >= self.min_confidence:
                stop_loss, take_profit = self.get_stop_levels(direction)
                
                print(f"\n=== {direction.capitalize()} Entry ===")
                print(f"Price: {close:.2f}, Stop: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
                print(f"Confidence: {confidence:.2f}")
                
                if direction == 'buy':
                    self.buy(size=position_size)
                else:
                    self.sell(size=position_size)

    def get_stop_levels(self, direction: str) -> tuple[float, float]:
        """Calculate dynamic stop levels based on support/resistance and ATR."""
        close = self.data.Close[-1]
        support = self.support[-1]
        resistance = self.resistance[-1]
        atr = self.atr[-1]
        
        if direction == 'buy':
            stop_loss = support - (atr * self.stop_loss_atr_mult)
            take_profit = resistance - (atr * 0.5) # Target just below resistance
        else: # sell
            stop_loss = resistance + (atr * self.stop_loss_atr_mult)
            take_profit = support + (atr * 0.5) # Target just above support
            
        return (stop_loss, take_profit)

        # Exit conditions
        if self.position:
            reasons = []
            bb_pos = self.bb_position[-1]
            
            if self.position.is_long and bb_pos > self.bb_pos_upper_threshold:
                reasons.append("Reached top of Bollinger Band")
            elif self.position.is_short and bb_pos < self.bb_pos_lower_threshold:
                reasons.append("Reached bottom of Bollinger Band")
            
            if reasons:
                print(f"\n=== Exit Signal ===")
                print(f"Exit reasons: {', '.join(reasons)}")
                self.position.close()

# Run Backtest with increased cash for BTC prices
bt = Backtest(data, RangingStrategy, cash=1000000, commission=.002)
result = bt.run()

# Print performance metrics
print("\n=== Backtest Results ===")
print(f"Return: {result['Return [%]']:.2f}%")
print(f"Max. Drawdown: {result['Max. Drawdown [%]']:.2f}%")
print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
print(f"Win Rate: {result['Win Rate [%]']:.2f}%")
print(f"Trades: {result['# Trades']}")

# Plot performance
bt.plot()
