import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta

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
class VolatileStrategy(Strategy):
    # Parameters from volatile.py config
    volatility_threshold = 0.02
    volume_multiplier = 1.2
    atr_period = 14
    rsi_period = 14
    bb_period = 20
    bb_std = 2
    min_volume = 500000
    stop_loss_atr_mult = 2.0
    take_profit_atr_mult = 3.0
    volatility_lookback = 20
    volume_lookback = 20
    bb_squeeze_threshold = 0.1
    momentum_threshold = 0.02
    signal_threshold = 0.3

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Volatility indicators
        self.atr = self.I(ta.ATR, high, low, close, timeperiod=self.atr_period)
        
        # Calculate returns and volatility
        log_prices = np.log(close)
        returns = np.concatenate([[np.nan], np.diff(log_prices)])
        self.volatility = self.I(lambda: pd.Series(returns).rolling(self.volatility_lookback).std().values)
        self.volatility_ma = self.I(lambda: pd.Series(self.volatility).rolling(10).mean().values)
        self.volatility_ratio = self.I(lambda: self.volatility / self.volatility_ma)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close,
                             timeperiod=self.bb_period,
                             nbdevup=self.bb_std,
                             nbdevdn=self.bb_std,
                             matype=0)
        self.bb_upper = self.I(lambda: bb_upper)
        self.bb_middle = self.I(lambda: bb_middle)
        self.bb_lower = self.I(lambda: bb_lower)
        self.bb_width = self.I(lambda: (bb_upper - bb_lower) / bb_middle)
        self.bb_squeeze = self.I(lambda: self.bb_width < pd.Series(self.bb_width).rolling(20).mean().values * 0.8)
        
        # Momentum indicators
        self.rsi = self.I(ta.RSI, close, timeperiod=self.rsi_period)
        self.momentum = self.I(lambda: pd.Series(close).pct_change(5).values)
        self.price_position = self.I(lambda: (close - bb_lower) / (bb_upper - bb_lower))
        self.vol_breakout = self.I(lambda: (self.volatility_ratio > 1.5) & (pd.Series(self.volatility_ratio).shift(1) <= 1.5))
        
        # Volume analysis
        self.volume_ma = self.I(ta.SMA, volume, self.volume_lookback)
        self.volume_ratio = self.I(lambda: np.where(self.volume_ma > 0, volume / self.volume_ma, 1))

    def next(self):
        # Risk management - 5% of capital per trade (more aggressive for volatile strategy)
        cash_available = self.equity
        btc_price = self.data.Close[-1]
        position_size = max(1, int((cash_available * 0.05) / btc_price))

        # Get current indicators
        close = self.data.Close[-1]
        volatility = self.volatility[-1]
        volatility_ma = self.volatility_ma[-1]
        volume_ratio = self.volume_ratio[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        atr = self.atr[-1]

        # Debug output
        print(f"\n=== Signal Debug ===")
        print(f"Close: {close:.2f}, Volatility: {volatility:.4f} (MA: {volatility_ma:.4f})")
        print(f"BB Upper: {bb_upper:.2f}, BB Lower: {bb_lower:.2f}")
        print(f"Volume Ratio: {volume_ratio:.2f}, ATR: {atr:.2f}")
        print(f"Current position: {self.position}")

        # Entry signals with confidence scoring
        if not self.position:
            conditions = {
                'volatility': self.volatility_ratio[-1] > 1.3 or self.vol_breakout[-1],
                'volume': self.volume_ratio[-1] > self.volume_multiplier,
                'momentum_up': self.momentum[-1] > self.momentum_threshold,
                'momentum_down': self.momentum[-1] < -self.momentum_threshold,
                'upper_breakout': close > self.bb_upper[-1],
                'lower_breakout': close < self.bb_lower[-1],
                'squeeze_breakout': self.bb_squeeze[-1] and self.volatility_ratio[-1] > 1.2,
                'price_pos_high': self.price_position[-1] > 0.7,
                'price_pos_low': self.price_position[-1] < 0.3
            }

            direction = None
            confidence = 0

            if conditions['volatility'] and conditions['volume']:
                if conditions['upper_breakout'] and conditions['momentum_up']:
                    direction = 'buy'
                    confidence = 0.8
                elif conditions['lower_breakout'] and conditions['momentum_down']:
                    direction = 'sell'
                    confidence = 0.8
            
            if not direction and conditions['squeeze_breakout']:
                if conditions['price_pos_high'] and conditions['momentum_up']:
                    direction = 'buy'
                    confidence = 0.6
                elif conditions['price_pos_low'] and conditions['momentum_down']:
                    direction = 'sell'
                    confidence = 0.6

            if direction and confidence >= self.signal_threshold:
                vol_multiplier = min(2.5, max(1.0, self.volatility_ratio[-1]))
                stop_distance = atr * self.stop_loss_atr_mult * vol_multiplier
                profit_distance = atr * self.take_profit_atr_mult * vol_multiplier
                
                if direction == 'buy':
                    stop_loss = close - stop_distance
                    take_profit = close + profit_distance
                    print(f"\n=== Buy Entry ===")
                else:
                    stop_loss = close + stop_distance
                    take_profit = close - profit_distance
                    print(f"\n=== Sell Entry ===")
                
                print(f"Price: {close:.2f}, Stop: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
                print(f"Confidence: {confidence:.2f}, Vol Multiplier: {vol_multiplier:.2f}")

                if direction == 'buy':
                    self.buy(size=position_size)
                else:
                    self.sell(size=position_size)

        # Exit conditions
        if self.position:
            exit_reasons = []
            
            # Mean reversion
            if abs(close - self.bb_middle[-1]) / self.bb_middle[-1] < 0.02:
                exit_reasons.append("Mean reversion detected")
            
            # Volatility collapse
            if self.volatility_ratio[-1] < 0.8:
                exit_reasons.append("Volatility collapse")
            
            # RSI extremes with position normalization
            price_position_normal = 0.3 < self.price_position[-1] < 0.7
            if (self.rsi[-1] > 80 or self.rsi[-1] < 20) and price_position_normal:
                exit_reasons.append(f"RSI extreme ({self.rsi[-1]:.2f}) with position normalization")
            
            if exit_reasons:
                print(f"\n=== Exit Signal ===")
                print(f"Exit reasons: {', '.join(exit_reasons)}")
                self.position.close()

# Run Backtest with increased cash for BTC prices
bt = Backtest(data, VolatileStrategy, cash=1000000, commission=.002)
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
