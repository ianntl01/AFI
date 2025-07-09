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
class BearStrategy(Strategy):
    # Parameters from bear.py config
    rsi_threshold = 40
    rsi_exit_threshold = 50
    min_volume = 500000
    stop_loss_atr_mult = 2.0
    take_profit_atr_mult = 3.0
    adx_strong_trend = 25
    min_confirmations = 4

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Indicators from bear.py
        self.ema_short = self.I(ta.EMA, close, timeperiod=8)
        self.ema_long = self.I(ta.EMA, close, timeperiod=21)
        self.long_term_trend = self.I(ta.SMA, close, timeperiod=100)
        
        # MACD with updated periods
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=8, slowperiod=17, signalperiod=9)
        self.macd = self.I(lambda: macd)
        self.macd_signal = self.I(lambda: macd_signal)
        self.macd_hist = self.I(lambda: macd_hist)
        
        # RSI
        self.rsi = self.I(ta.RSI, close, timeperiod=14)
        
        # ATR for volatility
        self.atr = self.I(ta.ATR, high, low, close, timeperiod=14)
        
        # ADX for trend strength
        self.adx = self.I(ta.ADX, high, low, close, timeperiod=25)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: bb_upper)
        self.bb_lower = self.I(lambda: bb_lower)
        self.bb_width = self.I(lambda: (bb_upper - bb_lower) / bb_middle)
        
        # Volume MA
        self.volume_ma = self.I(ta.SMA, volume, timeperiod=20)
        self.volume_ratio = self.I(lambda: np.where(self.volume_ma > 0, volume / self.volume_ma, 1))

    def next(self):
        # Risk management - 10% of capital per trade
        cash_available = self.equity
        btc_price = self.data.Close[-1]
        position_size = max(1, int((cash_available * 0.1) / btc_price))

        # Get current indicators
        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        long_term_trend = self.long_term_trend[-1]

        # Entry condition checks
        is_long_term_bearish = close < long_term_trend
        rsi_cross_down = prev_rsi >= self.rsi_threshold and rsi < self.rsi_threshold
        strong_trend = self.adx[-1] > self.adx_strong_trend
        macd_bearish = self.macd_hist[-1] < 0
        bb_range = self.bb_upper[-1] - self.bb_lower[-1]
        bb_position = (close - self.bb_lower[-1]) / bb_range if bb_range > 0 else 0.5
        bb_bearish = bb_position > 0.7
        volume_spike = self.volume_ratio[-1] > 1.5 and volume > self.min_volume

        confirmations = [is_long_term_bearish, rsi_cross_down, strong_trend, macd_bearish, bb_bearish, volume_spike]
        confidence = sum(confirmations) / len(confirmations)

        # Debug output
        print(f"\n=== Signal Debug ===")
        print(f"Long-term bearish: {is_long_term_bearish}")
        print(f"RSI cross down: {rsi_cross_down} ({prev_rsi:.2f} -> {rsi:.2f})")
        print(f"ADX strong trend: {strong_trend} ({self.adx[-1]:.2f})")
        print(f"MACD bearish: {macd_bearish} ({self.macd_hist[-1]:.2f})")
        print(f"BB bearish: {bb_bearish} ({bb_position:.2f})")
        print(f"Volume spike: {volume_spike} ({volume:.2f} vs MA {volume_ma:.2f})")
        print(f"Confidence: {confidence:.2f}")
        print(f"Current position: {self.position}")

        # Entry signal - short position
        if confidence * len(confirmations) >= self.min_confirmations and not self.position:
            # Calculate dynamic stop levels using ATR and ADX
            atr = self.atr[-1]
            adx = self.adx[-1]
            trend_strength = min(1.5, max(0.8, adx / self.adx_strong_trend))
            stop_loss = close + (self.stop_loss_atr_mult * trend_strength * atr)
            take_profit = close - (self.take_profit_atr_mult * trend_strength * atr)
            
            print(f"\n=== Short Entry ===")
            print(f"Price: {close:.2f}, Stop: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            self.sell(size=position_size)

        # Exit conditions
        exit_triggers = []
        if self.position and prev_rsi <= self.rsi_exit_threshold and rsi > self.rsi_exit_threshold:
            exit_triggers.append("RSI exit threshold reached")

        if self.position and self.adx[-1] < 20:
            exit_triggers.append("Trend weakening (ADX < 20)")

        macd_crossover = (self.macd[-2] < self.macd_signal[-2] and 
                         self.macd[-1] > self.macd_signal[-1])
        if self.position and macd_crossover:
            exit_triggers.append("MACD bullish crossover")
            
        if exit_triggers:
            print(f"\n=== Exit Signal ===")
            print(f"Exit reasons: {', '.join(exit_triggers)}")
            self.position.close()

# Run Backtest with increased cash for BTC prices
bt = Backtest(data, BearStrategy, cash=1000000, commission=.002)
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
