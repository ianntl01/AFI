#!/usr/bin/env python3
"""
Debug script to diagnose why no positions are being created
"""

import yaml
import ccxt
import sys
import os
from pathlib import Path

# Add the current directory to path to ensure local imports work
sys.path.append(str(Path(__file__).parent))

from core.market_analyzer import MarketAnalysisSystem
from strategies.ranging import RangingStrategy

def load_config():
    """Load configuration"""
    config_path = Path('config.yaml')
    if not config_path.exists():
        config_path = Path('../config.yaml')
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def debug_testnet_connection():
    """Debug testnet connection and balance"""
    config = load_config()
    
    print("=== TESTNET CONNECTION DEBUG ===")
    
    # Create exchange connection
    exchange_config = {
        'apiKey': config['exchange']['testnet_api']['key'],
        'secret': config['exchange']['testnet_api']['secret'],
        'sandbox': True,  # Enable testnet
        'enableRateLimit': True,
    }
    
    exchange = ccxt.binance(exchange_config)
    exchange.set_sandbox_mode(True)

    
    try:
        # Check connection
        server_time = exchange.fetch_time()
        print(f"✅ Connected to testnet. Server time: {server_time}")
        
        # Check balance
        balance = exchange.fetch_balance()
        print(f"\n=== TESTNET BALANCE ===")
        for asset, amounts in balance.items():
            if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                print(f"{asset}: {amounts['total']}")
        
        # Check if we have USDT
        usdt_balance = balance.get('USDT', {}).get('total', 0)
        print(f"\nUSDT Balance: {usdt_balance}")
        
        if usdt_balance == 0:
            print("❌ No USDT balance! You need USDT to buy other assets.")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def debug_market_analysis():
    """Debug market analysis system"""
    print("\n=== MARKET ANALYSIS DEBUG ===")
    
    config = load_config()
    
    # Initialize market analyzer
    market_analyzer = MarketAnalysisSystem(
        exchange='binance',
        testnet=True,
        timeframe='1h'
    )
    
    # Test market regime detection using the proper method
    analysis = market_analyzer.analyze_market()
    print("\nMarket Analysis Results:")
    for symbol, data in analysis.items():
        if 'error' not in data:
            print(f"{symbol}: {data['regime']} (Confidence: {data['confidence']:.2f})")
    
    # Get approved tokens from analysis results
    approved_tokens = [symbol for symbol, data in analysis.items() 
                      if 'error' not in data and data['confidence'] > 0.4]
    print(f"\nApproved Tokens (confidence > 0.4): {approved_tokens}")
    
    # Test individual token analysis
    for token in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        try:
            # Get current price
            ticker = market_analyzer.exchange.fetch_ticker(token)
            print(f"\n{token}:")
            print(f"  Price: ${ticker['last']:.2f}")
            
            # Get market data from exchange
            ohlcv = market_analyzer.exchange.fetch_ohlcv(token, market_analyzer.timeframe, limit=1)
            if ohlcv:
                print(f"  Open: {ohlcv[0][1]}")
                print(f"  High: {ohlcv[0][2]}")
                print(f"  Low: {ohlcv[0][3]}")
                print(f"  Close: {ohlcv[0][4]}")
                print(f"  Volume: {ohlcv[0][5]}")
            else:
                print(f"  ❌ No OHLCV data available")
                
        except Exception as e:
            print(f"  ❌ Error getting data for {token}: {e}")

def debug_ranging_strategy():
    """Debug ranging strategy signal generation"""
    print("\n=== RANGING STRATEGY DEBUG ===")
    
    config = load_config()
    
    # Initialize strategy
    strategy = RangingStrategy(config['strategies']['ranging'])
    
    # Test signal generation
    test_tokens = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Mock market data (you'd get this from market analyzer)
    mock_market_data = {
        'BTC/USDT': {
            'price': 65000,
            'rsi': 45,  # Neutral RSI
            'volume': 1000000,
            'volatility': 0.02,
            'trend': 0.01,
            'bollinger_position': 0.3
        },
        'ETH/USDT': {
            'price': 3500,
            'rsi': 55,
            'volume': 800000,
            'volatility': 0.025,
            'trend': -0.005,
            'bollinger_position': 0.6
        },
        'SOL/USDT': {
            'price': 150,
            'rsi': 40,
            'volume': 500000,
            'volatility': 0.03,
            'trend': 0.02,
            'bollinger_position': 0.2
        }
    }
    
    for token in test_tokens:
        print(f"\n{token}:")
        market_data = mock_market_data.get(token, {})
        
        # Test signal generation with proper DataFrame input
        try:
            import pandas as pd
            df = pd.DataFrame([{
                'open': market_data['price'] * 0.99,
                'high': market_data['price'] * 1.01,
                'low': market_data['price'] * 0.98,
                'close': market_data['price'],
                'volume': market_data['volume'],
                'rsi': market_data['rsi'],
                'volatility': market_data['volatility'],
                'trend': market_data['trend'],
                'bollinger_position': market_data['bollinger_position']
            }])
            signal = strategy.generate_signal(token, df)
            print(f"  Signal: {signal}")
        except Exception as e:
            print(f"  ❌ Error generating signal: {str(e)}")
        
        # Check strategy conditions
        print(f"  RSI: {market_data.get('rsi', 'N/A')}")
        print(f"  Volume: {market_data.get('volume', 'N/A')}")
        print(f"  Volatility: {market_data.get('volatility', 'N/A')}")
        print(f"  Trend: {market_data.get('trend', 'N/A')}")

def main():
    """Main debug function"""
    print("🔍 DEBUGGING AFI TRADING SYSTEM")
    print("=" * 50)
    
    # Step 1: Check testnet connection
    if not debug_testnet_connection():
        print("\n❌ Fix testnet connection first!")
        return
    
    # Step 2: Debug market analysis
    debug_market_analysis()
    
    # Step 3: Debug strategy
    debug_ranging_strategy()
    
    print("\n" + "=" * 50)
    print("🔍 DEBUG COMPLETE")
    print("\nIf you see errors above, those are likely why no positions are created.")

if __name__ == "__main__":
    main()
