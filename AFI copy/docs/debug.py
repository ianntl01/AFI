#!/usr/bin/env python3
"""
Updated debug script aligned with new system architecture
"""

import yaml
import ccxt
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Add the project root to the path to ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.market_analyzer import MarketAnalysisSystem
from execution.order_executor import SmartOrderExecutor, SimulationMode
from agents.orchestrator import StrategyOrchestrator
from strategies.ranging import RangingStrategy
from strategies.bull import BullStrategy
from strategies.bear import BearStrategy
from strategies.volatile import VolatileStrategy


class DebugLogger:
    """Simple logger for debug output"""
    def log_trade(self, trade_data):
        print(f"\n=== TRADE EXECUTED ===\n{trade_data}")

def load_config():
    """Load configuration"""
    config_path = Path('AFI/config.yaml')
    if not config_path.exists():
        config_path = Path('../AFI/config.yaml')
    
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
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def initialize_system():
    """Initialize trading system components"""
    config = load_config()
    
    # Market analyzer
    market_analyzer = MarketAnalysisSystem(
        exchange='binance',
        testnet=True,
        timeframe='1h'
    )
    
    # Order executor (paper trading mode)
    order_executor = SmartOrderExecutor(
        exchange=None,  # Will use simulation mode
        risk_manager=market_analyzer,
        paper_trading=True
    )
    
    # Strategies
    strategies = {
        'bull': BullStrategy(config['strategies']['bull']),
        'bear': BearStrategy(config['strategies']['bear']),
        'ranging': RangingStrategy(config['strategies']['ranging']), 
        'volatile': VolatileStrategy(config['strategies']['volatile'])
    }
    
    # Orchestrator
    orchestrator = StrategyOrchestrator(
        market_analyzer=market_analyzer,
        strategies=strategies,
        order_executor=order_executor,
        decision_logger=DebugLogger()
    )
    
    return market_analyzer, order_executor, orchestrator

def debug_strategy_execution(market_analyzer, order_executor, orchestrator):
    """Test strategy execution with new architecture"""
    print("\n=== STRATEGY EXECUTION DEBUG ===")
    
    # Run market analysis
    analysis = market_analyzer.analyze_market()
    print("\nMarket Analysis Results:")
    for symbol, data in analysis.items():
        if 'error' not in data:
            print(f"{symbol}: {data['regime']} (Confidence: {data['confidence']:.2f})")
    
    # Activate appropriate strategy
    orchestrator.activate_strategy()
    if not orchestrator.active_strategy:
        print("❌ No strategy activated")
        return
    
    print(f"\nActive Strategy: {orchestrator.active_strategy.__class__.__name__}")
    
    # Test execution
    orchestrator.monitor_and_execute()
    
    # Show results
    print("\n=== EXECUTION RESULTS ===")
    print(f"Open Positions: {len(order_executor.positions)}")
    print(f"Performance: {order_executor.get_performance_report()}")

def main():
    """Main debug function"""
    print("🔍 DEBUGGING AFI TRADING SYSTEM (NEW ARCHITECTURE)")
    print("=" * 50)
    
    # Step 1: Check testnet connection
    if not debug_testnet_connection():
        print("\n❌ Fix testnet connection first!")
        return
    
    # Step 2: Initialize system components
    market_analyzer, order_executor, orchestrator = initialize_system()
    
    # Step 3: Test strategy execution
    debug_strategy_execution(market_analyzer, order_executor, orchestrator)
    
    print("\n" + "=" * 50)
    print("🔍 DEBUG COMPLETE")

if __name__ == "__main__":
    main()
