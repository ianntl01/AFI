# docs/debug_pos.py
import asyncio
import logging
from decimal import Decimal
import sys
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main import TradingSystem

# --- Configuration ---
TEST_TOKEN = 'BTC/USDT' # A high-volume token for reliable data

def setup_debug_logging():
    """Sets up a simple logger for the debug script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    return logging.getLogger("PositionDebugger")

async def run_execution_test(logger):
    """Initializes the system and runs an isolated execution test."""
    logger.info("--- Starting Execution Pipeline Test ---")
    
    # 1. Initialize the Trading System to get access to components
    logger.info("Initializing trading system from 'config.yaml'...")
    system = TradingSystem('config.yaml')
    if not system.initialize_components():
        logger.error("Failed to initialize system components. Aborting test.")
        return

    logger.info("System components initialized successfully.")
    
    # 2. Get required components
    executor = system.components.get('order_executor')
    market_analyzer = system.components.get('market_analyzer')

    if not executor or not market_analyzer:
        logger.error("Could not retrieve executor or market_analyzer from system.")
        return

    # 3. Fetch live market analysis to build a valid signal
    logger.info(f"Fetching market analysis for {TEST_TOKEN}...")
    analysis_results = market_analyzer.analyze_market()
    token_analysis = analysis_results.get(TEST_TOKEN)

    if not token_analysis or 'market_data' not in token_analysis:
        logger.error(f"Could not get valid market analysis for {TEST_TOKEN}. Aborting.")
        return

    logger.info(f"Successfully fetched analysis. Current regime: {token_analysis.get('regime')}")

    # 4. Construct a valid signal dictionary (THE CRITICAL STEP)
    # This mimics what a correctly implemented strategy should provide.
    trade_signal = {
        'direction': 'buy',
        'order_type': 'market',
        'strategy': 'debug_test',
        'regime': token_analysis['regime'],
        'strategy_regime': token_analysis['regime'],  # Ensure regimes match for validation
        'market_data': token_analysis['market_data'] # The missing piece
    }
    logger.info("Constructed valid trade signal.")

    # 5. Execute the trade
    try:
        logger.info(f"--- Attempting to OPEN position for {TEST_TOKEN}... ---")
        trade_result = executor.execute_trade(TEST_TOKEN, trade_signal)
        
        if trade_result and trade_result.get('success'):
            logger.info(f"SUCCESS: Opened position. Details: {trade_result.get('trade')}")
            
            # 6. Wait and then close the position
            logger.info("Waiting 10 seconds before closing...")
            await asyncio.sleep(10)
            
            logger.info(f"--- Attempting to CLOSE position for {TEST_TOKEN}... ---")
            exit_signal = {'exit_type': 'market'}
            exit_result = executor.exit_trade(TEST_TOKEN, exit_signal, force=True)
            
            if exit_result:
                logger.info(f"SUCCESS: Closed position. Details: {exit_result}")
            else:
                logger.error("FAILURE: Failed to close position.")
        else:
            logger.error(f"FAILURE: Could not open position. Reason: {trade_result}")

    except Exception as e:
        logger.error(f"An exception occurred during trade execution: {e}", exc_info=True)

    logger.info("--- Execution Pipeline Test Finished ---")

if __name__ == "__main__":
    logger = setup_debug_logging()
    asyncio.run(run_execution_test(logger))
