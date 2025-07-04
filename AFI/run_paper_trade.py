from main import TradingSystem
import logging
import asyncio

def setup_logging():
    """Configure logging for paper trading"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler('paper_trading.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.debug("Logging system initialized")

async def run_trading_system():
    # Setup logging
    setup_logging()
    logger = logging.getLogger('PaperTrading')
    
    try:
        # Initialize trading system with paper trading config
        logger.info("Initializing paper trading system...")
        system = TradingSystem('config.yaml')
        
        if system.initialize_components():
            logger.info("System initialized successfully. Starting paper trading...")
            await system.run_trading_loop()
        else:
            logger.error("Failed to initialize trading system")
            
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
        system.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_trading_system())
