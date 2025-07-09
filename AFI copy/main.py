# main.py
import logging
import signal
import sys
import time
import yaml
import asyncio
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import the FastAPI app
from api.server import app, system_manager

# Import trading components
from execution.order_executor import SmartOrderExecutor
from core.market_analyzer import MarketAnalysisSystem, RiskManager
from agents.orchestrator import StrategyOrchestrator
from strategies.bull import BullStrategy
from strategies.bear import BearStrategy
from strategies.ranging import RangingStrategy
from strategies.volatile import VolatileStrategy

class TradingSystem:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the trading system"""
        # Get the directory where the script is located
        script_dir = Path(__file__).parent
        config_file = script_dir / config_path
        
        # Check if config file exists in the script directory
        if not config_file.exists():
            # Check if config file exists in the parent directory
            config_file = script_dir.parent / config_path

        # Check if config file exists in the project root directory
        if not config_file.exists():
            config_file = Path(config_path).absolute()

        self.config = self._load_config(str(config_file))
        self.running = False
        self.components = {}
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            sys.exit(f"Failed to load configuration: {str(e)}")

    def _setup_logging(self):
        """Configure logging system"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Main logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/trading_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingSystem')
        
        # Decision logger
        decision_logger = logging.getLogger('decision_logger')
        decision_logger.setLevel(logging.INFO)
        decision_handler = logging.FileHandler('logs/trading_decisions.log')
        decision_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        decision_logger.addHandler(decision_handler)
        
        self.logger.info("Logging system initialized")

    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize market analysis system
            market_analyzer = MarketAnalysisSystem(
                exchange=self.config['exchange']['name'],
                testnet=self.config['exchange']['testnet'],
                timeframe=self.config['trading']['timeframe']
            )
            self.components['market_analyzer'] = market_analyzer
            
            # Create risk manager instance
            risk_manager = RiskManager(
                exchange=self.config['exchange']['name'],
                max_position_size=self.config['execution']['position_size']
            )
            risk_manager.market_analyzer = market_analyzer  # Link the market analyzer
            
            # Initialize order executor
            order_executor = SmartOrderExecutor(
                exchange=market_analyzer.exchange,
                risk_manager=risk_manager,
                paper_trading=self.config['trading']['paper_trading']
            )
            self.components['order_executor'] = order_executor
            
            # Initialize strategies
            strategies = {
                'bull': BullStrategy(self.config['strategies']['bull']),
                'bear': BearStrategy(self.config['strategies']['bear']),
                'ranging': RangingStrategy(self.config['strategies']['ranging']),
                'volatile': VolatileStrategy(self.config['strategies']['volatile'])
            }
            
            # Initialize orchestrator
            orchestrator = StrategyOrchestrator(
                market_analyzer=market_analyzer,
                strategies=strategies,
                order_executor=order_executor,
                decision_logger=logging.getLogger('decision_logger')
            )
            self.components['orchestrator'] = orchestrator
            
            self.logger.info("System components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            return False

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown()

    def check_system_health(self) -> bool:
        """Check health of all system components"""
        try:
            # Check exchange connection
            self.components['market_analyzer'].exchange.fetch_time()
            
            # Check strategy orchestrator
            if not self.components['orchestrator'].active_strategy:
                self.logger.warning("No active strategy")
            
            # Check order executor
            executor = self.components['order_executor']
            metrics = executor.get_performance_report()
            
            self.logger.info(f"System Health - Active Positions: {metrics['active_positions']}, "
                           f"Success Rate: {metrics['success_rate']:.2%}")
            
            return True
        except Exception as e:
            self.logger.error(f"System health check failed: {str(e)}")
            return False

    def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("Initiating system shutdown...")
        self.running = False
        
        try:
            # Close all positions if configured
            if self.config['trading'].get('close_positions_on_shutdown', True):
                executor = self.components['order_executor']
                for token in executor.positions:
                    executor.exit_trade(token, {'exit_type': 'market'}, force=True)
            
            # Cleanup components
            for name, component in self.components.items():
                self.logger.info(f"Shutting down {name}...")
                if hasattr(component, 'cleanup'):
                    component.cleanup()
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
        finally:
            sys.exit(0)

    async def run_trading_loop(self):
        """Async trading system execution loop"""
        self.running = True
        self._setup_signal_handlers()
        
        self.logger.info("Starting trading system...")
        while self.running:
            try:
                cycle_start = time.time()
                
                # Check system health
                if not self.check_system_health():
                    self.logger.error("System health check failed. Pausing execution...")
                    await asyncio.sleep(60)
                    continue
                
                # Run orchestrator cycle
                self.components['orchestrator'].run()
                
                # Calculate and enforce cycle interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(self.config['trading']['cycle_interval'] - cycle_time, 1)
                
                self.logger.info(f"Cycle completed in {cycle_time:.2f}s. "
                               f"Sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"System error: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

async def run_server():
    """Run the FastAPI server"""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Entry point for the trading system and API server"""
    try:
        # Initialize trading system
        trading_system = TradingSystem('config.yaml')
        
        if trading_system.initialize_components():
            # Share the trading system with the API server
            system_manager.system = trading_system
            system_manager.state = "running"
            
            # Only run the trading task - server should be run separately
            trading_task = asyncio.create_task(trading_system.run_trading_loop())
            await trading_task
            
        else:
            sys.exit("Failed to initialize trading system")
            
    except Exception as e:
        logging.critical(f"Fatal system error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())