import time
import logging
from datetime import datetime
from typing import Dict, Optional
from AFI.strategies.bull import BullStrategy
from AFI.strategies.bear import BearStrategy
from AFI.strategies.ranging import RangingStrategy
from AFI.strategies.volatile import VolatileStrategy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class StrategyOrchestrator:
    def __init__(self, market_analyzer, strategies, order_executor, decision_logger):
        """
        Initialize the strategy orchestrator
        
        Args:
            market_analyzer: MarketAnalysisSystem instance
            strategies: Dict of strategy instances
            order_executor: SmartOrderExecutor instance
            decision_logger: Logger for trade decisions
        """
        self.market_analyzer = market_analyzer
        self.strategies = strategies
        self.order_executor = order_executor
        self.decision_logger = decision_logger

        # Provide strategies with access to the market analyzer
        for strategy in self.strategies.values():
            if hasattr(strategy, 'set_market_analyzer'):
                strategy.set_market_analyzer(self.market_analyzer)
                logger.info(f"Market analyzer set for {strategy.__class__.__name__}")
        
        # Connect market analyzer to risk manager
        if hasattr(order_executor, 'risk_manager'):
            order_executor.risk_manager.market_analyzer = market_analyzer
            
        self.active_strategy = None
        self.active_tokens = set()  # Track active positions
        self.update_interval = 300  # 5 minutes

    def activate_strategy(self):
        """Activate the appropriate strategy based on the current market regime"""
        try:
            # Get regime groups and analysis from market analyzer
            regime_groups, full_analysis = self.market_analyzer.get_regime_groups()
            
            # Find the regime with the most tokens
            active_regime = max(regime_groups.items(), key=lambda x: len(x[1]), default=(None, []))
            regime = active_regime[0]
            
            if not regime:
                logger.warning("No valid market regime detected")
                return
            
            logger.info(f"Current Market Regime: {regime}")

            # Get approved tokens with their confidence scores
            approved_tokens = {}
            for token_data in active_regime[1]:
                symbol = token_data['symbol']
                confidence = token_data['confidence']
                if confidence >= self.market_analyzer.strategy_manager.min_confidence:
                    approved_tokens[symbol] = confidence
            
            logger.info(f"Approved Tokens: {approved_tokens}")

            # Map regime names to strategy keys
            regime_to_strategy = {
                'Bullish': 'bull',
                'Bearish': 'bear',
                'Volatile': 'volatile',
                'Ranging': 'ranging'
            }

            strategy_key = regime_to_strategy.get(regime)
            if not strategy_key or strategy_key not in self.strategies:
                logger.error(f"No strategy found for regime: {regime}")
                return

            # Activate the corresponding strategy
            self.active_strategy = self.strategies[strategy_key]
            
            # Set approved tokens for the active strategy
            self.active_strategy.set_tokens(approved_tokens)
            logger.info(f"Activated Strategy: {self.active_strategy.__class__.__name__}")

        except Exception as e:
            logger.error(f"Strategy activation error: {e}")
            raise

    def monitor_and_execute(self):
        """Monitor and execute trades based on the active strategy and market analysis."""
        if not self.active_strategy:
            logger.warning("No active strategy, skipping execution cycle.")
            return

        try:
            _, market_analysis = self.market_analyzer.get_regime_groups()
            tokens_to_monitor = list(self.active_strategy.tokens.keys()) if hasattr(self.active_strategy, 'tokens') else []

            if not tokens_to_monitor:
                logger.info("No approved tokens for the active strategy.")
                return

            for token in tokens_to_monitor:
                analysis = market_analysis.get(token)
                if not self._validate_analysis_data(analysis):
                    logger.warning(f"Incomplete analysis data for {token}, skipping.")
                    continue

                position = self.order_executor.get_position(token)

                # Check for exit signals first if a position is open
                if position and position.get('side'):
                    exit_signal = self.active_strategy.generate_exit_signal(token, analysis, position)
                    if exit_signal:
                        logger.info(f"Exit signal generated for {token}: {exit_signal.get('reason', 'No reason specified')}")
                        result = self.order_executor.exit_trade(token, exit_signal)
                        if result:
                            self.active_tokens.discard(token)
                            self._log_trade(token, 'EXIT', result, analysis.get('confidence', 0))
                        continue # Skip entry logic if an exit was processed

                # If no position is open, check for entry signals
                if not position or not position.get('side'):
                    entry_signal = self.active_strategy.generate_entry_signal(token, analysis)
                    if entry_signal:
                        # Add stop levels if the strategy supports them
                        if hasattr(self.active_strategy, 'get_stop_levels'):
                            direction = entry_signal.get('direction')
                            stop_loss, take_profit = self.active_strategy.get_stop_levels(token, analysis, direction)
                            entry_signal['stop_loss'] = stop_loss
                            entry_signal['take_profit'] = take_profit
                        
                        logger.info(f"Entry signal generated for {token}: {entry_signal.get('direction')}")
                        result = self.order_executor.execute_trade(token, entry_signal)
                        if result:
                            self.active_tokens.add(token)
                            self._log_trade(token, 'ENTRY', result, analysis.get('confidence', 0))

        except Exception as e:
            logger.error(f"An error occurred during the monitor_and_execute cycle: {e}", exc_info=True)

    def _validate_analysis_data(self, analysis):
        """Validate that analysis data has all required fields"""
        required_fields = ['regime', 'confidence', 'market_data']
        return all(field in analysis for field in required_fields)

    def _log_trade(self, token: str, action: str, result: Dict, confidence: float):
        """Log trade details"""
        try:
            self.decision_logger.log_trade({
                'timestamp': datetime.now().isoformat(),
                'token': token,
                'action': action,
                'strategy': self.active_strategy.__class__.__name__,
                'order_id': result.get('id'),
                'price': result.get('price'),
                'amount': result.get('amount'),
                'confidence': confidence,
                'status': result.get('status')
            })
        except Exception as e:
            logger.error(f"Trade logging failed: {e}")

    def run(self):
        """Run the orchestrator in a loop"""
        logger.info("Starting Strategy Orchestrator...")
        while True:
            try:
                self.activate_strategy()
                self.monitor_and_execute()
                
                # Log current state
                logger.info(f"Active Strategy: {self.active_strategy.__class__.__name__ if self.active_strategy else 'None'}")
                logger.info(f"Active Tokens: {len(self.active_tokens)}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Orchestrator run error: {e}")
                time.sleep(60)  # Wait before retrying