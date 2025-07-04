import time
import logging
from datetime import datetime
from typing import Dict, Optional
from strategies.bull import BullStrategy
from strategies.bear import BearStrategy
from strategies.ranging import RangingStrategy
from strategies.volatile import VolatileStrategy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
strategies = {
    'bull': BullStrategy(),
    'bear': BearStrategy(),
    'ranging': RangingStrategy(),
    'volatile': VolatileStrategy()
}
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
        """Monitor and execute trades based on market conditions"""
        if not self.active_strategy:
            logger.warning("No active strategy")
            return
            
        try:
            # Get current market analysis
            regime_groups, market_analysis = self.market_analyzer.get_regime_groups()
            
            for token, analysis in market_analysis.items():
                # Skip tokens with errors or incomplete data
                if "error" in analysis or 'confidence' not in analysis or 'market_data' not in analysis:
                    continue

                # Skip if confidence is below minimum
                if analysis['confidence'] < self.market_analyzer.strategy_manager.min_confidence:
                    continue

                # Ensure analysis has all required fields for strategy
                if not self._validate_analysis_data(analysis):
                    logger.warning(f"Incomplete analysis data for {token}")
                    continue

                # Generate signals
                entry_signal = self.active_strategy.generate_entry_signal(token, analysis)
                exit_signal = None
                if token in self.active_tokens:
                    exit_signal = self.active_strategy.generate_exit_signal(token, analysis)

                # Process entry signal
                if entry_signal and token not in self.active_tokens:
                    signal = {
                        'direction': entry_signal,
                        'order_type': 'market',
                        'strategy': self.active_strategy.__class__.__name__,
                        'regime': analysis['regime'],
                        'strategy_regime': self.active_strategy.get_preferred_regime(),
                        'oco_enabled': hasattr(self.active_strategy, 'get_stop_levels'),
                        'confidence': analysis['confidence'],
                        'market_data': analysis['market_data']
                    }

                    # Add stop levels if supported
                    if signal['oco_enabled']:
                        stop_loss, take_profit = self.active_strategy.get_stop_levels(token, analysis)
                        signal.update({
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })

                    result = self.order_executor.execute_trade(token, signal)
                    if result:
                        self.active_tokens.add(token)
                        self._log_trade(token, 'ENTRY', result, analysis['confidence'])

                # Process exit signal
                if exit_signal and token in self.active_tokens:
                    exit_signal = {
                        'direction': 'sell' if entry_signal == 'buy' else 'buy',
                        'order_type': 'market',
                        'strategy': self.active_strategy.__class__.__name__,
                        'regime': analysis['regime'],
                        'confidence': analysis['confidence'],
                        'market_data': analysis['market_data']
                    }
                    
                    result = self.order_executor.exit_trade(token, exit_signal)
                    if result:
                        self.active_tokens.remove(token)
                        self._log_trade(token, 'EXIT', result, analysis['confidence'])

        except Exception as e:
            logger.error(f"Trade execution error: {e}")

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