# execution/order_executor.py
import ccxt
import time
import logging
import random
from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderExecutionError(Exception):
    """Custom exception for execution failures"""
    pass

class SimulationMode:
    """Pure simulation mode - no exchange connection needed"""
    
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        
    def execute_order(self, symbol, side, amount, order_type='market', price=None):
        """Simulate order execution"""
        # Get current price (you can use a price feed or mock data)
        current_price = self.get_mock_price(symbol)
        
        trade = {
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': current_price,
            'timestamp': time.time(),
            'id': f"sim_{len(self.trade_history)}"
        }
        
        # Update simulated balance and positions
        if side == 'buy':
            cost = amount * current_price
            if cost <= self.balance:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + amount
                self.trade_history.append(trade)
                return trade
            else:
                raise ccxt.InsufficientFunds(f"Simulation failed for {side} {amount} {symbol}: Insufficient balance")
        else:  # sell
            if symbol in self.positions and self.positions[symbol] >= amount:
                self.balance += amount * current_price
                self.positions[symbol] -= amount
                self.trade_history.append(trade)
                return trade
            else:
                raise ccxt.InsufficientFunds(f"Simulation failed for {side} {amount} {symbol}: Insufficient position")
    
    def get_mock_price(self, symbol):
        """Return mock price - replace with real price feed for better simulation"""
        base_prices = {
            'BTC/USDT': 65000,
            'ETH/USDT': 3500,
            'AVAX/USDT': 35,
            'SUI/USDT': 2.5
        }
        base = base_prices.get(symbol, 100)
        # Add some random variation
        return base * (1 + random.uniform(-0.02, 0.02))

class SmartOrderExecutor:
    def __init__(self, 
                 exchange: ccxt.Exchange,
                 risk_manager,
                 paper_trading: bool = False):
        """
        Hybrid order executor with risk-aware execution and strategy adaptation
        
        :param exchange: CCXT exchange instance
        :param risk_manager: RiskManager instance from market analyzer
        :param paper_trading: Enable simulated trading mode
        """
        self.market_analyzer = risk_manager.market_analyzer if hasattr(risk_manager, 'market_analyzer') else None
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading
        self.open_orders = {}
        self.positions = {}
        self.strategy_params = {}
        self.performance_metrics = {
            'total_trades': 0,
            'total_slippage': 0.0,
            'success_rate': 0.0
        }

        # Initialize simulation mode if paper trading is enabled
        if paper_trading:
            self.simulation = SimulationMode(initial_balance=10000)
        
        # Execution parameters
        self._base_slippage = 0.001  # 0.1%
        self._max_retries = 3
        self._order_timeout = 30  # seconds

    def set_strategy_parameters(self, strategy: str, params: Dict):
        """Update strategy-specific execution parameters"""
        self.strategy_params[strategy] = params
        logger.info(f"Updated execution params for {strategy}: {params}")

    def calculate_position_size(self, 
                               token: str,
                               signal: Dict) -> Optional[Decimal]:
        """Risk-adjusted position sizing with strategy multipliers"""
        try:
            # Ensure signal has required fields
            if not isinstance(signal, dict) or 'regime' not in signal or 'market_data' not in signal:
                logger.warning(f"Incomplete signal data for {token}")
                return None
            
            # Add confidence if missing (for backward compatibility)
            if 'confidence' not in signal:
                signal['confidence'] = 0.7  # Default confidence
            
            # Get base risk parameters
            risk_data = self.risk_manager.analyze_risk({token: signal})[0]
            if risk_data['recommendation'].startswith('REJECT'):
                logger.warning(f"Trade rejected by risk manager: {token}")
                return None

            # Get portfolio equity
            if self.paper_trading:
                equity = Decimal(self.simulation.balance)
            else:
                balance = self.exchange.fetch_balance()
                equity = Decimal(balance['USDT']['free'])

            # Apply strategy multiplier
            strategy = signal.get('strategy', 'default')
            params = self.strategy_params.get(strategy, {})
            size_multiplier = Decimal(params.get('size_multiplier', 1.0))
            
            # Calculate base size
            max_size = equity * Decimal(self.risk_manager.max_position_size)
            max_size *= size_multiplier

            # Reduce size based on risk score
            risk_score = Decimal(risk_data['risk_score'])
            size = max_size * (Decimal(1) - risk_score)

            # Get current price
            ticker = self.exchange.fetch_ticker(token)
            price = Decimal(ticker['last'])

            return size / price  # Return in base currency
        except Exception as e:
            logger.error(f"Position calc failed: {str(e)}")
            return None

    def execute_order(self, 
                     token: str,
                     order_type: str,
                     side: str,
                     amount: Decimal,
                     **kwargs) -> Optional[Dict]:
        """Core order execution with retry logic"""
        if self.paper_trading:
            try:
                return self.simulation.execute_order(
                    symbol=token,
                    side=side,
                    amount=float(amount),
                    order_type=order_type,
                    price=None
                )
            except ccxt.InsufficientFunds as e:
                logger.error(f"Paper trading failed: {e}")
                raise OrderExecutionError(f"Failed to execute paper trade for {token}") from e

        for attempt in range(self._max_retries):
            try:
                # Calculate dynamic slippage
                volatility = self.risk_manager._check_volatility(token)
                slippage = max(self._base_slippage, volatility * 0.5)
                
                # Prepare order parameters
                params = {
                    'slippage': slippage,
                    **self.strategy_params.get(kwargs.get('strategy', 'default'), {}),
                    **kwargs
                }

                # Convert Decimal to float for CCXT
                amount_float = float(amount.quantize(Decimal('1e-8')))
                
                order = self.exchange.create_order(
                    symbol=token,
                    type=order_type,
                    side=side,
                    amount=amount_float,
                    params=params
                )

                # Store order metadata
                order_meta = {
                    **order,
                    'strategy': kwargs.get('strategy'),
                    'execution_time': datetime.now().isoformat(),
                    'slippage': slippage
                }
                self.open_orders[order['id']] = order_meta
                self._update_performance_metrics(order_meta)
                
                return order_meta
            except ccxt.InsufficientFunds as e:
                logger.error(f"Insufficient funds: {str(e)}")
                break
            except ccxt.NetworkError as e:
                logger.warning(f"Network error (attempt {attempt+1}): {str(e)}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                break

        raise OrderExecutionError(f"Failed to execute {side} order for {token}")

    def execute_trade(self, 
                 token: str,
                 signal: Dict) -> Optional[Dict]:
        """Execute a full trade workflow, including risk checks, position sizing, and order placement."""
        try:
            # 1. Validate Signal Format
            if not isinstance(signal, dict) or 'direction' not in signal:
                logger.error(f"Invalid or incomplete signal for {token}: {signal}")
                return None

            # 2. Set Default Values (preserves the rich signal object)
            signal.setdefault('order_type', 'market')
            signal.setdefault('strategy', 'default')
            signal.setdefault('oco_enabled', False)

            # 3. Calculate Position Size (Risk-Adjusted)
            size = self.calculate_position_size(token, signal)
            if not size:
                return None

            # Execute main order
            order = self.execute_order(
                token=token,
                order_type=signal['order_type'],
                side=signal['direction'],
                amount=size,
                strategy=signal['strategy']
            )

            # Place OCO orders if required
            if signal['oco_enabled'] and signal.get('stop_loss') and signal.get('take_profit'):
                self.place_oco_orders(token, size, signal)

            # Update position tracking
            self.positions[token] = {
                'entry_price': order['price'],
                'size': size,
                'entry_time': datetime.now().isoformat(),
                'strategy': signal['strategy'],
                'direction': signal['direction'],
                'risk_score': self.risk_manager.analyze_risk({token: signal})[0]['risk_score']
            }

            return order

        except OrderExecutionError as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return None

    def exit_trade(self, 
                  token: str,
                  signal: Dict,
                  force: bool = False) -> Optional[Dict]:
        """Exit position with smart order routing"""
        position = self.get_position(token)
        if not position:
            logger.warning(f"No position found for {token}")
            return None

        try:
            # Determine exit type
            exit_type = 'market' if force else signal.get('exit_type', 'market')
            
            # Execute exit order
            order = self.execute_order(
                token=token,
                order_type=exit_type,
                side='sell' if position['direction'] == 'buy' else 'buy',
                amount=position['size'],
                strategy=position.get('strategy')
            )

            # Clean up position
            del self.positions[token]
            self._cancel_associated_orders(token)

            return order
        except OrderExecutionError as e:
            logger.error(f"Exit failed: {str(e)}")
            return None

    def place_oco_orders(self, 
                        token: str,
                        amount: Decimal,
                        signal: Dict):
        """Place One-Cancels-Other orders with volatility adjustment"""
        try:
            # Calculate dynamic prices
            stop_loss = Decimal(signal['stop_loss'])
            take_profit = Decimal(signal['take_profit'])
            volatility = self.risk_manager._check_volatility(token)
            
            # Adjust prices with volatility-based buffers
            sl_price = stop_loss * (Decimal(1) - Decimal(volatility * 0.1))
            tp_price = take_profit * (Decimal(1) + Decimal(volatility * 0.1))

            # Place OCO order
            order = self.exchange.create_order(
                symbol=token,
                type='OCO',
                side='sell' if signal['direction'] == 'buy' else 'buy',
                amount=float(amount),
                price=float(tp_price),
                stopPrice=float(sl_price),
                params={
                    'stopLimitPrice': float(sl_price * Decimal(0.99)),
                    'stopLimitTimeInForce': 'GTC'
                }
            )

            # Track OCO orders
            self.open_orders[order['orderListId']] = {
                **order,
                'type': 'OCO',
                'token': token
            }
        except Exception as e:
            logger.error(f"OCO order failed: {str(e)}")

    def update_orders(self):
        """Periodic order status update and cleanup"""
        for order_id in list(self.open_orders.keys()):
            try:
                order = self.exchange.fetch_order(order_id)
                if order['status'] in ['closed', 'canceled']:
                    del self.open_orders[order_id]
            except Exception as e:
                logger.error(f"Order update failed: {str(e)}")

    def get_position(self, token: str) -> Optional[Dict]:
        """Retrieve the current position for a given token."""
        return self.positions.get(token)

    def get_open_orders(self, token: str = None) -> Dict:
        """Retrieve all open orders, optionally filtered by token."""
        if token:
            return {oid: o for oid, o in self.open_orders.items() if o.get('symbol') == token}
        return self.open_orders

    def _cancel_associated_orders(self, token: str):
        """Cancel all orders for a given token"""
        for order_id, order in list(self.open_orders.items()):
            if order.get('symbol') == token:
                try:
                    self.exchange.cancel_order(order_id)
                    del self.open_orders[order_id]
                except Exception as e:
                    logger.error(f"Order cancellation failed: {str(e)}")

    def _update_performance_metrics(self, order: Dict):
        """Track execution quality metrics"""
        try:
            expected_price = Decimal(order.get('price', 0))
            actual_price = Decimal(self.exchange.fetch_ticker(order['symbol'])['last'])
            
            slippage = abs(actual_price - expected_price) / expected_price
            self.performance_metrics['total_slippage'] += float(slippage)
            self.performance_metrics['total_trades'] += 1
            
            success_rate = self.performance_metrics['success_rate']
            new_success = 1 if order['filled'] > 0 else 0
            self.performance_metrics['success_rate'] = (
                (success_rate * (self.performance_metrics['total_trades'] - 1) + new_success) 
                / self.performance_metrics['total_trades']
            )
        except Exception as e:
            logger.error(f"Metric update failed: {str(e)}")

    def get_performance_report(self) -> Dict:
        """Get current execution performance metrics"""
        return {
            **self.performance_metrics,
            'open_orders': len(self.open_orders),
            'active_positions': len(self.positions),
            'avg_slippage': (
                self.performance_metrics['total_slippage'] / 
                max(1, self.performance_metrics['total_trades'])
            )
        }






