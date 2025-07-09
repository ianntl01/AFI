from AFI.core.exchange_config import ExchangeConfig
import pandas as pd
import numpy as np
import ccxt
import ta
import logging
from typing import Dict, Optional, List, Tuple, Union
from datetime import datetime
import time
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKENS = ["BTC/USDT", "SOL/USDT", "AVAX/USDT", "SUI/USDT", "ETH/USDT", "DOGE/USDT", "BONK/USDT", "PEPE/USDT", "SUI/USDT", "WIF/USDT"]
TIMEZONE = "UTC"
class MarketRegimeAnalyzer:
    def __init__(self, validation_window: int = 24):
        self.validation_window = validation_window
        self.current_regime = None
        self.last_update = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators"""
        try:
            # Basic indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], 20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], 50)
            df['rsi'] = ta.momentum.rsi(df['close'], 14)
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Advanced indicators
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['bollinger_high'] = ta.volatility.bollinger_hband(df['close'])
            df['bollinger_low'] = ta.volatility.bollinger_lband(df['close'])
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

            # Additional indicators for strategy compatibility
            df['ema_9'] = ta.trend.ema_indicator(df['close'], 9)
            df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """Determine the current market regime with enhanced logic"""
        try:
            latest = df.iloc[-1]
            
            # Define regime conditions with weights
            regime_conditions = {
                'Bullish': {
                    'trend': (latest['sma_20'] > latest['sma_50'], 0.3),
                    'rsi': (latest['rsi'] > 60, 0.2),
                    'macd': (latest['macd'] > 0, 0.2),
                    'volume': (latest['volume_ratio'] > 1.2, 0.1),
                    'bollinger': (latest['close'] > latest['bollinger_high'], 0.2)
                },
                'Bearish': {
                    'trend': (latest['sma_20'] < latest['sma_50'], 0.3),
                    'rsi': (latest['rsi'] < 40, 0.2),
                    'macd': (latest['macd'] < 0, 0.2),
                    'volume': (latest['volume_ratio'] > 1.2, 0.1),
                    'bollinger': (latest['close'] < latest['bollinger_low'], 0.2)
                },
                'Ranging': {
                    'rsi_range': (45 <= latest['rsi'] <= 55, 0.4),
                    'volatility': (latest['volatility'] < df['volatility'].mean() * 0.8, 0.3),
                    'bollinger': (latest['bollinger_low'] <= latest['close'] <= latest['bollinger_high'], 0.3)
                },
                'Volatile': {
                    'volatility': (latest['volatility'] > df['volatility'].mean() * 1.5, 0.4),
                    'volume': (latest['volume_ratio'] > 1.5, 0.3),
                    'bollinger': (latest['close'] < latest['bollinger_low'] or latest['close'] > latest['bollinger_high'], 0.3)
                }
            }

            # Calculate weighted scores
            regime_scores = {}
            for regime, conditions in regime_conditions.items():
                score = sum(weight * int(condition) for condition, weight in conditions.values())
                regime_scores[regime] = score

            # Determine best regime
            best_regime = max(regime_scores, key=regime_scores.get)
            total_weight = sum(weight for _, weight in regime_conditions[best_regime].values())
            confidence = regime_scores[best_regime] / total_weight

            # Add market data for strategy use
            result = {
                'regime': best_regime,
                'confidence': confidence,
                'market_data': {
                    'close': latest['close'],
                    'volume': latest['volume'],
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'volatility': latest['volatility'],
                    'sma_20': latest['sma_20'],
                    'sma_50': latest['sma_50']
                }
            }

            self.current_regime = best_regime
            self.last_update = datetime.now()
            
            return result

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            raise

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Main analysis function"""
        try:
            df = self.calculate_indicators(df)
            return self.detect_regime(df)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise

class MarketAnalysisSystem:
    def __init__(self, exchange: str = 'binance', testnet: bool = True, 
                 validation_window: int = 24, timeframe: str = '1h'):
        self.timeframe = timeframe
        self.validation_window = validation_window
        self.exchange_name = exchange
        self.exchange_config = ExchangeConfig()
        self.setup_exchange(testnet)
        self.analyzer = MarketRegimeAnalyzer(validation_window)
        self.strategy_manager = StrategyManager()
        self.last_analysis = {}
        self.update_interval = 300

    def setup_exchange(self, testnet: bool = True):
        """Initialize exchange connection with improved error handling and fallback"""
        retry_settings = self.exchange_config.get_retry_settings()
        max_retries = retry_settings['max_retries']
        delay = retry_settings['delay']
        backoff = retry_settings['backoff_factor']
        
        # First try testnet if specified and it hasn't been marked as unavailable
        if testnet and not self.exchange_config.should_use_mainnet_fallback():
            for attempt in range(max_retries):
                try:
                    self._configure_exchange(True)
                    self.exchange.fetch_time()
                    logger.info(f"Successfully connected to {self.exchange_name} testnet")
                    self.exchange_config.update_testnet_status(True)
                    return
                except Exception as e:
                    if "502 Bad Gateway" in str(e):
                        logger.warning("Testnet appears to be under maintenance")
                        self.exchange_config.update_testnet_status(False)
                        break
                    elif attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Testnet connection attempt {attempt + 1} failed: {e}. "
                                     f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Testnet connection failed after {max_retries} attempts")
                        self.exchange_config.update_testnet_status(False)

        # Fallback to mainnet or use it directly if specified
        try:
            if testnet:
                logger.info("Falling back to mainnet due to testnet unavailability")
            self._configure_exchange(False)
            self.exchange.fetch_time()
            logger.info(f"Successfully connected to {self.exchange_name} mainnet")
        except Exception as e:
            error_msg = f"Exchange setup failed on both networks: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _configure_exchange(self, use_testnet: bool):
        """Configure exchange with appropriate settings"""
        params = self.exchange_config.get_exchange_params(use_testnet)
        self.exchange = getattr(ccxt, self.exchange_name)(params)
        
        if use_testnet:
            self.exchange.set_sandbox_mode(True)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate market data quality"""
        try:
            if df is None or df.empty:
                return False
            
            # Check for missing values
            if df.isnull().any().any():
                logger.warning("Dataset contains missing values")
                return False
            
            # Check for zero volumes
            if (df['volume'] == 0).any():
                logger.warning("Dataset contains zero volumes")
                return False
            
            # Check for minimum data points
            if len(df) < self.validation_window:
                logger.warning(f"Insufficient data points. Required: {self.validation_window}, Got: {len(df)}")
                return False
            
            # Check for price continuity
            price_gaps = df['close'].pct_change().abs()
            if price_gaps.max() > 0.5:  # 50% price gap threshold
                logger.warning("Large price gaps detected in the dataset")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

    def fetch_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch market data with improved error handling for public endpoints"""
        max_retries = self.exchange_config.max_retry_count
        retry_delay = self.exchange_config.retry_delay
        
        for attempt in range(max_retries):
            try:
                # Try fetching OHLCV data (public endpoint)
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    self.timeframe, 
                    limit=limit,
                    params={'price': 'spot'}  # Ensure we're getting spot market data
                )
                
                if not ohlcv:
                    raise ValueError("Empty OHLCV data received")

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                if not self.validate_data(df):
                    logger.warning(f"Invalid data for {symbol}")
                    return None

                return df
                
            except Exception as e:
                if "Invalid Api-Key" in str(e):
                    # If we get an API key error, configure exchange for public access only
                    logger.info("Configuring exchange for public access only")
                    self.exchange.apiKey = None
                    self.exchange.secret = None
                    continue
                    
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Data fetch attempt {attempt + 1} failed for {symbol}: {e}. "
                                 f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Data fetch failed for {symbol} after {max_retries} attempts: {e}")
                    return None

    def analyze_market(self) -> Dict:
        """Analyze market regimes for all tokens"""
        current_time = datetime.now()
        results = {}

        try:
            for symbol in TOKENS:
                # Check if we need to update the analysis
                last_analysis = self.last_analysis.get(symbol, {})
                last_update = last_analysis.get('timestamp')
                
                if last_update and (current_time - datetime.fromisoformat(last_update)).seconds < self.update_interval:
                    results[symbol] = last_analysis
                    continue

                df = self.fetch_data(symbol)
                if df is None:
                    results[symbol] = {"error": "Invalid or missing market data"}
                    continue

                analysis_result = self.analyzer.analyze(df)
                analysis_result.update({
                    'symbol': symbol,
                    'timeframe': self.timeframe,
                    'timestamp': current_time.isoformat(),
                    'price': df['close'].iloc[-1],
                    'volume': df['volume'].iloc[-1]
                })
                
                results[symbol] = analysis_result
                self.last_analysis[symbol] = analysis_result

            return results

        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            raise

    def get_regime_groups(self) -> Tuple[Dict, Dict]:
        """Get and validate regime groups"""
        try:
            analysis_results = self.analyze_market()
            regime_groups = self.strategy_manager.group_by_regime(analysis_results)
            return regime_groups, analysis_results
        except Exception as e:
            logger.error(f"Regime grouping error: {e}")
            raise
class RiskManager:
    def __init__(self, exchange: str = 'binance', max_position_size: float = 0.03):
        self.exchange = exchange
        self.max_position_size = max_position_size  # 3% maximum position size
        self.market_analyzer = None  # Will be set by orchestrator
        self.risk_levels = {
            'LOW': {'max_size': 0.03, 'stop_loss': 0.02},
            'MEDIUM': {'max_size': 0.02, 'stop_loss': 0.015},
            'HIGH': {'max_size': 0.01, 'stop_loss': 0.01}
        }
        self.position_limits = {
            'total_positions': 10,
            'per_regime': 5,
            'max_risk_exposure': 0.25  # 25% maximum total risk exposure
        }
        self.current_positions = {}

    def analyze_risk(self, market_analysis: Dict) -> List[Dict]:
        """Process market analysis and generate risk assessments"""
        try:
            risk_report = []
            total_risk = self._calculate_total_risk()

            for symbol, analysis in market_analysis.items():
                if "error" in analysis:
                    continue
                    
                # Check if we have the required data
                if not analysis.get('market_data') or 'confidence' not in analysis:
                    logger.warning(f"Incomplete market data for {symbol}, skipping risk analysis")
                    continue

                # Calculate base risk metrics
                volatility_risk = analysis.get('market_data', {}).get('volatility', 0)
                volume_risk = 1 - (1 / analysis.get('market_data', {}).get('volume_ratio', 1))
                regime_risk = {
                    'Bullish': 0.3,
                    'Bearish': 0.4,
                    'Ranging': 0.2,
                    'Volatile': 0.5
                }.get(analysis['regime'], 0.3)

                # Calculate combined risk score
                risk_score = (
                    volatility_risk * 0.4 +
                    volume_risk * 0.3 +
                    regime_risk * 0.3
                )

                # Determine risk level
                risk_level = self._determine_risk_level(risk_score)
                
                # Generate position size recommendation
                position_size = self._calculate_position_size(
                    risk_score, 
                    analysis['confidence'],
                    total_risk
                )

                # Generate final recommendation
                recommendation = self._generate_recommendation(
                    analysis['confidence'],
                    risk_score,
                    position_size,
                    total_risk
                )

                risk_report.append({
                    'token': symbol,
                    'regime': analysis['regime'],
                    'confidence': analysis['confidence'],
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'position_size': position_size,
                    'recommendation': recommendation,
                    'stop_loss': self.risk_levels[risk_level]['stop_loss'],
                    'market_data': analysis.get('market_data', {})
                })

            return risk_report

        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            raise

    def _calculate_total_risk(self) -> float:
        """Calculate current total risk exposure"""
        return sum(position.get('risk_score', 0) for position in self.current_positions.values())

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on risk score"""
        if risk_score > 0.7:
            return 'HIGH'
        elif risk_score > 0.5:
            return 'MEDIUM'
        return 'LOW'

    def _calculate_position_size(self, risk_score: float, confidence: float, total_risk: float) -> float:
        """Calculate recommended position size"""
        # Base size on risk level
        risk_level = self._determine_risk_level(risk_score)
        base_size = self.risk_levels[risk_level]['max_size']

        # Adjust for confidence
        confidence_factor = min(confidence, 0.9)  # Cap at 90%
        size = base_size * confidence_factor

        # Adjust for total risk exposure
        if total_risk > self.position_limits['max_risk_exposure']:
            size *= (self.position_limits['max_risk_exposure'] / total_risk)

        return round(size, 4)

    def _generate_recommendation(self, confidence: float, risk_score: float, 
                               position_size: float, total_risk: float) -> str:
        """Generate trading recommendation"""
        if confidence < 0.65:
            return "REJECT (low confidence)"
        elif risk_score > 0.8:
            return "REJECT (high risk)"
        elif total_risk > self.position_limits['max_risk_exposure']:
            return "REJECT (risk exposure limit)"
        elif len(self.current_positions) >= self.position_limits['total_positions']:
            return "REJECT (position limit reached)"
        else:
            return f"APPROVE (size: {position_size:.1%})"

    def update_position(self, token: str, position_data: Dict):
        """Update position tracking"""
        self.current_positions[token] = position_data

    def remove_position(self, token: str):
        """Remove position from tracking"""
        self.current_positions.pop(token, None)

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'total_positions': len(self.current_positions),
            'total_risk': self._calculate_total_risk(),
            'current_exposure': sum(p.get('position_size', 0) for p in self.current_positions.values()),
            'positions_by_regime': self._count_positions_by_regime()
        }

    def _count_positions_by_regime(self) -> Dict:
        """Count positions by regime"""
        regime_counts = {}
        for position in self.current_positions.values():
            regime = position.get('regime', 'unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return regime_counts

    def validate_trade(self, token: str, signal: Dict) -> bool:
        """Validate if a trade meets risk management criteria"""
        try:
            # Get current risk metrics
            metrics = self.get_risk_metrics()
            
            # Check position limits
            if metrics['total_positions'] >= self.position_limits['total_positions']:
                logger.warning(f"Position limit reached: {metrics['total_positions']}")
                return False

            # Check regime position limits
            regime = signal.get('regime', 'unknown')
            regime_count = metrics['positions_by_regime'].get(regime, 0)
            if regime_count >= self.position_limits['per_regime']:
                logger.warning(f"Regime position limit reached for {regime}")
                return False

            # Check risk exposure
            if metrics['total_risk'] > self.position_limits['max_risk_exposure']:
                logger.warning(f"Risk exposure limit reached: {metrics['total_risk']:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False
class StrategyManager:
    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.strategy_details = {
            'Bullish': {
                'description': 'Trend following long positions',
                'color': '\033[92m',
                'preferred_indicators': ['sma_20', 'sma_50', 'rsi', 'macd']
            },
            'Bearish': {
                'description': 'Short selling opportunities',
                'color': '\033[91m',
                'preferred_indicators': ['sma_20', 'sma_50', 'rsi', 'macd']
            },
            'Ranging': {
                'description': 'Mean reversion strategy',
                'color': '\033[93m',
                'preferred_indicators': ['bollinger_high', 'bollinger_low', 'rsi']
            },
            'Volatile': {
                'description': 'Breakout trading',
                'color': '\033[96m',
                'preferred_indicators': ['atr', 'volatility', 'volume_ratio']
            }
        }
        self.performance_metrics = {}

    def group_by_regime(self, results: dict) -> dict:
        """Organize tokens by their detected regime"""
        try:
            groups = {regime: [] for regime in self.strategy_details.keys()}
            
            for symbol, data in results.items():
                if "error" in data or data.get('confidence', 0) < self.min_confidence:
                    continue
                
                regime = data['regime']
                groups[regime].append({
                    'symbol': symbol,
                    'confidence': data['confidence'],
                    'timestamp': data['timestamp'],
                    'price': data.get('price'),
                    'volume': data.get('volume'),
                    'market_data': data.get('market_data', {})
                })
            
            # Sort tokens by confidence within each group
            for regime in groups:
                groups[regime] = sorted(groups[regime], key=lambda x: x['confidence'], reverse=True)
            
            return groups

        except Exception as e:
            logger.error(f"Error grouping by regime: {e}")
            raise

    def get_strategy_requirements(self, regime: str) -> Dict:
        """Get strategy requirements for a specific regime"""
        try:
            if regime not in self.strategy_details:
                raise ValueError(f"Unknown regime: {regime}")
            
            return {
                'indicators': self.strategy_details[regime]['preferred_indicators'],
                'min_confidence': self.min_confidence,
                'description': self.strategy_details[regime]['description']
            }
        except Exception as e:
            logger.error(f"Error getting strategy requirements: {e}")
            raise

    def print_regime_groups(self, regime_groups: dict):
        """Enhanced visualization of regime groups"""
        try:
            print("\n" + "="*55)
            print(f"{'STRATEGY GROUPS':^55}")
            print("="*55)
            
            for regime, tokens in regime_groups.items():
                if not tokens:
                    continue
                    
                details = self.strategy_details.get(regime, {})
                color = details.get('color', '')
                reset = '\033[0m'
                
                print(f"\n{color}■ {regime} Strategy ({len(tokens)} tokens) {reset}")
                print(f"➤ Strategy Type: {details.get('description', '')}")
                print("-"*55)
                
                for token in tokens:
                    confidence = token['confidence']
                    symbol = token['symbol']
                    price = token.get('price', 'N/A')
                    print(f"  ▸ {symbol:<12} Confidence: {confidence:.2f} Price: {price}")
                
                print(f"{color}{'-'*55}{reset}")
        
        except Exception as e:
            logger.error(f"Error printing regime groups: {e}")
            raise

    def update_performance_metrics(self, regime: str, metrics: Dict):
        """Update performance metrics for a regime"""
        try:
            if regime in self.strategy_details:
                self.performance_metrics[regime] = metrics
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
if __name__ == "__main__":
    try:
        # Initialize components
        analysis_system = MarketAnalysisSystem(testnet=True)
        risk_manager = RiskManager(exchange='binance')
        
        # Fetch and analyze market data
        regime_groups, full_analysis = analysis_system.get_regime_groups()
        risk_report = risk_manager.analyze_risk(full_analysis)

        # Print detailed analysis
        print("\n" + "="*55)
        print(f"{'MARKET REGIME ANALYSIS':^55}")
        print("="*55)
        
        headers = ["Token", "Regime", "Confidence", "Risk Score", "Recommendation"]
        rows = []
        
        for token in risk_report:
            rows.append([
                token['token'],
                token['regime'],
                f"{token['confidence']:.2f}",
                f"{token['risk_score']:.2f}",
                token['recommendation']
            ])
                
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Print strategy groups
        analysis_system.strategy_manager.print_regime_groups(regime_groups)

    except Exception as e:
        logger.error(f"System failure: {str(e)}")
        raise
