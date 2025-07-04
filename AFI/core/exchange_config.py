import yaml
import logging
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class ExchangeConfig:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self.last_testnet_check = None
        self.testnet_available = False
        self.mainnet_fallback = self.config.get('exchange', {}).get('mainnet_fallback', True)
        self.recv_window = self.config.get('exchange', {}).get('recv_window', 5000)
        self.max_retry_count = self.config.get('exchange', {}).get('max_retry_count', 3)
        self.retry_delay = self.config.get('exchange', {}).get('retry_delay', 5)
        self.testnet_check_interval = self.config.get('exchange', {}).get('testnet_check_interval', 300)  # 5 minutes
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            script_dir = Path(__file__).parent
            config_file = script_dir / self.config_path
            
            # Check different possible locations
            if not config_file.exists():
                config_file = script_dir.parent / self.config_path
            if not config_file.exists():
                config_file = Path(self.config_path).absolute()
                
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return {}

    def get_exchange_params(self, testnet: bool = True) -> Dict:
        """Get exchange configuration parameters"""
        params = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'recvWindow': self.recv_window,
                'adjustForTimeDifference': True
            },
            'timeout': 30000,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
        
        # Only add API credentials if they exist and we're not in paper trading mode
        if not self.config.get('trading', {}).get('paper_trading', True):
            if testnet and 'testnet_api' in self.config.get('exchange', {}):
                params.update({
                    'apiKey': self.config['exchange']['testnet_api'].get('key'),
                    'secret': self.config['exchange']['testnet_api'].get('secret')
                })
            elif not testnet and 'mainnet_api' in self.config.get('exchange', {}):
                params.update({
                    'apiKey': self.config['exchange']['mainnet_api'].get('key'),
                    'secret': self.config['exchange']['mainnet_api'].get('secret')
                })
        
        return params

    def check_testnet_availability(self) -> bool:
        """Check if enough time has passed to retry testnet"""
        current_time = datetime.now()
        
        if (self.last_testnet_check is None or 
            (current_time - self.last_testnet_check).total_seconds() > self.testnet_check_interval):
            self.last_testnet_check = current_time
            return True
            
        return False

    def should_use_mainnet_fallback(self) -> bool:
        """Determine if mainnet fallback should be used"""
        return self.mainnet_fallback and not self.testnet_available

    def update_testnet_status(self, available: bool):
        """Update testnet availability status"""
        self.testnet_available = available
        self.last_testnet_check = datetime.now()

    def get_retry_settings(self) -> Dict:
        """Get retry configuration"""
        return {
            'max_retries': self.max_retry_count,
            'delay': self.retry_delay,
            'backoff_factor': 2
        }