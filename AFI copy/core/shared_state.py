import json
from datetime import datetime
from typing import Dict, Optional
import redis
from redis.lock import Lock

class SharedStateManager:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """Initialize shared state manager with Redis backend"""
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.local_cache: Dict[str, dict] = {}
        self.ttl = 60  # Cache expiration in seconds

    def update_state(self, key: str, data: dict) -> bool:
        """
        Thread-safe state update
        Args:
            key: State key (e.g. 'market_regime')
            data: Dictionary of state data
        Returns:
            bool: True if update succeeded
        """
        try:
            # Serialize data with timestamp
            state_data = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            serialized = json.dumps(state_data)
            
            # Update with Redis lock
            with Lock(self.redis, f"lock_{key}", timeout=5):
                self.redis.setex(key, self.ttl, serialized)
                self.local_cache[key] = state_data
            return True
        except Exception as e:
            print(f"Failed to update state: {str(e)}")
            return False

    def get_state(self, key: str) -> Optional[dict]:
        """
        Get cached state with fallback
        Args:
            key: State key to retrieve
        Returns:
            dict: State data if available, None otherwise
        """
        # Check local cache first
        if key in self.local_cache:
            return self.local_cache[key]['data']
            
        # Fallback to Redis
        try:
            serialized = self.redis.get(key)
            if serialized:
                data = json.loads(serialized)
                self.local_cache[key] = data
                return data['data']
        except Exception as e:
            print(f"Failed to get state: {str(e)}")
            
        return None

    def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            return self.redis.ping()
        except Exception:
            return False
