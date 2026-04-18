"""
Redis cache implementation for model responses.
"""
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from src.models import RouteResponse, CacheEntry


class ResponseCache:
    """Redis-based cache for model responses."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self._redis: Optional[redis.Redis] = None
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to Redis server."""
        try:
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self._redis.ping()
            self._connected = True
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            self._redis.close()
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self._connected or not self._redis:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            self._connected = False
            return False
    
    def _generate_key(self, model: str, messages: list, params: dict) -> str:
        """Generate a cache key from request parameters."""
        key_data = {
            "model": model,
            "messages": messages,
            "params": params
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return f"router:cache:{hashlib.sha256(key_str.encode()).hexdigest()[:32]}"
    
    def get(
        self,
        model: str,
        messages: list,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[RouteResponse]:
        """Get cached response if available."""
        if not self.is_connected():
            return None
        
        key = self._generate_key(model, messages, params or {})
        
        try:
            data = self._redis.get(key)
            if not data:
                return None
            
            entry = CacheEntry.model_validate_json(data)
            
            if entry.is_expired():
                self._redis.delete(key)
                return None
            
            response = entry.response
            response.cached = True
            return response
            
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        model: str,
        messages: list,
        response: RouteResponse,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Store response in cache."""
        if not self.is_connected():
            return False
        
        key = self._generate_key(model, messages, params or {})
        ttl = ttl or self.default_ttl
        
        try:
            entry = CacheEntry(
                key=key,
                response=response,
                ttl_seconds=ttl
            )
            
            self._redis.setex(
                key,
                ttl,
                entry.model_dump_json()
            )
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def invalidate(self, pattern: str = "router:cache:*") -> int:
        """Invalidate cached entries matching pattern."""
        if not self.is_connected():
            return 0
        
        try:
            keys = self._redis.keys(pattern)
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache invalidate error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            info = self._redis.info()
            keys = self._redis.keys("router:cache:*")
            
            return {
                "connected": True,
                "cached_entries": len(keys),
                "used_memory": info.get("used_memory_human", "unknown"),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1
                )
            }
        except Exception as e:
            return {"connected": True, "error": str(e)}


class MockCache:
    """In-memory mock cache for testing without Redis."""
    
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self._connected = True
    
    def connect(self) -> bool:
        return True
    
    def disconnect(self):
        self._cache.clear()
    
    def is_connected(self) -> bool:
        return True
    
    def _generate_key(self, model: str, messages: list, params: dict) -> str:
        key_data = {
            "model": model,
            "messages": messages,
            "params": params
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return f"router:cache:{hashlib.sha256(key_str.encode()).hexdigest()[:32]}"
    
    def get(
        self,
        model: str,
        messages: list,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[RouteResponse]:
        key = self._generate_key(model, messages, params or {})
        entry = self._cache.get(key)
        
        if not entry or entry.is_expired():
            if entry:
                del self._cache[key]
            return None
        
        response = entry.response
        response.cached = True
        return response
    
    def set(
        self,
        model: str,
        messages: list,
        response: RouteResponse,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        key = self._generate_key(model, messages, params or {})
        entry = CacheEntry(
            key=key,
            response=response,
            ttl_seconds=ttl or self.default_ttl
        )
        self._cache[key] = entry
        return True
    
    def invalidate(self, pattern: str = "") -> int:
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "cached_entries": len(self._cache),
            "used_memory": "mock",
            "hit_rate": 0.0
        }
