import redis
import json
from functools import wraps
import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)

def cache_key(*args, **kwargs):
    """Generate a cache key from function arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

def cache_data(expire_time=300):
    """
    Cache decorator that stores function results in Redis
    :param expire_time: Time in seconds before cache expires (default: 5 minutes)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get cached result
            cached_result = redis_client.get(key)
            if cached_result:
                return json.loads(cached_result)
            
            # If not cached, execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(
                key,
                timedelta(seconds=expire_time),
                json.dumps(result)
            )
            return result
        return wrapper
    return decorator

def clear_cache(pattern="*"):
    """Clear cache entries matching the given pattern"""
    for key in redis_client.scan_iter(pattern):
        redis_client.delete(key)

def cache_stock_data(symbol: str, data: dict, expire_time=300):
    """Cache stock data with expiration"""
    key = f"stock:{symbol}"
    redis_client.setex(key, timedelta(seconds=expire_time), json.dumps(data))

def get_cached_stock_data(symbol: str):
    """Retrieve cached stock data"""
    key = f"stock:{symbol}"
    data = redis_client.get(key)
    return json.loads(data) if data else None

def cache_user_preferences(user_id: int, preferences: dict):
    """Cache user preferences"""
    key = f"user:preferences:{user_id}"
    redis_client.set(key, json.dumps(preferences))

def get_cached_user_preferences(user_id: int):
    """Retrieve cached user preferences"""
    key = f"user:preferences:{user_id}"
    data = redis_client.get(key)
    return json.loads(data) if data else None

def invalidate_user_cache(user_id: int):
    """Invalidate all cache entries for a specific user"""
    pattern = f"user:*:{user_id}"
    clear_cache(pattern)

# Example usage of cache decorator
@cache_data(expire_time=300)
def get_stock_analysis(symbol: str, timeframe: str = "1d"):
    """Example function using cache decorator"""
    # Implement your stock analysis logic here
    pass 