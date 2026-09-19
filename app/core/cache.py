import redis.asyncio as redis
import json
import hashlib
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

# connection pool created once — reused across all requests
_redis_client = None

async def get_redis():
    """Get or create Redis connection."""
    global _redis_client
    if _redis_client is None:
        logger.info(f"Connecting to Redis at {settings.REDIS_URL}...")
        _redis_client = await redis.from_url(settings.REDIS_URL)
        logger.info("Connected to Redis.")
    return _redis_client

def cache_key(query: str) -> str:
    """Generate a deterministic cache key from the query."""
    return f"arxiv-agent:{hashlib.md5(query.encode()).hexdigest()}"

async def get_cached(query:str) -> dict | None:
    """Return cached result if it exists, None otherwise."""
    try:
        client = await get_redis()
        cached = await client.get(cache_key(query))
        if cached:
            logger.info(f"Cache hit for query: '{query[:50]}'")
            return json.loads(cached)
        else:
            logger.info(f"Cache miss for query: '{query[:50]}'")
            return None
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
        return None

async def set_cached(query:str, result:dict, ttl:int=21600):
    """Cache result with TTL in seconds (default 6 hours)."""
    try:
        client = await get_redis()
        await client.set(cache_key(query), json.dumps(result), ex=ttl)
        logger.info(f"Cached result for query: '{query[:50]}' with TTL {ttl}s")
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")