"""
Redis client for caching search results.
"""

import hashlib
import json
from typing import Optional

import redis.asyncio as aioredis

from src.config import CACHE_TTL_SECONDS, REDIS_URL


class RedisCache:
    """Async Redis client for search result caching."""

    def __init__(self) -> None:
        self.client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Initialize the Redis connection pool."""
        self.client = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await self.client.ping()
        print("Redis connected successfully.")

    async def close(self) -> None:
        """Close the Redis connection."""
        if self.client:
            await self.client.close()

    def _make_key(self, query: str, page: int = 1, size: int = 10) -> str:
        """Generate a stable cache key from search parameters."""
        normalized = query.strip().lower()
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        return f"search:{query_hash}:p{page}:s{size}"

    async def get_cached_results(
        self,
        query: str,
        page: int = 1,
        size: int = 10,
    ) -> Optional[dict]:
        """Return cached search results, or None on cache miss."""
        if not self.client:
            return None

        key = self._make_key(query, page, size)
        cached = await self.client.get(key)

        if cached:
            return json.loads(cached)
        return None

    async def cache_results(
        self,
        query: str,
        results: dict,
        page: int = 1,
        size: int = 10,
    ) -> None:
        """Store search results in cache with a TTL."""
        if not self.client:
            return

        key = self._make_key(query, page, size)
        await self.client.setex(key, CACHE_TTL_SECONDS, json.dumps(results))

    async def invalidate_all(self) -> None:
        """Clear all cached search results."""
        if not self.client:
            return

        cursor = 0
        while True:
            cursor, keys = await self.client.scan(cursor, match="search:*", count=100)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break


cache = RedisCache()
