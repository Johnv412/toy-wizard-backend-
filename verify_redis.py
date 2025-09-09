#!/usr/bin/env python3
"""
Redis Connection and Caching Verification Script
Tests Redis setup and basic caching operations
"""

import asyncio
import logging
import json
from datetime import datetime

import redis.asyncio as redis
from app.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Test data
TEST_CACHE_KEY = "test:verification:key"
TEST_CACHE_VALUE = {
    "estimated_price": 29.99,
    "price_range": {"min": 20.0, "max": 40.0},
    "timestamp": datetime.now().isoformat(),
    "verified": True
}

async def check_redis_connection():
    """Test basic Redis connection"""
    try:
        # Initialize Redis client
        if hasattr(settings, 'REDIS_URL'):
            redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                encoding='utf-8',
                decode_responses=True,
                max_connections=5
            )
        else:
            logger.error("❌ REDIS_URL not configured in settings")
            return False

        # Test connection with ping
        pong = await redis_client.ping()
        if pong:
            logger.info("✅ Redis connection successful")
            return redis_client
        else:
            logger.error("❌ Redis ping failed")
            return False

    except Exception as e:
        logger.error(f"❌ Redis connection failed: {str(e)}")
        return False

async def test_basic_operations(redis_client):
    """Test basic Redis operations"""
    try:
        # Test SET operation
        await redis_client.set("test:key", "test_value", ex=60)
        logger.info("✅ Redis SET operation successful")

        # Test GET operation
        value = await redis_client.get("test:key")
        if value == "test_value":
            logger.info("✅ Redis GET operation successful")
        else:
            logger.error("❌ Redis GET operation failed - unexpected value")
            return False

        # Test expiration
        await asyncio.sleep(2)  # Wait a bit
        ttl = await redis_client.ttl("test:key")
        if ttl > 0:
            logger.info(f"✅ Redis TTL working (expires in {ttl}s)")
        else:
            logger.warning("⚠️ Redis TTL may not be working as expected")

        return True

    except Exception as e:
        logger.error(f"❌ Redis basic operations failed: {str(e)}")
        return False

async def test_caching_simulation(redis_client):
    """Test caching operations similar to pricing service"""
    try:
        # Test JSON serialization (like pricing service)
        json_value = json.dumps(TEST_CACHE_VALUE)
        await redis_client.set(TEST_CACHE_KEY, json_value, ex=3600)
        logger.info("✅ Redis cache SET with JSON successful")

        # Test cache retrieval
        cached_json = await redis_client.get(TEST_CACHE_KEY)
        if cached_json:
            cached_data = json.loads(cached_json)
            if cached_data.get("verified"):
                logger.info("✅ Redis cache GET and JSON parsing successful")
            else:
                logger.error("❌ Redis cache data verification failed")
                return False
        else:
            logger.error("❌ Redis cache GET failed - no data retrieved")
            return False

        # Test cache key pattern matching (like pricing service)
        keys = await redis_client.keys("test:*")
        if TEST_CACHE_KEY in keys:
            logger.info("✅ Redis key pattern matching successful")
        else:
            logger.warning("⚠️ Redis key pattern matching may not work as expected")

        return True

    except Exception as e:
        logger.error(f"❌ Redis caching simulation failed: {str(e)}")
        return False

async def cleanup_test_data(redis_client):
    """Clean up test data"""
    try:
        await redis_client.delete("test:key", TEST_CACHE_KEY)
        logger.info("✅ Test data cleanup successful")
    except Exception as e:
        logger.warning(f"⚠️ Test data cleanup failed: {str(e)}")

async def main():
    """Main verification function"""
    logger.info("🔍 Starting Redis connection and caching verification...")

    # Check connection
    redis_client = await check_redis_connection()
    if not redis_client:
        logger.error("💥 Redis verification failed - connection issues")
        return 1

    # Test basic operations
    basic_ok = await test_basic_operations(redis_client)
    if not basic_ok:
        await cleanup_test_data(redis_client)
        logger.error("💥 Redis verification failed - basic operations issues")
        return 1

    # Test caching simulation
    cache_ok = await test_caching_simulation(redis_client)
    if not cache_ok:
        await cleanup_test_data(redis_client)
        logger.error("💥 Redis verification failed - caching issues")
        return 1

    # Cleanup
    await cleanup_test_data(redis_client)

    logger.info("🎉 Redis setup verification completed successfully!")
    logger.info("✅ Redis connection: OK")
    logger.info("✅ Basic operations: OK")
    logger.info("✅ Caching functionality: OK")

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)