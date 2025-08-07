"""
Simple cache warmer for development
"""

import logging

logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Simple cache warmer for development mode
    """
    
    def __init__(self):
        self.initialized = True
        logger.info("✅ CacheWarmer initialized (development mode)")
    
    async def warm_cache(self):
        """
        Mock cache warming for development
        """
        logger.info("Cache warming completed (mock)")
        return True
    
    async def health_check(self):
        """
        Health check for cache warmer
        """
        return {
            "status": "healthy",
            "mode": "development",
            "initialized": self.initialized
        }


# Global instance
_cache_warmer = None

def get_cache_warmer() -> CacheWarmer:
    """
    Get or create cache warmer instance
    """
    global _cache_warmer
    if _cache_warmer is None:
        _cache_warmer = CacheWarmer()
    return _cache_warmer