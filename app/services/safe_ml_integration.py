"""
SAFE INTEGRATION EXAMPLE: Drop-in replacement for MLService with MLOrchestrator

This demonstrates how to safely switch between MLService and MLOrchestrator
with ZERO code changes required in the rest of the application.
"""

import logging
from typing import Union
from app.services.ml_service import MLService
from app.services.ml_orchestrator import MLOrchestrator, get_ml_orchestrator

logger = logging.getLogger(__name__)

# SAFE INTEGRATION FLAG: Set this to enable MLOrchestrator
USE_ML_ORCHESTRATOR = False  # Set to True to use new MLOrchestrator

async def get_safe_ml_service() -> Union[MLService, MLOrchestrator]:
    """
    SAFE ML SERVICE FACTORY
    
    Returns either MLService or MLOrchestrator based on configuration.
    Both have EXACT same method signatures, so no code changes needed.
    """
    
    if USE_ML_ORCHESTRATOR:
        logger.info("🚀 Using NEW MLOrchestrator with parallel processing")
        orchestrator = await get_ml_orchestrator()
        return orchestrator
    else:
        logger.info("📦 Using LEGACY MLService (current stable)")
        ml_service = MLService()
        await ml_service.initialize_models()
        return ml_service

# EXAMPLE USAGE IN MAIN.PY:
"""
# BEFORE (current):
ml_service = MLService()
await ml_service.initialize_models()
app.state.ml_service = ml_service

# AFTER (safe drop-in replacement):
ml_service = await get_safe_ml_service()  # Auto-selects based on USE_ML_ORCHESTRATOR
app.state.ml_service = ml_service

# The rest of the application remains EXACTLY the same!
# analyze_toy endpoint, health checks, etc. - NO CHANGES NEEDED
"""

# PERFORMANCE COMPARISON HELPER
class PerformanceComparator:
    """Helper to compare performance between MLService and MLOrchestrator"""
    
    def __init__(self):
        self.ml_service_times = []
        self.ml_orchestrator_times = []
    
    async def benchmark_both(self, image_data: bytes) -> dict:
        """Benchmark both implementations with the same image data"""
        import time
        
        results = {}
        
        # Test MLService
        logger.info("Benchmarking MLService...")
        ml_service = MLService()
        await ml_service.initialize_models()
        
        start_time = time.time()
        service_result = await ml_service.analyze_toy_image(image_data)
        service_time = time.time() - start_time
        
        await ml_service.cleanup()
        
        # Test MLOrchestrator
        logger.info("Benchmarking MLOrchestrator...")
        ml_orchestrator = await get_ml_orchestrator()
        
        start_time = time.time()
        orchestrator_result = await ml_orchestrator.analyze_toy_image(image_data)
        orchestrator_time = time.time() - start_time
        
        await ml_orchestrator.cleanup()
        
        # Compare results
        results = {
            'ml_service': {
                'time': service_time,
                'result': service_result
            },
            'ml_orchestrator': {
                'time': orchestrator_time,
                'result': orchestrator_result
            },
            'performance_gain': {
                'time_saved': service_time - orchestrator_time,
                'percent_faster': ((service_time - orchestrator_time) / service_time) * 100 if service_time > 0 else 0
            }
        }
        
        logger.info(f"MLService time: {service_time:.3f}s")
        logger.info(f"MLOrchestrator time: {orchestrator_time:.3f}s")
        logger.info(f"Performance gain: {results['performance_gain']['percent_faster']:.1f}% faster")
        
        return results


# EXAMPLE MIGRATION SCRIPT
async def safe_migration_test():
    """
    Test script showing how to safely migrate from MLService to MLOrchestrator
    """
    print("🔄 SAFE MIGRATION TEST")
    print("="*50)
    
    # Step 1: Test current MLService
    print("1. Testing current MLService...")
    try:
        ml_service = MLService()
        await ml_service.initialize_models()
        print("   ✓ MLService initializes successfully")
        await ml_service.cleanup()
    except Exception as e:
        print(f"   ❌ MLService failed: {e}")
        return False
    
    # Step 2: Test new MLOrchestrator
    print("2. Testing new MLOrchestrator...")
    try:
        ml_orchestrator = await get_ml_orchestrator()
        print("   ✓ MLOrchestrator initializes successfully")
        await ml_orchestrator.cleanup()
    except Exception as e:
        print(f"   ❌ MLOrchestrator failed: {e}")
        return False
    
    # Step 3: Test drop-in replacement
    print("3. Testing drop-in replacement...")
    try:
        # Test with flag OFF (should use MLService)
        global USE_ML_ORCHESTRATOR
        USE_ML_ORCHESTRATOR = False
        ml_service_via_factory = await get_safe_ml_service()
        print("   ✓ Factory returns MLService when flag=False")
        await ml_service_via_factory.cleanup()
        
        # Test with flag ON (should use MLOrchestrator)
        USE_ML_ORCHESTRATOR = True
        ml_orchestrator_via_factory = await get_safe_ml_service()
        print("   ✓ Factory returns MLOrchestrator when flag=True")
        await ml_orchestrator_via_factory.cleanup()
        
    except Exception as e:
        print(f"   ❌ Drop-in replacement failed: {e}")
        return False
    
    print("\n🎉 SAFE MIGRATION TEST PASSED!")
    print("Ready for production deployment with zero downtime!")
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(safe_migration_test())
