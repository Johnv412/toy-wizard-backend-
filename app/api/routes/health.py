"""
Health check and system status routes
"""

from fastapi import APIRouter, Depends
import psutil
from datetime import datetime
import logging

from app.core.config import settings
from app.services.ml_service_simple import MLService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ToyResaleWizard API",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU information
        gpu_available = False  # Simplified for development
        gpu_count = 0
        gpu_memory = []
        
        # GPU info simplified for development
        gpu_memory = []
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total / 1024**3,  # GB
                    "available": memory.available / 1024**3,  # GB
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total / 1024**3,  # GB
                    "free": disk.free / 1024**3,  # GB
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "gpu": {
                "available": gpu_available,
                "count": gpu_count,
                "devices": gpu_memory
            },
            "config": {
                "upload_dir": settings.UPLOAD_DIR,
                "max_file_size_mb": settings.MAX_FILE_SIZE / 1024**2,
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "cache_expiry_hours": settings.CACHE_EXPIRY_HOURS
            }
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/health/ml")
async def ml_health_check():
    """Check ML service status"""
    try:
        from main import app
        ml_service = app.state.ml_service
        
        # Check if models are loaded
        yolo_status = "loaded" if ml_service.yolo_model is not None else "not_loaded"
        clip_status = "loaded" if ml_service.clip_model is not None else "not_loaded"
        
        # Check device
        device = ml_service.device
        
        return {
            "status": "healthy",
            "ml_service": {
                "yolo_model": yolo_status,
                "clip_model": clip_status,
                "device": device,
                "categories_count": len(ml_service.toy_categories),
                "brands_count": len(ml_service.brand_patterns)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in ML health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/health/database") 
async def database_health_check():
    """Check database connectivity with async support"""
    try:
        from app.core.database import get_db, ToyAnalysis
        from sqlalchemy import select
        
        async for db in get_db():
            # Test database query
            result = await db.execute(select(ToyAnalysis))
            analyses = result.scalars().all()
            count = len(analyses)
            
            return {
                "status": "healthy",
                "database": {
                    "connected": True,
                    "total_analyses": count,
                    "engine_info": str(db.bind.engine.url).replace(db.bind.engine.url.password or "", "***")
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error in database health check: {e}")
        return {
            "status": "unhealthy",
            "database": {
                "connected": False,
                "error": str(e)
            },
            "timestamp": datetime.utcnow().isoformat()
        }