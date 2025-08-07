#!/usr/bin/env python3
"""
ToyResaleWizard FastAPI Backend
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.core.middleware import SecurityMiddleware, add_security_headers
from app.services.ml_service_simple import MLService, get_ml_orchestrator
from app.services.pricing_service import PricingService
from app.api.routes import analysis, health, pricing, toys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize global MLOrchestrator (single instance for all requests)
        logger.info("Initializing global MLOrchestrator...")
        ml_orchestrator = await get_ml_orchestrator()
        app.state.ml_orchestrator = ml_orchestrator
        
        # Initialize ML service (for backwards compatibility)
        logger.info("Initializing ML service...")
        ml_service = MLService()
        await ml_service.initialize_models()
        app.state.ml_service = ml_service
        
        # Initialize pricing service
        logger.info("Initializing pricing service...")
        pricing_service = PricingService()
        app.state.pricing_service = pricing_service
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down application...")
        
        # Clean up MLOrchestrator (primary cleanup)
        if hasattr(app.state, 'ml_orchestrator'):
            logger.info("Cleaning up MLOrchestrator...")
            await app.state.ml_orchestrator.cleanup()
        
        # Clean up legacy ML service
        if hasattr(app.state, 'ml_service'):
            # Clean up ML models and GPU memory
            if hasattr(app.state.ml_service, 'cleanup'):
                await app.state.ml_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="ToyResaleWizard API",
    description="AI-powered toy identification and valuation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add security middleware (before CORS)
app.add_middleware(SecurityMiddleware)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS.split(",") if settings.ALLOWED_HOSTS else ["*"]
)

# CORS middleware with strict security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With"
    ],  # Specific headers only
    expose_headers=["X-Total-Count"],
    max_age=600  # Cache preflight for 10 minutes
)

# Add security headers middleware
@app.middleware("http")
async def security_headers_middleware(request, call_next):
    response = await call_next(request)
    return add_security_headers(response)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(pricing.router, prefix="/api/pricing", tags=["pricing"])
app.include_router(toys.router, prefix="/api/toys", tags=["toys"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ToyResaleWizard API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )