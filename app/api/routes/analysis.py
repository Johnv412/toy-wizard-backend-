"""
Analysis API routes for toy identification and evaluation
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, List
import logging
import uuid
import os
import asyncio
import hashlib
from datetime import datetime, timedelta

from app.core.config import settings
from app.core.database import get_db, ToyAnalysis
from app.core.security import validate_image_file, generate_secure_filename
from app.services.ml_service_simple import MLService, get_ml_orchestrator
from app.services.pricing_service import PricingService
from app.services.cache_warmer import CacheWarmer
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)
router = APIRouter()

# OPTIMIZATION: Global request deduplication cache
active_requests = {}  # Cache for in-flight requests
request_results = {}  # Cache for recent results (5min TTL)

async def get_ml_service() -> MLService:
    """Dependency to get ML service"""
    from main import app
    return app.state.ml_service

async def get_ml_orchestrator_dep() -> MLService:
    """Dependency to get MLOrchestrator directly"""
    from main import app
    return app.state.ml_orchestrator

async def get_pricing_service() -> PricingService:
    """Dependency to get pricing service"""
    from main import app
    return app.state.pricing_service

@router.post("/analyze-toy")
async def analyze_toy(
    file: UploadFile = File(...),
    ml_service: MLService = Depends(get_ml_service),
    pricing_service: PricingService = Depends(get_pricing_service),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a toy image and return comprehensive results with security validation
    """
    try:
        # Read file content
        content = await file.read()
        
        # Comprehensive file validation
        is_valid, error_message = validate_image_file(content, file.filename or "upload.jpg")
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid file: {error_message}")
        
        # Generate secure filename
        secure_filename = generate_secure_filename(file.filename or "upload.jpg")
        image_path = os.path.join(settings.UPLOAD_DIR, secure_filename)
        
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save uploaded image securely
        try:
            with open(image_path, 'wb') as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # OPTIMIZATION: Request deduplication for identical image content
        import hashlib
        content_hash = hashlib.md5(content).hexdigest()
        
        # Check if identical request is already in progress
        if content_hash in active_requests:
            logger.info(f"Deduplicating request for content hash {content_hash[:8]}")
            try:
                return await active_requests[content_hash]
            except Exception as e:
                logger.warning(f"Deduplicated request failed: {e}")
                # Continue with new request if deduplicated one fails
        
        # Check recent results cache (5min TTL)
        if content_hash in request_results:
            result_data, timestamp = request_results[content_hash]
            if datetime.now() - timestamp < timedelta(minutes=5):
                logger.info(f"Returning cached result for content hash {content_hash[:8]}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": {**result_data, "cache_hit": True, "cached_at": timestamp.isoformat()},
                        "message": "Analysis completed successfully (cached)"
                    }
                )
            else:
                # Remove expired cache entry
                del request_results[content_hash]
        
        # Create future for request deduplication
        request_future = asyncio.Future()
        active_requests[content_hash] = request_future
        
        try:
            # OPTIMIZATION: Parallel execution of ML analysis and basic pricing
            logger.info(f"Starting optimized parallel analysis for image {secure_filename}")
            
            # Start ML analysis and basic pricing estimation in parallel
            from app.services.pricing_service import get_pricing_service
            
            # Get optimized pricing service
            optimized_pricing_service = await get_pricing_service()
            
            # Convert image to base64 for AI service
            import base64
            image_base64 = base64.b64encode(content).decode('utf-8')

            # Create parallel tasks
            ai_task = asyncio.create_task(
                ai_service.analyze_toy_image(image_base64)
            )

            # Start basic pricing with default category while AI processes
            basic_pricing_task = asyncio.create_task(
                optimized_pricing_service.get_price_estimate_optimized(
                    toy_name="unknown",  # Will be updated with AI results
                    category="other",    # Default category
                    condition_score=7.0, # Default condition
                    brand=None
                )
            )

            # Wait for AI analysis to complete
            analysis_result = await ai_task
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            request_future.set_exception(e)
            raise
        finally:
            # Cleanup
            if content_hash in active_requests:
                del active_requests[content_hash]
        
        # OPTIMIZATION: Adaptive timeout based on toy complexity
        def calculate_adaptive_timeout(toy_name: str, category: str) -> float:
            """Calculate timeout based on query complexity and category"""
            base_timeout = 1.5
            
            # Longer timeout for complex/rare toys
            complexity_factors = {
                'rare': 1.5, 'vintage': 1.3, 'collectible': 1.4, 
                'limited edition': 1.6, 'exclusive': 1.4
            }
            
            toy_lower = toy_name.lower()
            for factor_key, multiplier in complexity_factors.items():
                if factor_key in toy_lower:
                    base_timeout *= multiplier
                    break
            
            # Category-based timeout adjustments
            category_timeouts = {
                'collectible': 1.3, 'action_figure': 1.1, 'doll': 1.2,
                'vehicle': 1.0, 'plush': 0.9, 'building_set': 1.1
            }
            
            base_timeout *= category_timeouts.get(category, 1.0)
            return min(base_timeout, 4.0)  # Cap at 4 seconds
        
        # Calculate adaptive timeout
        adaptive_timeout = calculate_adaptive_timeout(
            analysis_result['toy_name'], 
            analysis_result['category']
        )
        
        # Get detailed pricing with ML results (may hit cache)
        detailed_pricing_task = asyncio.create_task(
            optimized_pricing_service.get_price_estimate_optimized(
                toy_name=analysis_result['toy_name'],
                category=analysis_result['category'],
                condition_score=analysis_result['condition_score'],
                brand=analysis_result.get('brand')
            )
        )
        
        # Use adaptive timeout for better success rate
        try:
            pricing_result = await asyncio.wait_for(detailed_pricing_task, timeout=adaptive_timeout)
            logger.info(f"Detailed pricing completed within {adaptive_timeout}s timeout")
        except asyncio.TimeoutError:
            logger.warning(f"Detailed pricing timed out after {adaptive_timeout}s, using basic pricing")
            pricing_result = await basic_pricing_task
        
        # OPTIMIZATION: Enhanced result combination with performance metrics
        processing_time = getattr(analysis_result, 'processing_time', None)
        cache_status = getattr(pricing_result, 'cache_status', 'unknown')
        
        final_result = {
            **analysis_result,
            'pricing': pricing_result,
            'image_id': secure_filename.split('.')[0],
            'analysis_id': str(uuid.uuid4()),
            'performance_metrics': {
                'ai_processing_time': processing_time,
                'pricing_cache_status': cache_status,
                'total_api_calls': 2,  # AI + Pricing
                'optimization_used': 'parallel_execution'
            }
        }
        
        # Store analysis in database with async session
        try:
            # Map AI service fields to database model
            rarity_mapping = {
                'common': 3.0,
                'uncommon': 5.0,
                'rare': 7.0,
                'very_rare': 8.0,
                'collectible': 9.0
            }

            toy_analysis = ToyAnalysis(
                toy_name=analysis_result['toy_name'],
                category=analysis_result['category'],
                brand=analysis_result.get('brand'),
                condition_score=analysis_result['condition_score'],
                estimated_price=pricing_result['estimated_price'],
                rarity_score=rarity_mapping.get(analysis_result.get('rarity', 'uncommon'), 5.0),
                confidence=analysis_result['confidence'],
                image_path=image_path,
                analysis_data=str(final_result)
            )
            
            db.add(toy_analysis)
            await db.commit()
            await db.refresh(toy_analysis)
            
            final_result['database_id'] = toy_analysis.id
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            # Continue without database storage rather than failing completely
            final_result['database_id'] = None
            final_result['note'] = "Analysis completed but not stored in database"
        
        # OPTIMIZATION: Cache result for deduplication and cleanup
        request_results[content_hash] = (final_result, datetime.now())
        
        # Cleanup old cache entries (keep last 100)
        if len(request_results) > 100:
            oldest_keys = sorted(request_results.keys(), 
                               key=lambda k: request_results[k][1])[:20]
            for old_key in oldest_keys:
                del request_results[old_key]
        
        # Complete the request future for deduplication
        if not request_future.done():
            request_future.set_result(JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": final_result,
                    "message": "Analysis completed successfully"
                }
            ))
        
        logger.info(f"Analysis completed for {analysis_result['toy_name']}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": final_result,
                "message": "Analysis completed successfully"
            }
        )
        
    except Exception as analysis_error:
        # Ensure request future is completed on error
        if content_hash in active_requests and not request_future.done():
            request_future.set_exception(analysis_error)
        # Clean up uploaded file on error
        if 'image_path' in locals() and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        raise analysis_error
        
    finally:
        # Cleanup active request
        active_requests.pop(content_hash, None)

@router.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific analysis by ID with SQL injection protection"""
    try:
        # Use async query with bound parameters
        stmt = select(ToyAnalysis).where(ToyAnalysis.id == analysis_id)
        result = await db.execute(stmt)
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "success": True,
            "data": {
                "id": analysis.id,
                "toy_name": analysis.toy_name,
                "category": analysis.category,
                "brand": analysis.brand,
                "condition_score": analysis.condition_score,
                "estimated_price": analysis.estimated_price,
                "rarity_score": analysis.rarity_score,
                "confidence": analysis.confidence,
                "created_at": analysis.created_at.isoformat(),
                "updated_at": analysis.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@router.get("/analyses")
async def list_analyses(
    skip: int = 0, 
    limit: int = 20, 
    category: str = None,
    db = Depends(get_db)
):
    """List all analyses with optional filtering"""
    try:
        query = db.query(ToyAnalysis)
        
        if category:
            query = query.filter(ToyAnalysis.category == category)
        
        analyses = query.offset(skip).limit(limit).all()
        total = query.count()
        
        return {
            "success": True,
            "data": {
                "analyses": [
                    {
                        "id": a.id,
                        "toy_name": a.toy_name,
                        "category": a.category,
                        "brand": a.brand,
                        "condition_score": a.condition_score,
                        "estimated_price": a.estimated_price,
                        "rarity_score": a.rarity_score,
                        "confidence": a.confidence,
                        "created_at": a.created_at.isoformat()
                    }
                    for a in analyses
                ],
                "total": total,
                "skip": skip,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to list analyses")

@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: int, db = Depends(get_db)):
    """Delete a specific analysis"""
    try:
        analysis = db.query(ToyAnalysis).filter(ToyAnalysis.id == analysis_id).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Delete image file if it exists
        if analysis.image_path and os.path.exists(analysis.image_path):
            try:
                os.remove(analysis.image_path)
            except Exception as e:
                logger.warning(f"Failed to delete image file: {e}")
        
        db.delete(analysis)
        db.commit()
        
        return {
            "success": True,
            "message": "Analysis deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete analysis")

@router.get("/categories")
async def get_categories():
    """Get list of available toy categories"""
    try:
        categories = [
            {"id": "action_figure", "name": "Action Figures", "description": "Superhero and character figures"},
            {"id": "doll", "name": "Dolls", "description": "Fashion dolls, baby dolls, collectible dolls"},
            {"id": "vehicle", "name": "Vehicles", "description": "Cars, trucks, trains, planes"},
            {"id": "building_set", "name": "Building Sets", "description": "LEGO, blocks, construction toys"},
            {"id": "plush", "name": "Plush Toys", "description": "Teddy bears, stuffed animals"},
            {"id": "board_game", "name": "Board Games", "description": "Board games, card games, puzzles"},
            {"id": "educational", "name": "Educational", "description": "Learning toys, science kits"},
            {"id": "outdoor", "name": "Outdoor Toys", "description": "Sports equipment, outdoor games"},
            {"id": "electronic", "name": "Electronic Toys", "description": "Electronic games, robots"},
            {"id": "collectible", "name": "Collectibles", "description": "Trading cards, figurines"}
        ]
        
        return {
            "success": True,
            "data": {"categories": categories}
        }
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categories")

@router.get("/stats")
async def get_analysis_stats(db = Depends(get_db)):
    """Get analysis statistics"""
    try:
        total_analyses = db.query(ToyAnalysis).count()
        
        # Category distribution
        category_stats = db.query(
            ToyAnalysis.category,
            db.func.count(ToyAnalysis.id).label('count')
        ).group_by(ToyAnalysis.category).all()
        
        # Average metrics
        avg_condition = db.query(db.func.avg(ToyAnalysis.condition_score)).scalar() or 0
        avg_price = db.query(db.func.avg(ToyAnalysis.estimated_price)).scalar() or 0
        avg_rarity = db.query(db.func.avg(ToyAnalysis.rarity_score)).scalar() or 0
        avg_confidence = db.query(db.func.avg(ToyAnalysis.confidence)).scalar() or 0
        
        return {
            "success": True,
            "data": {
                "total_analyses": total_analyses,
                "category_distribution": [
                    {"category": cat, "count": count}
                    for cat, count in category_stats
                ],
                "averages": {
                    "condition_score": round(float(avg_condition), 2),
                    "estimated_price": round(float(avg_price), 2),
                    "rarity_score": round(float(avg_rarity), 2),
                    "confidence": round(float(avg_confidence), 2)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@router.post("/warm-cache")
async def warm_pricing_cache(
    category: str = None,
    pricing_service: PricingService = Depends(get_pricing_service)
):
    """Warm pricing cache for popular toys to improve performance"""
    try:
        from app.services.cache_warmer import CacheWarmer
        cache_warmer = CacheWarmer(pricing_service)
        
        if category:
            # Warm specific category
            result = await cache_warmer.warm_category_cache(category)
            return {
                "success": True,
                "data": result,
                "message": f"Cache warming completed for category: {category}"
            }
        else:
            # Warm all popular toys
            result = await cache_warmer.warm_popular_caches(max_concurrent=3)
            return {
                "success": True,
                "data": result,
                "message": f"Cache warming completed for {result['successful']} toys"
            }
            
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to warm cache")

@router.get("/cache-status")
async def get_cache_status():
    """Get current cache status and statistics"""
    try:
        # Memory cache stats
        memory_cache_size = len(request_results)
        
        # Active requests
        active_requests_count = len(active_requests)
        
        return {
            "success": True,
            "data": {
                "memory_cache": {
                    "size": memory_cache_size,
                    "max_size": 100
                },
                "active_requests": active_requests_count,
                "deduplication_enabled": True,
                "cache_ttl_minutes": 5
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache status")

@router.websocket("/analyze-toy-ws")
async def analyze_toy_websocket(
    websocket: WebSocket,
    ml_service: MLService = Depends(get_ml_service),
    pricing_service: PricingService = Depends(get_pricing_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Real-time toy analysis via WebSocket
    """
    await websocket.accept()
    
    try:
        # Receive image data from client
        data = await websocket.receive_json()
        image_base64 = data.get("image_base64")
        
        if not image_base64:
            await websocket.send_json({"error": "No image data provided"})
            return
        
        # Send status update
        await websocket.send_json({"status": "Analyzing toy image..."})
        
        # Perform AI analysis
        analysis_result = await ai_service.analyze_toy_image(image_base64)
        
        # Send analysis results
        await websocket.send_json({
            "status": "Analysis complete",
            "result": analysis_result
        })
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket analysis error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()