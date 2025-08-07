"""
Toy management routes
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging

from app.core.database import get_db, ToyAnalysis

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def list_toys(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_condition: Optional[float] = None,
    max_condition: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Optional[str] = Query("created_at", regex="^(created_at|estimated_price|condition_score|rarity_score)$"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
    db = Depends(get_db)
):
    """List toys with filtering and sorting options"""
    try:
        query = db.query(ToyAnalysis)
        
        # Apply filters
        if category:
            query = query.filter(ToyAnalysis.category == category)
        
        if brand:
            query = query.filter(ToyAnalysis.brand == brand)
        
        if min_condition is not None:
            query = query.filter(ToyAnalysis.condition_score >= min_condition)
        
        if max_condition is not None:
            query = query.filter(ToyAnalysis.condition_score <= max_condition)
        
        if min_price is not None:
            query = query.filter(ToyAnalysis.estimated_price >= min_price)
        
        if max_price is not None:
            query = query.filter(ToyAnalysis.estimated_price <= max_price)
        
        # Apply sorting
        if sort_order == "desc":
            query = query.order_by(getattr(ToyAnalysis, sort_by).desc())
        else:
            query = query.order_by(getattr(ToyAnalysis, sort_by).asc())
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        toys = query.offset(skip).limit(limit).all()
        
        return {
            "success": True,
            "data": {
                "toys": [
                    {
                        "id": toy.id,
                        "toy_name": toy.toy_name,
                        "category": toy.category,
                        "brand": toy.brand,
                        "condition_score": toy.condition_score,
                        "estimated_price": toy.estimated_price,
                        "rarity_score": toy.rarity_score,
                        "confidence": toy.confidence,
                        "created_at": toy.created_at.isoformat(),
                        "updated_at": toy.updated_at.isoformat()
                    }
                    for toy in toys
                ],
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": total,
                    "has_more": skip + limit < total
                },
                "filters": {
                    "category": category,
                    "brand": brand,
                    "min_condition": min_condition,
                    "max_condition": max_condition,
                    "min_price": min_price,
                    "max_price": max_price,
                    "sort_by": sort_by,
                    "sort_order": sort_order
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing toys: {e}")
        raise HTTPException(status_code=500, detail="Failed to list toys")

@router.get("/{toy_id}")
async def get_toy(toy_id: int, db = Depends(get_db)):
    """Get detailed information about a specific toy"""
    try:
        toy = db.query(ToyAnalysis).filter(ToyAnalysis.id == toy_id).first()
        
        if not toy:
            raise HTTPException(status_code=404, detail="Toy not found")
        
        return {
            "success": True,
            "data": {
                "id": toy.id,
                "toy_name": toy.toy_name,
                "category": toy.category,
                "brand": toy.brand,
                "condition_score": toy.condition_score,
                "estimated_price": toy.estimated_price,
                "rarity_score": toy.rarity_score,
                "confidence": toy.confidence,
                "image_path": toy.image_path,
                "analysis_data": toy.analysis_data,
                "created_at": toy.created_at.isoformat(),
                "updated_at": toy.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting toy {toy_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get toy")

@router.get("/search/{query}")
async def search_toys(
    query: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db = Depends(get_db)
):
    """Search toys by name, brand, or category"""
    try:
        search_filter = f"%{query.lower()}%"
        
        toys = db.query(ToyAnalysis).filter(
            db.or_(
                ToyAnalysis.toy_name.ilike(search_filter),
                ToyAnalysis.brand.ilike(search_filter),
                ToyAnalysis.category.ilike(search_filter)
            )
        ).offset(skip).limit(limit).all()
        
        total = db.query(ToyAnalysis).filter(
            db.or_(
                ToyAnalysis.toy_name.ilike(search_filter),
                ToyAnalysis.brand.ilike(search_filter),
                ToyAnalysis.category.ilike(search_filter)
            )
        ).count()
        
        return {
            "success": True,
            "data": {
                "query": query,
                "toys": [
                    {
                        "id": toy.id,
                        "toy_name": toy.toy_name,
                        "category": toy.category,
                        "brand": toy.brand,
                        "condition_score": toy.condition_score,
                        "estimated_price": toy.estimated_price,
                        "rarity_score": toy.rarity_score,
                        "confidence": toy.confidence,
                        "created_at": toy.created_at.isoformat()
                    }
                    for toy in toys
                ],
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": total,
                    "has_more": skip + limit < total
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error searching toys with query '{query}': {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@router.get("/brands/list")
async def list_brands(db = Depends(get_db)):
    """Get list of all brands in the database"""
    try:
        brands = db.query(ToyAnalysis.brand).filter(
            ToyAnalysis.brand.isnot(None)
        ).distinct().all()
        
        brand_list = [brand[0] for brand in brands if brand[0]]
        
        return {
            "success": True,
            "data": {
                "brands": sorted(brand_list),
                "count": len(brand_list)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing brands: {e}")
        raise HTTPException(status_code=500, detail="Failed to list brands")

@router.get("/categories/stats")
async def get_category_stats(db = Depends(get_db)):
    """Get statistics by category"""
    try:
        stats = db.query(
            ToyAnalysis.category,
            db.func.count(ToyAnalysis.id).label('count'),
            db.func.avg(ToyAnalysis.condition_score).label('avg_condition'),
            db.func.avg(ToyAnalysis.estimated_price).label('avg_price'),
            db.func.avg(ToyAnalysis.rarity_score).label('avg_rarity'),
            db.func.min(ToyAnalysis.estimated_price).label('min_price'),
            db.func.max(ToyAnalysis.estimated_price).label('max_price')
        ).group_by(ToyAnalysis.category).all()
        
        category_stats = []
        for stat in stats:
            category_stats.append({
                "category": stat.category,
                "count": stat.count,
                "avg_condition": round(float(stat.avg_condition or 0), 2),
                "avg_price": round(float(stat.avg_price or 0), 2),
                "avg_rarity": round(float(stat.avg_rarity or 0), 2),
                "price_range": {
                    "min": float(stat.min_price or 0),
                    "max": float(stat.max_price or 0)
                }
            })
        
        return {
            "success": True,
            "data": {
                "category_stats": category_stats,
                "total_categories": len(category_stats)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting category stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get category statistics")

@router.delete("/{toy_id}")
async def delete_toy(toy_id: int, db = Depends(get_db)):
    """Delete a toy from the database"""
    try:
        toy = db.query(ToyAnalysis).filter(ToyAnalysis.id == toy_id).first()
        
        if not toy:
            raise HTTPException(status_code=404, detail="Toy not found")
        
        # Delete associated image file if it exists
        import os
        if toy.image_path and os.path.exists(toy.image_path):
            try:
                os.remove(toy.image_path)
            except Exception as e:
                logger.warning(f"Failed to delete image file: {e}")
        
        db.delete(toy)
        db.commit()
        
        return {
            "success": True,
            "message": f"Toy '{toy.toy_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting toy {toy_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete toy")