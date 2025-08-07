"""
Pricing API routes for market data and price estimates
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, bindparam
from typing import Optional
import logging
from datetime import datetime, timedelta

from app.core.database import get_db, PriceHistory
from app.services.pricing_service import PricingService

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_pricing_service() -> PricingService:
    """Dependency to get pricing service"""
    from main import app
    return app.state.pricing_service

@router.get("/estimate")
async def get_price_estimate(
    toy_name: str,
    category: str,
    condition_score: float = Query(..., ge=1.0, le=10.0),
    brand: Optional[str] = None,
    pricing_service: PricingService = Depends(get_pricing_service)
):
    """Get price estimate for a toy"""
    try:
        estimate = await pricing_service.get_price_estimate(
            toy_name=toy_name,
            category=category,
            condition_score=condition_score,
            brand=brand
        )
        
        return {
            "success": True,
            "data": estimate,
            "request": {
                "toy_name": toy_name,
                "category": category,
                "condition_score": condition_score,
                "brand": brand
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting price estimate: {e}")
        raise HTTPException(status_code=500, detail="Failed to get price estimate")

@router.get("/history/{toy_name}")
async def get_price_history(
    toy_name: str = Path(..., min_length=1, max_length=200, regex="^[a-zA-Z0-9\\s\\-_]+$"),
    days: int = Query(30, ge=1, le=365),
    source: Optional[str] = Query(None, regex="^[a-zA-Z0-9_]+$"),
    db: AsyncSession = Depends(get_db)
):
    """Get price history for a specific toy with SQL injection protection"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Use parameterized queries to prevent SQL injection
        stmt = select(PriceHistory).where(
            and_(
                PriceHistory.toy_name.ilike(bindparam("toy_pattern")),
                PriceHistory.created_at >= bindparam("cutoff_date")
            )
        )
        
        if source:
            stmt = stmt.where(PriceHistory.source == bindparam("source_filter"))
        
        # Execute with bound parameters
        if source:
            result = await db.execute(
                stmt,
                {
                    "toy_pattern": f"%{toy_name}%",
                    "cutoff_date": cutoff_date,
                    "source_filter": source
                }
            )
        else:
            result = await db.execute(
                stmt,
                {
                    "toy_pattern": f"%{toy_name}%", 
                    "cutoff_date": cutoff_date
                }
            )
        
        price_records = result.scalars().all()
        
        # Process the data
        history_data = []
        for record in price_records:
            history_data.append({
                "price": record.price,
                "condition": record.condition,
                "source": record.source,
                "sold_date": record.sold_date.isoformat() if record.sold_date else None,
                "listing_url": record.listing_url,
                "created_at": record.created_at.isoformat()
            })
        
        # Calculate statistics
        prices = [r.price for r in price_records]
        stats = {}
        if prices:
            stats = {
                "count": len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "median_price": sorted(prices)[len(prices) // 2]
            }
        
        return {
            "success": True,
            "data": {
                "toy_name": toy_name,
                "days": days,
                "source_filter": source,
                "history": history_data,
                "statistics": stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting price history for '{toy_name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to get price history")

@router.get("/trends")
async def get_pricing_trends(
    category: Optional[str] = None,
    brand: Optional[str] = None,
    days: int = Query(30, ge=7, le=365),
    db = Depends(get_db)
):
    """Get pricing trends and market analysis"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Base query for price history
        query = db.query(PriceHistory).filter(
            PriceHistory.created_at >= cutoff_date
        )
        
        # Apply filters if provided
        if category:
            # Note: This assumes toy_name contains category info
            # In production, you'd want a proper category field
            query = query.filter(PriceHistory.toy_name.ilike(f"%{category}%"))
        
        if brand:
            query = query.filter(PriceHistory.toy_name.ilike(f"%{brand}%"))
        
        price_records = query.order_by(PriceHistory.created_at.asc()).all()
        
        # Group by time periods (weekly)
        trends = {}
        for record in price_records:
            week_start = record.created_at.date() - timedelta(days=record.created_at.weekday())
            week_key = week_start.isoformat()
            
            if week_key not in trends:
                trends[week_key] = []
            trends[week_key].append(record.price)
        
        # Calculate weekly averages
        trend_data = []
        for week, prices in sorted(trends.items()):
            trend_data.append({
                "week": week,
                "avg_price": sum(prices) / len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "transaction_count": len(prices)
            })
        
        # Calculate overall trend direction
        if len(trend_data) >= 2:
            first_avg = trend_data[0]["avg_price"]
            last_avg = trend_data[-1]["avg_price"]
            trend_direction = "up" if last_avg > first_avg else "down" if last_avg < first_avg else "stable"
            trend_percentage = ((last_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        else:
            trend_direction = "insufficient_data"
            trend_percentage = 0
        
        return {
            "success": True,
            "data": {
                "filters": {
                    "category": category,
                    "brand": brand,
                    "days": days
                },
                "trend_direction": trend_direction,
                "trend_percentage": round(trend_percentage, 2),
                "weekly_data": trend_data,
                "total_transactions": len(price_records)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting pricing trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pricing trends")

@router.get("/market-summary")
async def get_market_summary(db = Depends(get_db)):
    """Get overall market summary and statistics"""
    try:
        # Get data from last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        
        recent_prices = db.query(PriceHistory).filter(
            PriceHistory.created_at >= cutoff_date
        ).all()
        
        if not recent_prices:
            return {
                "success": True,
                "data": {
                    "message": "No recent market data available",
                    "summary": {}
                }
            }
        
        # Calculate overall statistics
        prices = [p.price for p in recent_prices]
        
        # Group by source
        source_stats = {}
        for record in recent_prices:
            source = record.source
            if source not in source_stats:
                source_stats[source] = []
            source_stats[source].append(record.price)
        
        # Calculate source averages
        source_averages = {}
        for source, source_prices in source_stats.items():
            source_averages[source] = {
                "avg_price": sum(source_prices) / len(source_prices),
                "transaction_count": len(source_prices),
                "min_price": min(source_prices),
                "max_price": max(source_prices)
            }
        
        # Most active categories (based on toy names)
        category_activity = {}
        common_categories = ["lego", "pokemon", "star wars", "barbie", "marvel", "transformers"]
        
        for category in common_categories:
            count = sum(1 for p in recent_prices if category.lower() in p.toy_name.lower())
            if count > 0:
                category_prices = [p.price for p in recent_prices if category.lower() in p.toy_name.lower()]
                category_activity[category] = {
                    "transaction_count": count,
                    "avg_price": sum(category_prices) / len(category_prices)
                }
        
        return {
            "success": True,
            "data": {
                "period": "Last 30 days",
                "overall_stats": {
                    "total_transactions": len(recent_prices),
                    "avg_price": sum(prices) / len(prices),
                    "min_price": min(prices),
                    "max_price": max(prices),
                    "median_price": sorted(prices)[len(prices) // 2]
                },
                "source_breakdown": source_averages,
                "category_activity": category_activity,
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market summary")

@router.get("/sources")
async def get_price_sources():
    """Get list of available pricing sources"""
    try:
        sources = [
            {
                "id": "ebay",
                "name": "eBay",
                "description": "Sold listings from eBay marketplace",
                "type": "auction",
                "reliability": "high"
            },
            {
                "id": "amazon",
                "name": "Amazon",
                "description": "Current listings on Amazon",
                "type": "fixed_price",
                "reliability": "medium"
            },
            {
                "id": "mercari",
                "name": "Mercari",
                "description": "Sold listings from Mercari marketplace",
                "type": "fixed_price",
                "reliability": "medium"
            },
            {
                "id": "etsy",
                "name": "Etsy",
                "description": "Handmade and vintage toys on Etsy",
                "type": "fixed_price",
                "reliability": "low"
            }
        ]
        
        return {
            "success": True,
            "data": {
                "sources": sources,
                "count": len(sources)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting price sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to get price sources")

@router.post("/refresh-cache")
async def refresh_price_cache(
    pricing_service: PricingService = Depends(get_pricing_service)
):
    """Clear and refresh the pricing cache"""
    try:
        # Clear the cache
        pricing_service.price_cache.clear()
        
        return {
            "success": True,
            "message": "Price cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing price cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh cache")