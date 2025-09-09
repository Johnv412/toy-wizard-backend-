"""
Database configuration and models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.core.config import settings

# Convert sync DATABASE_URL to async
async_db_url = settings.DATABASE_URL
if async_db_url.startswith("sqlite"):
    async_db_url = async_db_url.replace("sqlite://", "sqlite+aiosqlite://")
elif async_db_url.startswith("postgresql"):
    async_db_url = async_db_url.replace("postgresql://", "postgresql+asyncpg://")

# Async database setup with connection pooling
if "sqlite" in async_db_url:
    engine = create_async_engine(
        async_db_url,
        echo=settings.DEBUG,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_async_engine(
        async_db_url,
        echo=settings.DEBUG,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600
    )

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Create base class
Base = declarative_base()

# Database models
class ToyAnalysis(Base):
    __tablename__ = "toy_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    toy_name = Column(String, index=True)
    category = Column(String, index=True)
    brand = Column(String, nullable=True)
    condition_score = Column(Float)
    estimated_price = Column(Float)
    rarity_score = Column(Float)
    confidence = Column(Float)
    image_path = Column(String)
    analysis_data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PriceHistory(Base):
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, index=True)
    toy_name = Column(String, index=True)
    source = Column(String)  # 'ebay', 'amazon', 'mercari', etc.
    price = Column(Float)
    condition = Column(String)
    sold_date = Column(DateTime)
    listing_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    analyses_count = Column(Integer, default=0)
    total_value = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)

class MLModelMetrics(Base):
    __tablename__ = "ml_model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

# Dependency to get async database session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()