#!/usr/bin/env python3
"""
PostgreSQL Database Verification Script
Checks connection and table existence for Toy Wizard Backend
"""

import asyncio
import logging
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
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

# Convert sync DATABASE_URL to async
async_db_url = settings.DATABASE_URL
if async_db_url.startswith("sqlite"):
    async_db_url = async_db_url.replace("sqlite://", "sqlite+aiosqlite://")
elif async_db_url.startswith("postgresql"):
    async_db_url = async_db_url.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
if "sqlite" in async_db_url:
    engine = create_async_engine(
        async_db_url,
        echo=False,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_async_engine(
        async_db_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600
    )

# Tables to check
EXPECTED_TABLES = [
    "toy_analyses",
    "price_history",
    "user_sessions",
    "ml_model_metrics"
]

async def check_connection():
    """Check database connection"""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        return False

async def check_tables():
    """Check if expected tables exist"""
    try:
        async with engine.begin() as conn:
            def get_table_names(conn):
                return inspect(conn).get_table_names()
            
            existing_tables = await conn.run_sync(get_table_names)
            
            logger.info(f"Found {len(existing_tables)} tables in database")
            
            missing_tables = []
            for table in EXPECTED_TABLES:
                if table in existing_tables:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.error(f"❌ Table '{table}' is missing")
                    missing_tables.append(table)
            
            if not missing_tables:
                logger.info("✅ All expected tables are present")
            else:
                logger.warning(f"⚠️  Missing tables: {', '.join(missing_tables)}")
            
            return len(missing_tables) == 0
            
    except Exception as e:
        logger.error(f"❌ Failed to check tables: {str(e)}")
        return False

async def main():
    """Main verification function"""
    logger.info("🔍 Starting PostgreSQL database verification...")
    
    connection_ok = await check_connection()
    tables_ok = False
    
    if connection_ok:
        tables_ok = await check_tables()
    
    if connection_ok and tables_ok:
        logger.info("🎉 PostgreSQL setup verification completed successfully!")
        return 0
    else:
        logger.error("💥 PostgreSQL setup verification failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)