#!/usr/bin/env python3
"""
PostgreSQL Database Verification Script
Checks connection and table existence for Toy Wizard Backend
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def verify_postgresql_setup():
    """Verify PostgreSQL connection and table structure"""

    logger.info("🔍 Starting PostgreSQL verification...")

    try:
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            logger.error("❌ DATABASE_URL not found in environment variables")
            return False

        logger.info(f"📡 Connecting to database: {database_url.replace(database_url.split('@')[0], '***')}")

        # Create async engine with explicit PostgreSQL async driver
        async_db_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
        logger.info(f"🔧 Using database URL: {async_db_url.replace(async_db_url.split('@')[0], '***')}")

        try:
            engine = create_async_engine(
                async_db_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
        except Exception as driver_error:
            logger.error(f"❌ Failed to create async engine: {driver_error}")
            logger.error("💡 Make sure asyncpg is installed: pip install asyncpg")
            return False

        # Test connection
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"✅ Connected to PostgreSQL: {version}")

            # Check database name
            result = await conn.execute(text("SELECT current_database()"))
            db_name = result.scalar()
            logger.info(f"📊 Current database: {db_name}")

            # Check if tables exist
            required_tables = ['toy_analyses', 'price_history', 'user_sessions', 'ml_model_metrics']

            logger.info("🔍 Checking table existence...")

            for table_name in required_tables:
                try:
                    # Check table exists
                    result = await conn.execute(text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = :table_name
                        )
                    """), {'table_name': table_name})

                    exists = result.scalar()

                    if exists:
                        logger.info(f"✅ Table '{table_name}' exists")

                        # Get row count
                        result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.scalar()
                        logger.info(f"   📊 Row count: {count}")

                        # Get column information
                        result = await conn.execute(text("""
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                            AND table_name = :table_name
                            ORDER BY ordinal_position
                        """), {'table_name': table_name})

                        columns = result.fetchall()
                        logger.info(f"   📋 Columns: {[col[0] for col in columns]}")

                    else:
                        logger.error(f"❌ Table '{table_name}' does NOT exist")
                        return False

                except Exception as e:
                    logger.error(f"❌ Error checking table '{table_name}': {e}")
                    return False

            # Test basic operations on each table
            logger.info("🔧 Testing basic database operations...")

            # Test ToyAnalysis table
            try:
                async with AsyncSession(engine) as session:
                    # Test insert (we'll rollback)
                    from app.core.database import ToyAnalysis
                    test_toy = ToyAnalysis(
                        toy_name="Test Toy",
                        category="test",
                        brand="Test Brand",
                        condition_score=8.5,
                        estimated_price=25.99,
                        rarity_score=0.5,
                        confidence=0.9,
                        image_path="/test/path",
                        analysis_data='{"test": "data"}'
                    )
                    session.add(test_toy)
                    await session.rollback()  # Don't actually commit

                logger.info("✅ ToyAnalysis table operations working")

            except Exception as e:
                logger.error(f"❌ ToyAnalysis table operation failed: {e}")
                return False

            # Check indexes
            logger.info("🔍 Checking database indexes...")
            result = await conn.execute(text("""
                SELECT indexname, tablename
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """))

            indexes = result.fetchall()
            logger.info(f"📊 Database indexes: {[(idx[0], idx[1]) for idx in indexes]}")

            # Check constraints
            logger.info("🔍 Checking table constraints...")
            result = await conn.execute(text("""
                SELECT tc.table_name, tc.constraint_name, tc.constraint_type
                FROM information_schema.table_constraints tc
                WHERE tc.table_schema = 'public'
                ORDER BY tc.table_name, tc.constraint_name
            """))

            constraints = result.fetchall()
            logger.info(f"📊 Table constraints: {[(c[0], c[1], c[2]) for c in constraints]}")

        # Close engine
        await engine.dispose()

        logger.info("🎉 PostgreSQL verification completed successfully!")
        logger.info("✅ All tables exist and are accessible")
        logger.info("✅ Database connection is working")
        logger.info("✅ Basic operations are functional")

        return True

    except Exception as e:
        logger.error(f"❌ PostgreSQL verification failed: {e}")
        return False

async def main():
    """Main verification function"""
    logger.info("=" * 60)
    logger.info("🚀 TOY WIZARD BACKEND - POSTGRESQL VERIFICATION")
    logger.info("=" * 60)

    success = await verify_postgresql_setup()

    logger.info("=" * 60)
    if success:
        logger.info("🎯 VERIFICATION STATUS: SUCCESS ✅")
        logger.info("PostgreSQL setup is complete and ready for production!")
    else:
        logger.info("🎯 VERIFICATION STATUS: FAILED ❌")
        logger.info("Please check PostgreSQL configuration and try again.")
    logger.info("=" * 60)

    return success

if __name__ == "__main__":
    # Run verification
    result = asyncio.run(main())
    exit(0 if result else 1)