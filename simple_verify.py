#!/usr/bin/env python3
"""
Simple PostgreSQL Verification Script
"""

import asyncio
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_postgresql_simple():
    """Simple PostgreSQL verification using psycopg2"""

    try:
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL not found in environment variables")
            return False

        print(f"📡 Connecting to database: {database_url.replace(database_url.split('@')[0], '***')}")

        # Parse the URL for psycopg2
        # postgresql://user:password@host:port/database
        if database_url.startswith('postgresql://'):
            url_parts = database_url.replace('postgresql://', '').split('@')
            if len(url_parts) == 2:
                user_pass = url_parts[0].split(':')
                host_db = url_parts[1].split('/')

                if len(user_pass) == 2 and len(host_db) == 2:
                    user = user_pass[0]
                    password = user_pass[1]
                    host_port = host_db[0].split(':')
                    host = host_port[0]
                    port = int(host_port[1]) if len(host_port) > 1 else 5432
                    database = host_db[1]

                    # Connect to PostgreSQL
                    conn = psycopg2.connect(
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password
                    )

                    cursor = conn.cursor()

                    # Test connection
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    print(f"✅ Connected to PostgreSQL: {version}")

                    # Check database name
                    cursor.execute("SELECT current_database()")
                    db_name = cursor.fetchone()[0]
                    print(f"📊 Current database: {db_name}")

                    # Check tables
                    required_tables = ['toy_analyses', 'price_history', 'user_sessions', 'ml_model_metrics']

                    print("🔍 Checking table existence...")

                    for table_name in required_tables:
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_schema = 'public'
                                AND table_name = %s
                            )
                        """, (table_name,))

                        exists = cursor.fetchone()[0]

                        if exists:
                            print(f"✅ Table '{table_name}' exists")

                            # Get row count
                            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                            count = cursor.fetchone()[0]
                            print(f"   📊 Row count: {count}")

                            # Get columns
                            cursor.execute("""
                                SELECT column_name, data_type, is_nullable
                                FROM information_schema.columns
                                WHERE table_schema = 'public'
                                AND table_name = %s
                                ORDER BY ordinal_position
                            """, (table_name,))

                            columns = cursor.fetchall()
                            col_names = [col[0] for col in columns]
                            print(f"   📋 Columns: {col_names}")

                        else:
                            print(f"❌ Table '{table_name}' does NOT exist")
                            return False

                    # Check indexes
                    print("🔍 Checking database indexes...")
                    cursor.execute("""
                        SELECT indexname, tablename
                        FROM pg_indexes
                        WHERE schemaname = 'public'
                        ORDER BY tablename, indexname
                    """)

                    indexes = cursor.fetchall()
                    print(f"📊 Database indexes: {[(idx[0], idx[1]) for idx in indexes]}")

                    conn.close()

                    print("🎉 PostgreSQL verification completed successfully!")
                    print("✅ All tables exist and are accessible")
                    print("✅ Database connection is working")

                    return True

    except Exception as e:
        print(f"❌ PostgreSQL verification failed: {e}")
        return False

    print("❌ Failed to parse database URL")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TOY WIZARD BACKEND - SIMPLE POSTGRESQL VERIFICATION")
    print("=" * 60)

    success = verify_postgresql_simple()

    print("=" * 60)
    if success:
        print("🎯 VERIFICATION STATUS: SUCCESS ✅")
        print("PostgreSQL setup is complete and ready for production!")
    else:
        print("🎯 VERIFICATION STATUS: FAILED ❌")
        print("Please check PostgreSQL configuration and try again.")
    print("=" * 60)

    exit(0 if success else 1)