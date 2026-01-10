#!/usr/bin/env python3
"""
PostgreSQL Database Setup Script for PackageAI

This script:
1. Creates the PostgreSQL database if it doesn't exist
2. Creates the packageai user with appropriate permissions
3. Initializes all required tables

Usage:
    python setup_postgresql.py [--create-db] [--drop-existing]

Requirements:
    - PostgreSQL must be installed and running
    - Admin access to PostgreSQL (postgres user)
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_psql_command(command: str, as_admin: bool = True, database: str = "postgres") -> bool:
    """Execute a PostgreSQL command via psql."""
    try:
        if as_admin:
            cmd = ["psql", "-U", "postgres", "-d", database, "-c", command]
        else:
            cmd = ["psql", "-d", database, "-c", command]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        logger.error("psql command not found. Is PostgreSQL installed?")
        return False
    except Exception as e:
        logger.error(f"Error executing psql: {e}")
        return False


def check_postgres_running() -> bool:
    """Check if PostgreSQL server is running."""
    try:
        result = subprocess.run(
            ["pg_isready"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.warning("pg_isready not found, skipping check")
        return True


def create_database_and_user(drop_existing: bool = False):
    """Create the PackageAI database and user."""
    
    logger.info("Setting up PostgreSQL for PackageAI...")
    
    if not check_postgres_running():
        logger.error("PostgreSQL is not running. Please start it first.")
        logger.info("  macOS: brew services start postgresql")
        logger.info("  Linux: sudo systemctl start postgresql")
        return False
    
    # Drop existing if requested
    if drop_existing:
        logger.warning("Dropping existing database and user...")
        run_psql_command("DROP DATABASE IF EXISTS packageai_db;")
        run_psql_command("DROP USER IF EXISTS packageai;")
    
    # Create user
    logger.info("Creating user 'packageai'...")
    create_user_sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'packageai') THEN
            CREATE ROLE packageai WITH LOGIN PASSWORD 'packageai_secure_password';
        END IF;
    END
    $$;
    """
    if not run_psql_command(create_user_sql):
        logger.error("Failed to create user")
        return False
    
    # Create database
    logger.info("Creating database 'packageai_db'...")
    if not run_psql_command(
        "SELECT 'exists' FROM pg_database WHERE datname = 'packageai_db'",
        database="postgres"
    ):
        if not run_psql_command("CREATE DATABASE packageai_db OWNER packageai;"):
            logger.error("Failed to create database")
            return False
    
    # Grant privileges
    logger.info("Granting privileges...")
    run_psql_command("GRANT ALL PRIVILEGES ON DATABASE packageai_db TO packageai;")
    
    logger.info("PostgreSQL setup completed successfully!")
    return True


def init_tables():
    """Initialize database tables using SQLAlchemy models."""
    logger.info("Initializing database tables...")
    
    try:
        from src.db import init_db, check_db_connection
        
        # Check connection
        status = check_db_connection()
        if status['status'] != 'healthy':
            logger.error(f"Database connection failed: {status.get('error', 'Unknown error')}")
            return False
        
        # Initialize tables
        init_db()
        logger.info("Database tables created successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize tables: {e}")
        return False


def print_connection_info():
    """Print database connection information."""
    try:
        from src.db import get_database_info
        info = get_database_info()
        
        logger.info("\n" + "="*50)
        logger.info("DATABASE CONNECTION INFO")
        logger.info("="*50)
        logger.info(f"  Type: {info.get('database_type', 'unknown')}")
        logger.info(f"  URL: {info.get('connection_url', 'unknown')}")
        logger.info(f"  Pool Size: {info.get('pool_size', 'N/A')}")
        logger.info("="*50 + "\n")
    except Exception as e:
        logger.warning(f"Could not get connection info: {e}")


def main():
    parser = argparse.ArgumentParser(description='Setup PostgreSQL for PackageAI')
    parser.add_argument('--create-db', action='store_true', 
                        help='Create database and user (requires admin access)')
    parser.add_argument('--drop-existing', action='store_true',
                        help='Drop existing database before creating')
    parser.add_argument('--init-tables', action='store_true',
                        help='Initialize database tables only')
    parser.add_argument('--info', action='store_true',
                        help='Show database connection info')
    
    args = parser.parse_args()
    
    # Default to showing info if no args
    if not any([args.create_db, args.init_tables, args.info]):
        args.info = True
        args.init_tables = True
    
    success = True
    
    if args.create_db:
        success = create_database_and_user(args.drop_existing) and success
    
    if args.init_tables:
        success = init_tables() and success
    
    if args.info:
        print_connection_info()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
