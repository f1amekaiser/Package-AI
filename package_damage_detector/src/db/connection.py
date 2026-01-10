"""
PostgreSQL Database Connection for PackageAI

Production-ready database connection with:
- PostgreSQL as primary database
- SQLite fallback for development
- Connection pooling and retry logic
- YAML configuration support
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Generator

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import OperationalError

from .models import Base

logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def load_db_config() -> dict:
    """Load database configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('database', {})
    return {}


def build_postgresql_url(config: dict) -> str:
    """Build PostgreSQL connection URL from configuration."""
    pg_config = config.get('postgresql', {})
    
    host = pg_config.get('host', 'localhost')
    port = pg_config.get('port', 5432)
    database = pg_config.get('database', 'packageai_db')
    username = pg_config.get('username', 'packageai')
    password = pg_config.get('password', 'packageai')
    
    # Check for environment variable overrides
    host = os.environ.get('POSTGRES_HOST', host)
    port = int(os.environ.get('POSTGRES_PORT', port))
    database = os.environ.get('POSTGRES_DB', database)
    username = os.environ.get('POSTGRES_USER', username)
    password = os.environ.get('POSTGRES_PASSWORD', password)
    
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"


def build_sqlite_url(config: dict) -> str:
    """Build SQLite connection URL from configuration."""
    sqlite_config = config.get('sqlite', {})
    db_path = sqlite_config.get('path', 'data/packageai.db')
    
    # Make path absolute
    if not os.path.isabs(db_path):
        db_path = Path(__file__).parent.parent.parent / db_path
    else:
        db_path = Path(db_path)
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def get_pool_config(config: dict) -> dict:
    """Extract connection pool configuration."""
    pool_config = config.get('postgresql', {}).get('pool', {})
    return {
        'pool_size': pool_config.get('max_size', 10),
        'max_overflow': pool_config.get('max_overflow', 5),
        'pool_timeout': pool_config.get('pool_timeout', 30),
        'pool_recycle': pool_config.get('pool_recycle', 1800),
        'pool_pre_ping': True  # Always enable connection health checks
    }


def create_db_engine(config: dict, retry_on_fail: bool = True):
    """
    Create database engine with retry logic.
    
    Attempts PostgreSQL first, falls back to SQLite if configured.
    """
    db_type = config.get('type', 'postgresql')
    
    # Check for DATABASE_URL environment variable (highest priority)
    env_url = os.environ.get('DATABASE_URL')
    if env_url:
        logger.info("Using DATABASE_URL from environment")
        if env_url.startswith('sqlite'):
            return create_engine(
                env_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
        else:
            pool_config = get_pool_config(config)
            return create_engine(
                env_url,
                poolclass=QueuePool,
                echo=False,
                **pool_config
            )
    
    # PostgreSQL configuration
    if db_type == 'postgresql':
        pg_url = build_postgresql_url(config)
        pool_config = get_pool_config(config)
        retry_config = config.get('postgresql', {}).get('retry', {})
        max_attempts = retry_config.get('max_attempts', 3)
        delay = retry_config.get('delay_seconds', 5)
        
        for attempt in range(1, max_attempts + 1):
            try:
                engine = create_engine(
                    pg_url,
                    poolclass=QueuePool,
                    echo=False,
                    **pool_config
                )
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(f"Connected to PostgreSQL database successfully")
                return engine
                
            except OperationalError as e:
                logger.warning(f"PostgreSQL connection attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts and retry_on_fail:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    if retry_on_fail:
                        logger.warning("PostgreSQL unavailable, falling back to SQLite")
                        break
                    raise
    
    # SQLite fallback
    sqlite_url = build_sqlite_url(config)
    logger.info(f"Using SQLite database: {sqlite_url}")
    return create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )


# Load configuration and create engine
_db_config = load_db_config()
engine = create_db_engine(_db_config)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize database - create all tables if they don't exist.
    Safe to call multiple times.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection for database sessions.
    Yields a session and ensures cleanup.
    
    Usage with FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a new database session (for non-FastAPI contexts).
    Caller is responsible for closing.
    
    Usage:
        session = get_db_session()
        try:
            # do work
            session.commit()
        finally:
            session.close()
    """
    return SessionLocal()


def check_db_connection() -> dict:
    """
    Check database connection health.
    Returns status information.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"
        return {
            "status": "healthy",
            "database_type": db_type,
            "connection": "active"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "failed"
        }


def get_database_info() -> dict:
    """
    Get database configuration information (non-sensitive).
    """
    db_url = str(engine.url)
    if "postgresql" in db_url:
        # Mask password in URL
        parts = db_url.split('@')
        if len(parts) > 1:
            db_url = parts[0].rsplit(':', 1)[0] + ':***@' + parts[1]
    
    return {
        "database_type": "postgresql" if "postgresql" in str(engine.url) else "sqlite",
        "connection_url": db_url,
        "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else 1,
        "tables_created": True
    }
