"""
PostgreSQL Database Connection for PackageAI

Provides SQLAlchemy engine, session factory, and dependency injection.
Supports PostgreSQL (production) with SQLite fallback (development).
"""

import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from .models import Base

logger = logging.getLogger(__name__)

# Database URL from environment or default (SQLite for easy local dev)
# For production: postgresql://packageai:packageai@localhost:5432/packageai_db
DATABASE_URL = os.environ.get('DATABASE_URL', None)

if DATABASE_URL is None:
    # Default to SQLite for development
    db_path = Path(__file__).parent.parent.parent / "data" / "packageai.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = f"sqlite:///{db_path}"
    logger.info(f"Using SQLite database at: {db_path}")

# Configure engine based on database type
if DATABASE_URL.startswith('sqlite'):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
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


def get_db():
    """
    Dependency injection for database sessions.
    Yields a session and ensures cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """
    Get a new database session (for non-FastAPI contexts).
    Caller is responsible for closing.
    """
    return SessionLocal()
