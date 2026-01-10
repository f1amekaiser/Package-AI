"""
PackageAI Database Module

Provides database connectivity, models, and utilities for
the Package Damage Detection System.

Supports PostgreSQL (production) with SQLite fallback (development).
"""

from .connection import (
    engine,
    SessionLocal,
    init_db,
    get_db,
    get_db_session,
    check_db_connection,
    get_database_info
)

from .models import (
    Base,
    InspectionHistory,
    Detection,
    AuditLog,
    EvidenceMetadata
)

__all__ = [
    # Connection
    'engine',
    'SessionLocal',
    'init_db',
    'get_db',
    'get_db_session',
    'check_db_connection',
    'get_database_info',
    # Models
    'Base',
    'InspectionHistory',
    'Detection',
    'AuditLog',
    'EvidenceMetadata'
]
