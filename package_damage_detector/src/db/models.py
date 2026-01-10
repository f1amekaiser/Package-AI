"""
SQLAlchemy Models for PackageAI Database

Defines persistent storage for inspection history, detections, audit logs, and evidence.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def generate_uuid():
    """Generate a UUID string compatible with both PostgreSQL and SQLite."""
    return str(uuid.uuid4())


class InspectionHistory(Base):
    """
    Stores complete inspection records for each package analyzed.
    """
    __tablename__ = 'inspection_history'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    inspection_id = Column(String(64), unique=True, nullable=False, index=True)
    package_id = Column(String(64), nullable=False, index=True)
    decision = Column(String(32), nullable=False)  # ACCEPT / REJECT / REVIEW_REQUIRED
    severity_score = Column(Integer, default=0, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    source = Column(String(32), default='AI', nullable=False)
    rationale = Column(Text, nullable=True)
    detections_count = Column(Integer, default=0, nullable=False)
    inference_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="inspection", cascade="all, delete-orphan")
    evidence = relationship("EvidenceMetadata", back_populates="inspection", uselist=False)

    def to_dict(self):
        return {
            'id': self.id,
            'inspection_id': self.inspection_id,
            'package_id': self.package_id,
            'decision': self.decision,
            'severity_score': self.severity_score,
            'confidence': self.confidence,
            'source': self.source,
            'rationale': self.rationale,
            'detections_count': self.detections_count,
            'inference_time_ms': self.inference_time_ms,
            'created_at': self.created_at.isoformat() + 'Z' if self.created_at else None
        }


class Detection(Base):
    """
    Stores individual detection results linked to an inspection.
    """
    __tablename__ = 'detections'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    inspection_id = Column(String(64), ForeignKey('inspection_history.inspection_id'), nullable=False, index=True)
    class_name = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    severity_level = Column(String(32), nullable=True)
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    inspection = relationship("InspectionHistory", back_populates="detections")

    def to_dict(self):
        return {
            'id': self.id,
            'inspection_id': self.inspection_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'severity_level': self.severity_level,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]
        }


class AuditLog(Base):
    """
    Immutable audit trail for all system events.
    Supports INSPECTED, DECISION_MADE, REVIEW_OVERRIDE actions.
    """
    __tablename__ = 'audit_logs'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    inspection_id = Column(String(64), nullable=True, index=True)
    package_id = Column(String(64), nullable=False, index=True)
    action = Column(String(32), nullable=False)  # INSPECTED / DECISION_MADE / REVIEW_OVERRIDE
    decision = Column(String(32), nullable=False)
    severity = Column(Integer, default=0, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    source = Column(String(32), default='AI', nullable=False)  # AI / Manual / System
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self):
        return {
            'id': self.id,
            'inspection_id': self.inspection_id,
            'package_id': self.package_id,
            'action': self.action,
            'decision': self.decision,
            'severity': self.severity,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.created_at.isoformat() + 'Z' if self.created_at else None
        }


class EvidenceMetadata(Base):
    """
    Stores evidence hashes for integrity verification.
    Immutable after creation.
    """
    __tablename__ = 'evidence_metadata'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    inspection_id = Column(String(64), ForeignKey('inspection_history.inspection_id'), unique=True, nullable=False, index=True)
    image_hash = Column(String(64), nullable=False)  # SHA-256
    detection_hash = Column(String(64), nullable=False)
    decision_hash = Column(String(64), nullable=False)
    record_hash = Column(String(64), nullable=False)
    storage_path = Column(String(512), nullable=True)
    verified = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    inspection = relationship("InspectionHistory", back_populates="evidence")

    def to_dict(self):
        return {
            'id': self.id,
            'inspection_id': self.inspection_id,
            'image_hash': self.image_hash,
            'detection_hash': self.detection_hash,
            'decision_hash': self.decision_hash,
            'record_hash': self.record_hash,
            'storage_path': self.storage_path,
            'verified': self.verified,
            'created_at': self.created_at.isoformat() + 'Z' if self.created_at else None
        }

