"""
Evidence Manager Module

Handles tamper-proof storage of inspection evidence including:
- Timestamped images from all cameras
- Detection results and decisions
- Cryptographic hash chain for integrity verification
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import shutil

import cv2
import numpy as np

from .decision_engine import Decision, DecisionType, Severity, ScoredDetection, Detection, InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class CaptureRecord:
    """Record of a single camera capture."""
    camera_id: str
    image_path: str
    image_hash: str
    resolution: tuple
    capture_time: str


@dataclass
class DetectionRecord:
    """Serializable detection record."""
    camera_id: str
    class_name: str
    class_id: int
    confidence: float
    bbox_normalized: tuple
    bbox_pixels: tuple
    severity_score: float
    severity_level: str


@dataclass
class DecisionRecord:
    """Serializable decision record."""
    automated_decision: str
    final_decision: str
    decided_by: str
    decision_timestamp: str
    notes: str


@dataclass
class EvidenceRecord:
    """
    Complete evidence record for a package inspection.
    """
    inspection_id: str
    package_id: str
    station_id: str
    timestamp_utc: str
    timestamp_local: str
    
    captures: List[CaptureRecord] = field(default_factory=list)
    detections: List[DetectionRecord] = field(default_factory=list)
    decision: Optional[DecisionRecord] = None
    
    # Integrity
    content_hash: str = ""
    previous_hash: str = ""
    record_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "inspection_id": self.inspection_id,
            "package_id": self.package_id,
            "station_id": self.station_id,
            "timestamp_utc": self.timestamp_utc,
            "timestamp_local": self.timestamp_local,
            "captures": [asdict(c) for c in self.captures],
            "detections": [asdict(d) for d in self.detections],
            "decision": asdict(self.decision) if self.decision else None,
            "integrity": {
                "content_hash": self.content_hash,
                "previous_hash": self.previous_hash,
                "record_hash": self.record_hash,
            }
        }


class EvidenceManager:
    """
    Manages tamper-proof evidence storage for package inspections.
    
    Features:
    - Stores images from all cameras
    - Records all detections and decisions
    - Maintains cryptographic hash chain
    - Supports local and remote storage
    """
    
    def __init__(
        self,
        storage_path: str = "evidence",
        station_id: str = "STATION-01",
        image_quality: int = 95,
        save_annotated: bool = True,
        save_composite: bool = True,
        enable_hash_chain: bool = True,
        hash_algorithm: str = "sha256"
    ):
        """
        Initialize the evidence manager.
        
        Args:
            storage_path: Base path for evidence storage
            station_id: Identifier for this inspection station
            image_quality: JPEG quality (1-100)
            save_annotated: Save images with detection annotations
            save_composite: Save composite image of all cameras
            enable_hash_chain: Enable hash chain for tamper detection
            hash_algorithm: Hash algorithm to use
        """
        self.storage_path = Path(storage_path)
        self.station_id = station_id
        self.image_quality = image_quality
        self.save_annotated = save_annotated
        self.save_composite = save_composite
        self.enable_hash_chain = enable_hash_chain
        self.hash_algorithm = hash_algorithm
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Track last record hash for chain
        self._last_record_hash = self._load_last_hash()
        
        logger.info(f"Evidence manager initialized at {self.storage_path}")
    
    def _load_last_hash(self) -> str:
        """Load the last record hash from chain file."""
        chain_file = self.storage_path / "chain_state.json"
        if chain_file.exists():
            try:
                with open(chain_file) as f:
                    state = json.load(f)
                    return state.get("last_hash", "")
            except Exception as e:
                logger.error(f"Failed to load chain state: {e}")
        return ""
    
    def _save_chain_state(self):
        """Save the current chain state."""
        chain_file = self.storage_path / "chain_state.json"
        try:
            with open(chain_file, "w") as f:
                json.dump({
                    "last_hash": self._last_record_hash,
                    "updated": datetime.utcnow().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Failed to save chain state: {e}")
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute hash of data."""
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.hash_algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()
    
    def _hash_file(self, filepath: Path) -> str:
        """Compute hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _generate_inspection_id(self, timestamp: datetime) -> str:
        """Generate unique inspection ID."""
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M%S")
        ms = timestamp.strftime("%f")[:3]
        return f"INS-{date_str}-{time_str}{ms}"
    
    def _get_evidence_dir(self, timestamp: datetime, inspection_id: str) -> Path:
        """Get evidence directory path for an inspection."""
        date_path = timestamp.strftime("%Y/%m/%d")
        return self.storage_path / date_path / inspection_id
    
    def store_evidence(
        self,
        package_id: str,
        images: Dict[str, np.ndarray],
        inference_results: List[InferenceResult],
        decision: Decision,
        annotated_images: Optional[Dict[str, np.ndarray]] = None
    ) -> EvidenceRecord:
        """
        Store complete evidence for a package inspection.
        
        Args:
            package_id: Unique package identifier
            images: Dict mapping camera_id to raw image
            inference_results: Results from all cameras
            decision: Final decision
            annotated_images: Optional dict of annotated images
            
        Returns:
            EvidenceRecord with all stored information
        """
        timestamp = datetime.utcnow()
        inspection_id = self._generate_inspection_id(timestamp)
        
        # Create evidence directory
        evidence_dir = self._get_evidence_dir(timestamp, inspection_id)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Store images and build capture records
        captures = []
        for camera_id, image in images.items():
            capture = self._store_image(
                evidence_dir, camera_id, image, "raw"
            )
            captures.append(capture)
            
            # Store annotated version if available
            if self.save_annotated and annotated_images and camera_id in annotated_images:
                self._store_image(
                    evidence_dir, camera_id, annotated_images[camera_id], "annotated"
                )
        
        # Store composite image
        if self.save_composite and len(images) > 0:
            composite = self._create_composite(images)
            self._store_image(evidence_dir, "composite", composite, "composite")
        
        # Build detection records
        detection_records = []
        for scored in decision.detections:
            det = scored.detection
            detection_records.append(DetectionRecord(
                camera_id=det.camera_id,
                class_name=det.class_name,
                class_id=det.class_id,
                confidence=det.confidence,
                bbox_normalized=det.bbox,
                bbox_pixels=det.bbox_pixels,
                severity_score=scored.severity_score,
                severity_level=scored.severity_level.name
            ))
        
        # Build decision record
        decision_record = DecisionRecord(
            automated_decision=decision.decision_type.name,
            final_decision=decision.final_decision.name,
            decided_by=decision.operator_id or "SYSTEM",
            decision_timestamp=decision.timestamp.isoformat() if decision.timestamp else timestamp.isoformat(),
            notes=decision.operator_notes or ""
        )
        
        # Create evidence record
        record = EvidenceRecord(
            inspection_id=inspection_id,
            package_id=package_id,
            station_id=self.station_id,
            timestamp_utc=timestamp.isoformat(),
            timestamp_local=datetime.now().isoformat(),
            captures=captures,
            detections=detection_records,
            decision=decision_record
        )
        
        # Compute integrity hashes
        record = self._compute_integrity(record)
        
        # Save record to JSON
        record_path = evidence_dir / "record.json"
        with open(record_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        
        logger.info(f"Evidence stored: {inspection_id} for package {package_id}")
        
        return record
    
    def _store_image(
        self,
        evidence_dir: Path,
        camera_id: str,
        image: np.ndarray,
        suffix: str
    ) -> CaptureRecord:
        """Store a single image and return capture record."""
        filename = f"{camera_id}_{suffix}.jpg"
        filepath = evidence_dir / filename
        
        # Save image
        cv2.imwrite(
            str(filepath),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
        )
        
        # Compute hash
        image_hash = self._hash_file(filepath)
        
        return CaptureRecord(
            camera_id=camera_id,
            image_path=str(filepath.relative_to(self.storage_path)),
            image_hash=image_hash,
            resolution=(image.shape[1], image.shape[0]),
            capture_time=datetime.utcnow().isoformat()
        )
    
    def _create_composite(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """Create a composite image from all camera views."""
        # Sort by camera ID for consistent ordering
        sorted_ids = sorted(images.keys())
        sorted_images = [images[cid] for cid in sorted_ids]
        
        # Resize all to same height
        target_height = 480
        resized = []
        for img in sorted_images:
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized.append(cv2.resize(img, (new_w, target_height)))
        
        # Concatenate horizontally
        composite = np.hstack(resized)
        
        return composite
    
    def _compute_integrity(self, record: EvidenceRecord) -> EvidenceRecord:
        """Compute integrity hashes for the record."""
        if not self.enable_hash_chain:
            return record
        
        # Content hash (everything except integrity section)
        content_dict = {
            "inspection_id": record.inspection_id,
            "package_id": record.package_id,
            "station_id": record.station_id,
            "timestamp_utc": record.timestamp_utc,
            "captures": [asdict(c) for c in record.captures],
            "detections": [asdict(d) for d in record.detections],
            "decision": asdict(record.decision) if record.decision else None
        }
        content_bytes = json.dumps(content_dict, sort_keys=True).encode()
        record.content_hash = self._compute_hash(content_bytes)
        
        # Chain hash
        record.previous_hash = self._last_record_hash
        chain_input = f"{record.content_hash}{record.previous_hash}".encode()
        record.record_hash = self._compute_hash(chain_input)
        
        # Update chain state
        self._last_record_hash = record.record_hash
        self._save_chain_state()
        
        return record
    
    def verify_record(self, record: EvidenceRecord) -> bool:
        """
        Verify the integrity of an evidence record.
        
        Args:
            record: EvidenceRecord to verify
            
        Returns:
            True if record integrity is valid
        """
        if not self.enable_hash_chain:
            return True
        
        # Recompute content hash
        content_dict = {
            "inspection_id": record.inspection_id,
            "package_id": record.package_id,
            "station_id": record.station_id,
            "timestamp_utc": record.timestamp_utc,
            "captures": [asdict(c) for c in record.captures],
            "detections": [asdict(d) for d in record.detections],
            "decision": asdict(record.decision) if record.decision else None
        }
        content_bytes = json.dumps(content_dict, sort_keys=True).encode()
        computed_content_hash = self._compute_hash(content_bytes)
        
        if computed_content_hash != record.content_hash:
            logger.error(f"Content hash mismatch for {record.inspection_id}")
            return False
        
        # Verify chain hash
        chain_input = f"{record.content_hash}{record.previous_hash}".encode()
        computed_record_hash = self._compute_hash(chain_input)
        
        if computed_record_hash != record.record_hash:
            logger.error(f"Record hash mismatch for {record.inspection_id}")
            return False
        
        logger.info(f"Record {record.inspection_id} verified successfully")
        return True
    
    def load_record(self, inspection_id: str) -> Optional[EvidenceRecord]:
        """
        Load an evidence record by inspection ID.
        
        Args:
            inspection_id: Inspection ID to load
            
        Returns:
            EvidenceRecord if found, None otherwise
        """
        # Search for record file
        for record_path in self.storage_path.rglob("record.json"):
            try:
                with open(record_path) as f:
                    data = json.load(f)
                    if data.get("inspection_id") == inspection_id:
                        return self._dict_to_record(data)
            except Exception:
                continue
        
        return None
    
    def _dict_to_record(self, data: Dict[str, Any]) -> EvidenceRecord:
        """Convert dictionary to EvidenceRecord."""
        captures = [
            CaptureRecord(**c) for c in data.get("captures", [])
        ]
        detections = [
            DetectionRecord(**d) for d in data.get("detections", [])
        ]
        decision = None
        if data.get("decision"):
            decision = DecisionRecord(**data["decision"])
        
        integrity = data.get("integrity", {})
        
        return EvidenceRecord(
            inspection_id=data["inspection_id"],
            package_id=data["package_id"],
            station_id=data["station_id"],
            timestamp_utc=data["timestamp_utc"],
            timestamp_local=data.get("timestamp_local", ""),
            captures=captures,
            detections=detections,
            decision=decision,
            content_hash=integrity.get("content_hash", ""),
            previous_hash=integrity.get("previous_hash", ""),
            record_hash=integrity.get("record_hash", "")
        )
    
    def cleanup_old_records(self, retention_days: int = 14) -> int:
        """
        Remove evidence older than retention period.
        
        Args:
            retention_days: Number of days to retain evidence
            
        Returns:
            Number of records deleted
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        deleted_count = 0
        
        for year_dir in self.storage_path.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    
                    try:
                        date_str = f"{year_dir.name}-{month_dir.name}-{day_dir.name}"
                        dir_date = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        if dir_date < cutoff:
                            shutil.rmtree(day_dir)
                            deleted_count += 1
                            logger.info(f"Deleted old evidence: {day_dir}")
                    except Exception as e:
                        logger.error(f"Error processing {day_dir}: {e}")
        
        return deleted_count


def create_evidence_manager(config: Dict[str, Any]) -> EvidenceManager:
    """
    Factory function to create an EvidenceManager from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured EvidenceManager instance
    """
    evidence_config = config.get("evidence", {})
    system_config = config.get("system", {})
    
    return EvidenceManager(
        storage_path=evidence_config.get("storage_path", "evidence"),
        station_id=system_config.get("station_id", "STATION-01"),
        image_quality=evidence_config.get("image_quality", 95),
        save_annotated=evidence_config.get("save_annotated", True),
        save_composite=evidence_config.get("save_composite", True),
        enable_hash_chain=evidence_config.get("enable_hash_chain", True),
        hash_algorithm=evidence_config.get("hash_algorithm", "sha256")
    )


class TwoStageEvidenceRecorder:
    """
    Evidence recorder for two-stage inference pipeline.
    
    Creates immutable inspection records with:
    - Original image + annotated image
    - Detection list with classifier decisions
    - SHA-256 hashes for integrity verification
    - Structured folder hierarchy: evidence/YYYY/MM/DD/INSPECTION_ID/
    """
    
    MODEL_VERSIONS = {
        "detector": "best.pt",
        "classifier": "damaged_classifier_best.pt",
        "pipeline_version": "2.0.0"
    }
    
    def __init__(
        self,
        storage_path: str = "evidence",
        station_id: str = "STATION-01",
        image_quality: int = 95
    ):
        self.storage_path = Path(storage_path)
        self.station_id = station_id
        self.image_quality = image_quality
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TwoStageEvidenceRecorder initialized at {self.storage_path}")
    
    def _sha256_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data).hexdigest()
    
    def _hash_image(self, image: np.ndarray) -> str:
        """Compute SHA-256 hash of image bytes."""
        _, encoded = cv2.imencode('.jpg', image)
        return self._sha256_hash(encoded.tobytes())
    
    def _generate_inspection_id(self) -> str:
        """Generate unique inspection ID."""
        now = datetime.utcnow()
        return f"INS-{now.strftime('%Y%m%d-%H%M%S')}-{now.strftime('%f')[:4]}"
    
    def _get_evidence_dir(self, inspection_id: str) -> Path:
        """Get evidence directory path: evidence/YYYY/MM/DD/INSPECTION_ID/"""
        now = datetime.utcnow()
        return self.storage_path / now.strftime("%Y/%m/%d") / inspection_id
    
    def record_inspection(
        self,
        original_image: np.ndarray,
        annotated_image: np.ndarray,
        detections: list,
        decision: str,
        reason: str,
        package_id: str = ""
    ) -> dict:
        """
        Record an inspection with full evidence.
        
        Args:
            original_image: Original uploaded image
            annotated_image: Image with bounding boxes drawn
            detections: List of TwoStageDetection objects
            decision: "ACCEPT" | "REJECT" | "REVIEW_REQUIRED"
            reason: Human-readable decision reason
            package_id: Optional package identifier
            
        Returns:
            Inspection record dict with paths and hashes
        """
        timestamp = datetime.utcnow()
        inspection_id = self._generate_inspection_id()
        
        # Create directory
        evidence_dir = self._get_evidence_dir(inspection_id)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original image
        original_path = evidence_dir / "original.jpg"
        cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        # Save annotated image
        annotated_path = evidence_dir / "annotated.jpg"
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        # Compute hashes
        image_hash = self._hash_image(original_image)
        
        # Build detection metadata
        detection_data = []
        for det in detections:
            if hasattr(det, 'to_dict'):
                detection_data.append(det.to_dict())
            else:
                detection_data.append({
                    "bbox": det.bbox,
                    "yolo_confidence": det.yolo_confidence,
                    "classifier_label": det.classifier_label,
                    "classifier_confidence": det.classifier_confidence
                })
        
        # Hash detection metadata
        detection_hash = self._sha256_hash(
            json.dumps(detection_data, sort_keys=True).encode()
        )
        
        # Hash decision
        decision_data = {"decision": decision, "reason": reason}
        decision_hash = self._sha256_hash(
            json.dumps(decision_data, sort_keys=True).encode()
        )
        
        # Combined record hash
        record_content = f"{image_hash}{detection_hash}{decision_hash}"
        record_hash = self._sha256_hash(record_content.encode())
        
        # Build record
        record = {
            "inspection_id": inspection_id,
            "package_id": package_id or f"PKG-{inspection_id}",
            "station_id": self.station_id,
            "timestamp_utc": timestamp.isoformat() + "Z",
            "timestamp_local": datetime.now().isoformat(),
            
            "images": {
                "original": "original.jpg",
                "annotated": "annotated.jpg"
            },
            
            "detections": detection_data,
            "detection_count": len(detection_data),
            
            "decision": {
                "result": decision,
                "reason": reason
            },
            
            "integrity": {
                "image_hash_sha256": image_hash,
                "detection_hash_sha256": detection_hash,
                "decision_hash_sha256": decision_hash,
                "record_hash_sha256": record_hash,
                "algorithm": "SHA-256"
            },
            
            "model_versions": self.MODEL_VERSIONS,
            
            "immutable": True,
            "record_version": "1.0"
        }
        
        # Save JSON record
        record_path = evidence_dir / "record.json"
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        
        # Make files read-only
        self._make_readonly(evidence_dir)
        
        logger.info(f"Evidence recorded: {inspection_id} → {decision}")
        
        return record
    
    def _make_readonly(self, directory: Path):
        """Make all files in directory read-only."""
        import stat
        for filepath in directory.iterdir():
            if filepath.is_file():
                # Remove write permissions
                current = filepath.stat().st_mode
                filepath.chmod(current & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    
    def verify_record(self, inspection_id: str) -> dict:
        """
        Verify integrity of an inspection record.
        
        Returns verification result with status and any mismatches.
        """
        # Find record
        for record_path in self.storage_path.rglob("record.json"):
            try:
                with open(record_path) as f:
                    record = json.load(f)
                    if record.get("inspection_id") == inspection_id:
                        return self._verify_record_integrity(record_path.parent, record)
            except Exception:
                continue
        
        return {"status": "NOT_FOUND", "inspection_id": inspection_id}
    
    def _verify_record_integrity(self, evidence_dir: Path, record: dict) -> dict:
        """Verify all hashes in a record."""
        results = {
            "inspection_id": record["inspection_id"],
            "status": "VALID",
            "checks": {}
        }
        
        # Check image hash
        original_path = evidence_dir / "original.jpg"
        if original_path.exists():
            img = cv2.imread(str(original_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            computed_hash = self._hash_image(img_rgb)
            stored_hash = record["integrity"]["image_hash_sha256"]
            
            results["checks"]["image"] = {
                "computed": computed_hash[:16] + "...",
                "stored": stored_hash[:16] + "...",
                "match": computed_hash == stored_hash
            }
            
            if computed_hash != stored_hash:
                results["status"] = "TAMPERED"
        
        # Check detection hash
        detection_hash = self._sha256_hash(
            json.dumps(record["detections"], sort_keys=True).encode()
        )
        stored_det_hash = record["integrity"]["detection_hash_sha256"]
        
        results["checks"]["detections"] = {
            "match": detection_hash == stored_det_hash
        }
        
        if detection_hash != stored_det_hash:
            results["status"] = "TAMPERED"
        
        return results

