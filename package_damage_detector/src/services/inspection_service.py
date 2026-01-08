"""
Inspection Service Module

Orchestrates the complete package inspection workflow:
1. Capture images from all cameras
2. Run inference on each image
3. Fuse detections and make decision
4. Store evidence
5. Return result
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.inference_engine import InferenceEngine, InferenceResult
from ..core.decision_engine import DecisionEngine, Decision, DecisionType
from ..core.evidence_manager import EvidenceManager, EvidenceRecord
from .camera_manager import CameraManager, CaptureFrame

logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Complete result of a package inspection."""
    inspection_id: str
    package_id: str
    timestamp: datetime
    
    # Decision
    decision: Decision
    
    # Evidence
    evidence_record: Optional[EvidenceRecord] = None
    
    # Timing
    capture_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    decision_time_ms: float = 0.0
    evidence_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Per-camera results
    inference_results: List[InferenceResult] = field(default_factory=list)
    
    @property
    def is_accept(self) -> bool:
        return self.decision.final_decision == DecisionType.ACCEPT
    
    @property
    def is_reject(self) -> bool:
        return self.decision.final_decision == DecisionType.REJECT
    
    @property
    def needs_review(self) -> bool:
        return self.decision.decision_type == DecisionType.REVIEW_REQUIRED and not self.decision.is_reviewed


class InspectionService:
    """
    Main inspection service that orchestrates the complete workflow.
    
    Coordinates:
    - Camera capture
    - AI inference
    - Decision making
    - Evidence storage
    """
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        decision_engine: DecisionEngine,
        evidence_manager: EvidenceManager,
        camera_manager: CameraManager
    ):
        """
        Initialize the inspection service.
        
        Args:
            inference_engine: Configured InferenceEngine
            decision_engine: Configured DecisionEngine
            evidence_manager: Configured EvidenceManager
            camera_manager: Configured CameraManager
        """
        self.inference_engine = inference_engine
        self.decision_engine = decision_engine
        self.evidence_manager = evidence_manager
        self.camera_manager = camera_manager
        
        self._inspection_count = 0
        
        logger.info("Inspection service initialized")
    
    def inspect_package(
        self,
        package_id: str,
        timeout_seconds: float = 5.0
    ) -> InspectionResult:
        """
        Perform a complete package inspection.
        
        Args:
            package_id: Unique identifier for the package
            timeout_seconds: Maximum time for the entire inspection
            
        Returns:
            InspectionResult with decision and evidence
        """
        start_time = time.perf_counter()
        timestamp = datetime.utcnow()
        
        self._inspection_count += 1
        inspection_id = f"INS-{timestamp.strftime('%Y%m%d-%H%M%S')}-{self._inspection_count:04d}"
        
        logger.info(f"Starting inspection {inspection_id} for package {package_id}")
        
        # Step 1: Capture from all cameras
        t0 = time.perf_counter()
        captures = self.camera_manager.capture_all()
        capture_time = (time.perf_counter() - t0) * 1000
        
        if not captures:
            logger.error("No images captured from cameras")
            return self._create_error_result(
                inspection_id, package_id, timestamp,
                "No images captured", capture_time
            )
        
        # Extract raw images
        images = {
            cam_id: capture.frame
            for cam_id, capture in captures.items()
        }
        
        # Step 2: Run inference on all images
        t1 = time.perf_counter()
        inference_results = []
        annotated_images = {}
        
        for cam_id, image in images.items():
            result = self.inference_engine.infer(image, camera_id=cam_id)
            inference_results.append(result)
            
            # Create annotated image
            if result.detections:
                annotated = self.inference_engine.annotate_image(
                    image, result.detections
                )
                annotated_images[cam_id] = annotated
            else:
                annotated_images[cam_id] = image.copy()
        
        inference_time = (time.perf_counter() - t1) * 1000
        
        # Step 3: Make decision
        t2 = time.perf_counter()
        decision = self.decision_engine.make_decision(
            inference_results, package_id
        )
        decision_time = (time.perf_counter() - t2) * 1000
        
        # Step 4: Store evidence
        t3 = time.perf_counter()
        evidence_record = self.evidence_manager.store_evidence(
            package_id=package_id,
            images=images,
            inference_results=inference_results,
            decision=decision,
            annotated_images=annotated_images
        )
        evidence_time = (time.perf_counter() - t3) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Inspection {inspection_id} complete: "
            f"{decision.decision_type.name} in {total_time:.1f}ms"
        )
        
        return InspectionResult(
            inspection_id=inspection_id,
            package_id=package_id,
            timestamp=timestamp,
            decision=decision,
            evidence_record=evidence_record,
            capture_time_ms=capture_time,
            inference_time_ms=inference_time,
            decision_time_ms=decision_time,
            evidence_time_ms=evidence_time,
            total_time_ms=total_time,
            inference_results=inference_results
        )
    
    def inspect_with_images(
        self,
        package_id: str,
        images: Dict[str, np.ndarray]
    ) -> InspectionResult:
        """
        Perform inspection with pre-captured images.
        
        Useful for testing or when images are captured externally.
        
        Args:
            package_id: Unique identifier for the package
            images: Dictionary mapping camera_id to image
            
        Returns:
            InspectionResult with decision and evidence
        """
        start_time = time.perf_counter()
        timestamp = datetime.utcnow()
        
        self._inspection_count += 1
        inspection_id = f"INS-{timestamp.strftime('%Y%m%d-%H%M%S')}-{self._inspection_count:04d}"
        
        # Run inference
        t1 = time.perf_counter()
        inference_results = []
        annotated_images = {}
        
        for cam_id, image in images.items():
            result = self.inference_engine.infer(image, camera_id=cam_id)
            inference_results.append(result)
            
            if result.detections:
                annotated = self.inference_engine.annotate_image(
                    image, result.detections
                )
                annotated_images[cam_id] = annotated
            else:
                annotated_images[cam_id] = image.copy()
        
        inference_time = (time.perf_counter() - t1) * 1000
        
        # Make decision
        t2 = time.perf_counter()
        decision = self.decision_engine.make_decision(
            inference_results, package_id
        )
        decision_time = (time.perf_counter() - t2) * 1000
        
        # Store evidence
        t3 = time.perf_counter()
        evidence_record = self.evidence_manager.store_evidence(
            package_id=package_id,
            images=images,
            inference_results=inference_results,
            decision=decision,
            annotated_images=annotated_images
        )
        evidence_time = (time.perf_counter() - t3) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return InspectionResult(
            inspection_id=inspection_id,
            package_id=package_id,
            timestamp=timestamp,
            decision=decision,
            evidence_record=evidence_record,
            capture_time_ms=0.0,
            inference_time_ms=inference_time,
            decision_time_ms=decision_time,
            evidence_time_ms=evidence_time,
            total_time_ms=total_time,
            inference_results=inference_results
        )
    
    def apply_operator_decision(
        self,
        result: InspectionResult,
        operator_decision: DecisionType,
        operator_id: str,
        notes: str = ""
    ) -> InspectionResult:
        """
        Apply operator override to an inspection result.
        
        Args:
            result: Original inspection result
            operator_decision: Operator's decision
            operator_id: Operator identifier
            notes: Optional notes
            
        Returns:
            Updated inspection result
        """
        # Update decision
        self.decision_engine.apply_operator_decision(
            result.decision,
            operator_decision,
            operator_id,
            notes
        )
        
        # Update evidence record if exists
        if result.evidence_record:
            # Note: In production, you'd update the stored record
            result.evidence_record.decision.final_decision = operator_decision.name
            result.evidence_record.decision.decided_by = operator_id
            result.evidence_record.decision.notes = notes
        
        logger.info(
            f"Operator {operator_id} set decision for {result.package_id} "
            f"to {operator_decision.name}"
        )
        
        return result
    
    def _create_error_result(
        self,
        inspection_id: str,
        package_id: str,
        timestamp: datetime,
        error_message: str,
        elapsed_ms: float
    ) -> InspectionResult:
        """Create an error result when inspection fails."""
        decision = Decision(
            decision_type=DecisionType.REVIEW_REQUIRED,
            package_id=package_id,
            timestamp=timestamp,
            rationale=f"Inspection error: {error_message}"
        )
        
        return InspectionResult(
            inspection_id=inspection_id,
            package_id=package_id,
            timestamp=timestamp,
            decision=decision,
            total_time_ms=elapsed_ms
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "inspection_count": self._inspection_count,
            "camera_status": self.camera_manager.get_camera_status()
        }


def create_inspection_service(
    config: Dict[str, Any],
    simulated_cameras: bool = False
) -> InspectionService:
    """
    Factory function to create an InspectionService from config.
    
    Args:
        config: Configuration dictionary
        simulated_cameras: Use simulated cameras for testing
        
    Returns:
        Configured InspectionService
    """
    from ..core.inference_engine import create_engine
    from ..core.decision_engine import create_decision_engine
    from ..core.evidence_manager import create_evidence_manager
    from .camera_manager import create_camera_manager
    
    # Create components
    inference_engine = create_engine(config)
    decision_engine = create_decision_engine(config)
    evidence_manager = create_evidence_manager(config)
    camera_manager = create_camera_manager(config, simulated=simulated_cameras)
    
    # Initialize cameras
    camera_manager.initialize()
    
    return InspectionService(
        inference_engine=inference_engine,
        decision_engine=decision_engine,
        evidence_manager=evidence_manager,
        camera_manager=camera_manager
    )
