"""
Decision Engine Module

Implements severity calculation and accept/reject/review decision logic
based on detection results from multiple cameras.
"""

import logging
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detection result (simplified for decision engine)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)
    bbox_pixels: Tuple[int, int, int, int] = (0, 0, 0, 0)
    camera_id: str = ""
    
    @property
    def area_normalized(self) -> float:
        """Calculate normalized area of detection box."""
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        return width * height


@dataclass
class InferenceResult:
    """Container for inference results from a single image."""
    image_path: str
    camera_id: str
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    
    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


def compute_severity(detections: List[Any]) -> Dict[str, Any]:
    """
    Calculate definitive severity based on strict confirmed damage rules.
    Backend-authoritative logic.
    
    Severity bands (aligned with decision thresholds):
    - 0-15: ACCEPT (no damage or all intact)
    - 16-49: REVIEW_REQUIRED (borderline damage, conf 0.50-0.84)
    - 50-100: REJECT (confirmed damage, conf >= 0.85)
    """
    # Step 1: Filter confirmed damages
    confirmed_damages = []
    
    for det in detections:
        # Handle different object types
        label = getattr(det, "classifier_label", getattr(det, "class_name", "")).lower()
        if isinstance(det, dict):
            label = det.get("class_name", "").lower()
            
        if label == "damaged":
            confirmed_damages.append(det)
            
    # Step 2: Severity logic based on decision thresholds
    if not confirmed_damages:
        # No damage detected - ACCEPT range (0-15)
        return {
            "severity_score": 0,
            "severity_label": "SAFE",
            "risk_level": "NONE"
        }
    
    # Get max confidence of CONFIRMED damages only
    max_conf = 0.0
    for det in confirmed_damages:
        if hasattr(det, "classifier_confidence"):
            conf = det.classifier_confidence
        elif hasattr(det, "confidence"):
            conf = det.confidence
        elif isinstance(det, dict):
            conf = det.get("confidence", 0.0)
        else:
            conf = 0.0
        max_conf = max(max_conf, conf)
    
    # Align severity with decision thresholds:
    # - conf >= 0.85 (REJECT) -> severity 50-100
    # - conf >= 0.50 (REVIEW) -> severity 16-49
    # - conf < 0.50 (ACCEPT despite damage) -> severity 1-15
    
    if max_conf >= 0.85:
        # REJECT range: 50-100
        # Map conf 0.85-1.0 to severity 50-100
        severity_score = int(50 + (max_conf - 0.85) / 0.15 * 50)
        severity_score = min(100, severity_score)
        return {
            "severity_score": severity_score,
            "severity_label": "HIGH",
            "risk_level": "CRITICAL"
        }
    elif max_conf >= 0.50:
        # REVIEW range: 16-49
        # Map conf 0.50-0.84 to severity 16-49
        severity_score = int(16 + (max_conf - 0.50) / 0.35 * 33)
        severity_score = min(49, severity_score)
        return {
            "severity_score": severity_score,
            "severity_label": "MEDIUM",
            "risk_level": "WARNING"
        }
    else:
        # Low confidence damage - still ACCEPT range: 1-15
        # Map conf 0.0-0.49 to severity 1-15
        severity_score = int(1 + max_conf / 0.50 * 14)
        severity_score = min(15, severity_score)
        return {
            "severity_score": severity_score,
            "severity_label": "LOW",
            "risk_level": "MINIMAL"
        }


class DecisionType(Enum):
    """Final decision types for package inspection."""
    ACCEPT = auto()
    REJECT = auto()
    REVIEW_REQUIRED = auto()


class Severity(Enum):
    """Damage severity levels."""
    NONE = auto()
    MINOR = auto()
    MODERATE = auto()
    SEVERE = auto()


@dataclass
class ScoredDetection:
    """Detection with calculated severity score."""
    detection: Detection
    severity_score: float
    severity_level: Severity
    size_factor: float
    confidence_factor: float
    base_weight: int


@dataclass 
class Decision:
    """
    Complete decision result for a package inspection.
    """
    decision_type: DecisionType
    package_id: str
    timestamp: datetime
    
    # All detections from all cameras
    detections: List[ScoredDetection] = field(default_factory=list)
    
    # Aggregated info
    total_detections: int = 0
    max_severity: Severity = Severity.NONE
    max_severity_score: float = 0.0
    
    # Decision rationale
    rationale: str = ""
    
    # Operator override (if any)
    operator_decision: Optional[DecisionType] = None
    operator_id: Optional[str] = None
    operator_notes: Optional[str] = None
    operator_timestamp: Optional[datetime] = None
    
    @property
    def final_decision(self) -> DecisionType:
        """Get final decision (operator override takes precedence)."""
        if self.operator_decision is not None:
            return self.operator_decision
        return self.decision_type
    
    @property
    def is_reviewed(self) -> bool:
        """Check if operator has reviewed this decision."""
        return self.operator_decision is not None


class DecisionEngine:
    """
    Decision engine for package damage assessment.
    
    Takes detection results from multiple cameras and produces
    an accept/reject/review decision with severity scoring.
    """
    
    # Default class severity weights
    DEFAULT_CLASS_WEIGHTS = {
        "structural_deformation": 2,
        "surface_breach": 4,
        "contamination_stain": 3,
        "compression_damage": 3,
        "tape_seal_damage": 4,
    }
    
    # Default size thresholds
    DEFAULT_SIZE_THRESHOLDS = {
        "large": 0.15,
        "medium": 0.05,
        "small": 0.02,
        "tiny": 0.0,
    }
    
    # Default confidence thresholds
    DEFAULT_CONF_THRESHOLDS = {
        "high": 0.85,
        "good": 0.70,
        "moderate": 0.50,
        "low": 0.0,
    }
    
    # Default severity thresholds
    DEFAULT_SEVERITY_THRESHOLDS = {
        "severe": 6.0,
        "moderate": 3.0,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the decision engine.
        
        Args:
            config: Optional configuration dictionary
        """
        config = config or {}
        decision_config = config.get("decision", {})
        
        # Load configuration
        self.class_weights = decision_config.get(
            "class_weights", self.DEFAULT_CLASS_WEIGHTS
        )
        self.size_thresholds = decision_config.get(
            "size_thresholds", self.DEFAULT_SIZE_THRESHOLDS
        )
        self.confidence_thresholds = decision_config.get(
            "confidence_thresholds", self.DEFAULT_CONF_THRESHOLDS
        )
        self.severity_thresholds = decision_config.get(
            "severity_thresholds", self.DEFAULT_SEVERITY_THRESHOLDS
        )
        
        # Decision rules
        rules = decision_config.get("rules", {})
        self.auto_reject_on_severe = rules.get("auto_reject_on_severe", True)
        self.review_on_multiple_minor = rules.get("review_on_multiple_minor", 3)
        self.operator_timeout = rules.get("operator_timeout", 30)
        
        # Fusion settings
        fusion = decision_config.get("fusion", {})
        self.corroboration_threshold = fusion.get("corroboration_threshold", 0.40)
        self.standalone_threshold = fusion.get("standalone_threshold", 0.70)
        
        logger.info("Decision engine initialized")
    
    def calculate_size_factor(self, area_normalized: float) -> float:
        """
        Calculate size factor based on detection area.
        
        Args:
            area_normalized: Detection area as fraction of image (0-1)
            
        Returns:
            Size factor multiplier
        """
        if area_normalized >= self.size_thresholds["large"]:
            return 2.0
        elif area_normalized >= self.size_thresholds["medium"]:
            return 1.5
        elif area_normalized >= self.size_thresholds["small"]:
            return 1.0
        else:
            return 0.5
    
    def calculate_confidence_factor(self, confidence: float) -> float:
        """
        Calculate confidence factor.
        
        Args:
            confidence: Model confidence (0-1)
            
        Returns:
            Confidence factor multiplier
        """
        if confidence >= self.confidence_thresholds["high"]:
            return 1.2
        elif confidence >= self.confidence_thresholds["good"]:
            return 1.0
        elif confidence >= self.confidence_thresholds["moderate"]:
            return 0.8
        else:
            return 0.5
    
    def calculate_severity(self, detection: Detection) -> Tuple[float, Severity]:
        """
        Calculate severity score and level for a detection.
        
        Args:
            detection: Detection object
            
        Returns:
            Tuple of (severity_score, severity_level)
        """
        # Get base weight for class
        base_weight = self.class_weights.get(detection.class_name, 2)
        
        # Calculate factors
        size_factor = self.calculate_size_factor(detection.area_normalized)
        conf_factor = self.calculate_confidence_factor(detection.confidence)
        
        # Calculate final score
        severity_score = base_weight * size_factor * conf_factor
        
        # Determine severity level
        if severity_score >= self.severity_thresholds["severe"]:
            severity_level = Severity.SEVERE
        elif severity_score >= self.severity_thresholds["moderate"]:
            severity_level = Severity.MODERATE
        else:
            severity_level = Severity.MINOR
        
        return severity_score, severity_level
    
    def score_detection(self, detection: Detection) -> ScoredDetection:
        """
        Score a single detection.
        
        Args:
            detection: Detection object
            
        Returns:
            ScoredDetection with severity information
        """
        base_weight = self.class_weights.get(detection.class_name, 2)
        size_factor = self.calculate_size_factor(detection.area_normalized)
        conf_factor = self.calculate_confidence_factor(detection.confidence)
        severity_score, severity_level = self.calculate_severity(detection)
        
        return ScoredDetection(
            detection=detection,
            severity_score=severity_score,
            severity_level=severity_level,
            size_factor=size_factor,
            confidence_factor=conf_factor,
            base_weight=base_weight
        )
    
    def fuse_detections(
        self,
        inference_results: List[InferenceResult]
    ) -> List[ScoredDetection]:
        """
        Fuse detections from multiple cameras.
        
        Applies corroboration logic for uncertain detections.
        
        Args:
            inference_results: Results from all cameras
            
        Returns:
            List of scored detections after fusion
        """
        all_detections = []
        
        # Collect all detections from all cameras
        for result in inference_results:
            for detection in result.detections:
                all_detections.append(detection)
        
        # Score all detections
        scored_detections = []
        for detection in all_detections:
            scored = self.score_detection(detection)
            
            # Check if detection needs corroboration
            if detection.confidence < self.standalone_threshold:
                # Look for corroborating detection from another camera
                has_corroboration = self._check_corroboration(
                    detection, all_detections
                )
                
                if not has_corroboration:
                    # Reduce severity for uncorroborated low-confidence detection
                    scored.severity_score *= 0.5
                    if scored.severity_level == Severity.SEVERE:
                        scored.severity_level = Severity.MODERATE
                    elif scored.severity_level == Severity.MODERATE:
                        scored.severity_level = Severity.MINOR
            
            scored_detections.append(scored)
        
        return scored_detections
    
    def _check_corroboration(
        self,
        target: Detection,
        all_detections: List[Detection]
    ) -> bool:
        """
        Check if a detection has corroboration from another camera.
        
        Args:
            target: Detection to check
            all_detections: All detections from all cameras
            
        Returns:
            True if corroboration found
        """
        for other in all_detections:
            # Skip same camera
            if other.camera_id == target.camera_id:
                continue
            
            # Same class?
            if other.class_name == target.class_name:
                # If another camera detected same class, consider corroborated
                if other.confidence >= self.corroboration_threshold:
                    return True
        
        return False
    
    def make_decision(
        self,
        inference_results: List[InferenceResult],
        package_id: str
    ) -> Decision:
        """
        Make accept/reject/review decision based on inference results.
        
        Args:
            inference_results: Results from all cameras
            package_id: Unique identifier for the package
            
        Returns:
            Decision object with complete analysis
        """
        timestamp = datetime.utcnow()
        
        # Fuse and score detections
        scored_detections = self.fuse_detections(inference_results)
        
        # If no detections, accept
        if not scored_detections:
            return Decision(
                decision_type=DecisionType.ACCEPT,
                package_id=package_id,
                timestamp=timestamp,
                detections=[],
                total_detections=0,
                max_severity=Severity.NONE,
                max_severity_score=0.0,
                rationale="No damage detected"
            )
        
        # Calculate definitive severity
        severity_info = compute_severity(scored_detections)
        severity_score = severity_info["severity_score"]
        severity_label_str = severity_info["severity_label"]
        
        # Map string label back to Enum
        if severity_label_str == "HIGH":
            max_severity = Severity.SEVERE
        elif severity_label_str == "MEDIUM":
            max_severity = Severity.MODERATE
        else:
            max_severity = Severity.MINOR  # SAFE maps to MINOR/NONE internally
            
        max_severity_score = float(severity_score)
        
        # Count by severity level (legacy support for rules below)
        severity_counts = {
            Severity.MINOR: 0,
            Severity.MODERATE: 0,
            Severity.SEVERE: 0,
        }
        
        # Use new definitive severity to populate counts
        for d in scored_detections:
            if d.detection.class_name == "damaged":
                # Re-evaluate individual severity using the definitive logic if needed
                # For now, we rely on the overall package severity
                pass
        
        # Update counts based on the definitive result
        if max_severity == Severity.SEVERE:
            severity_counts[Severity.SEVERE] = 1
        elif max_severity == Severity.MODERATE:
            severity_counts[Severity.MODERATE] = 1
        elif severity_score > 0:
             severity_counts[Severity.MINOR] = 1
        
        # Apply decision rules
        decision_type = DecisionType.ACCEPT
        rationale = ""
        
        # Rule 1: Auto-reject on severe damage
        if self.auto_reject_on_severe and severity_counts[Severity.SEVERE] > 0:
            decision_type = DecisionType.REJECT
            rationale = f"Severe damage detected ({severity_counts[Severity.SEVERE]} instance(s))"
        
        # Rule 2: Review on moderate damage
        elif severity_counts[Severity.MODERATE] > 0:
            decision_type = DecisionType.REVIEW_REQUIRED
            rationale = f"Moderate damage detected ({severity_counts[Severity.MODERATE]} instance(s))"
        
        # Rule 3: Review on multiple minor damages
        elif severity_counts[Severity.MINOR] >= self.review_on_multiple_minor:
            decision_type = DecisionType.REVIEW_REQUIRED
            rationale = f"Multiple minor damages detected ({severity_counts[Severity.MINOR]} instance(s))"
        
        # Rule 4: Accept with minor damage
        elif severity_counts[Severity.MINOR] > 0:
            decision_type = DecisionType.ACCEPT
            rationale = f"Minor damage only ({severity_counts[Severity.MINOR]} instance(s)) - acceptable"
        
        else:
            decision_type = DecisionType.ACCEPT
            rationale = "No significant damage"
        
        return Decision(
            decision_type=decision_type,
            package_id=package_id,
            timestamp=timestamp,
            detections=scored_detections,
            total_detections=len(scored_detections),
            max_severity=max_severity,
            max_severity_score=max_severity_score,
            rationale=rationale
        )
    
    def apply_operator_decision(
        self,
        decision: Decision,
        operator_decision: DecisionType,
        operator_id: str,
        notes: str = ""
    ) -> Decision:
        """
        Apply operator override to a decision.
        
        Args:
            decision: Original decision
            operator_decision: Operator's decision
            operator_id: Operator identifier
            notes: Optional notes from operator
            
        Returns:
            Updated decision with operator override
        """
        decision.operator_decision = operator_decision
        decision.operator_id = operator_id
        decision.operator_notes = notes
        decision.operator_timestamp = datetime.utcnow()
        
        logger.info(
            f"Operator {operator_id} overrode decision for {decision.package_id}: "
            f"{decision.decision_type.name} -> {operator_decision.name}"
        )
        
        return decision


def create_decision_engine(config: Dict[str, Any]) -> DecisionEngine:
    """
    Factory function to create a DecisionEngine from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DecisionEngine instance
    """
    return DecisionEngine(config)
