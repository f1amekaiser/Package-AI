"""
Two-Stage Inference Engine Module

Package damage detection using:
1. YOLO detector - finds potential damage regions
2. Binary classifier - confirms damaged/intact status
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwoStageDetection:
    """Detection result from two-stage pipeline (YOLO + Classifier)."""
    bbox: Dict[str, float]           # {"x1", "y1", "x2", "y2"} normalized
    yolo_confidence: float           # YOLO detection confidence
    classifier_label: str            # "damaged" | "intact"
    classifier_confidence: float     # Classifier confidence
    
    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "yolo_confidence": self.yolo_confidence,
            "classifier_label": self.classifier_label,
            "classifier_confidence": self.classifier_confidence
        }


class TwoStageInferenceEngine:
    """
    Two-Stage Inference Engine for package damage detection.
    
    Pipeline:
    1. YOLO detector finds damage regions
    2. Classifier confirms each region as damaged/intact
    
    Uses ultralytics YOLO package for both models.
    """
    
    def __init__(
        self,
        detector_path: str = "models/best.pt",
        classifier_path: str = "models/damaged_classifier_best.pt",
        detector_conf: float = 0.05,
        device: str = "cpu"
    ):
        """
        Initialize two-stage inference engine.
        
        Args:
            detector_path: Path to YOLO detection model
            classifier_path: Path to classification model
            detector_conf: YOLO confidence threshold
            device: Device for inference ("cpu" or "cuda:0")
        """
        self.detector_conf = detector_conf
        self.device = device
        
        logger.info("Initializing Two-Stage Inference Engine...")
        
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading detector: {detector_path}")
            self.detector = YOLO(detector_path)
            
            logger.info(f"Loading classifier: {classifier_path}")
            self.classifier = YOLO(classifier_path)
            
            logger.info("Two-stage engine initialized successfully")
            
        except ImportError:
            raise RuntimeError("ultralytics package required: pip install ultralytics")
    
    def infer(self, image: np.ndarray) -> List[TwoStageDetection]:
        """
        Run two-stage inference on an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of TwoStageDetection results
        """
        from PIL import Image as PILImage
        
        detections = []
        h, w = image.shape[:2]
        
        # Stage 1: YOLO Detection
        det_results = self.detector(image, conf=self.detector_conf, verbose=False)
        boxes = det_results[0].boxes
        
        if len(boxes) == 0:
            return []
        
        # Convert to PIL for cropping
        pil_image = PILImage.fromarray(image)
        
        # Stage 2: Classify each detection
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            yolo_conf = box.conf[0].item()
            
            # Normalize coordinates
            bbox_norm = {
                "x1": x1 / w,
                "y1": y1 / h,
                "x2": x2 / w,
                "y2": y2 / h
            }
            
            # Crop region
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            crop = pil_image.crop((x1i, y1i, x2i, y2i))
            
            # Skip very small crops
            if crop.width < 32 or crop.height < 32:
                continue
            
            # Run classifier
            cls_results = self.classifier(crop, verbose=False)
            probs = cls_results[0].probs
            cls_label = self.classifier.names[probs.top1]
            cls_conf = probs.top1conf.item()
            
            detection = TwoStageDetection(
                bbox=bbox_norm,
                yolo_confidence=yolo_conf,
                classifier_label=cls_label,
                classifier_confidence=cls_conf
            )
            detections.append(detection)
        
        return detections
    
    def infer_with_decision(
        self,
        image: np.ndarray,
        reject_threshold: float = 0.85,
        review_threshold: float = 0.50
    ) -> Tuple[str, List[TwoStageDetection], str]:
        """
        Run inference and make a decision.
        
        Args:
            image: RGB image as numpy array
            reject_threshold: Classifier confidence threshold for REJECT
            review_threshold: Classifier confidence threshold for REVIEW
            
        Returns:
            Tuple of (decision, detections, reason)
        """
        detections = self.infer(image)
        
        if not detections:
            return "ACCEPT", [], "No damage regions detected"
        
        # Count confirmed and borderline damages
        confirmed_damages = 0
        borderline_damages = 0
        max_conf = 0.0
        
        for det in detections:
            if det.classifier_label == "damaged":
                max_conf = max(max_conf, det.classifier_confidence)
                if det.classifier_confidence >= reject_threshold:
                    confirmed_damages += 1
                elif det.classifier_confidence >= review_threshold:
                    borderline_damages += 1
        
        # Make decision
        if confirmed_damages >= 1:
            return "REJECT", detections, f"{confirmed_damages} confirmed damage(s) detected"
        elif borderline_damages >= 1:
            return "REVIEW_REQUIRED", detections, f"{borderline_damages} borderline detection(s)"
        else:
            return "ACCEPT", detections, "All detections classified as intact"
