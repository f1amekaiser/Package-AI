"""
Inference Engine Module

Handles loading YOLOv5 model and running inference on images.
Supports both PyTorch and TensorRT backends.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import cv2
import torch

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)
    bbox_pixels: Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixels)
    camera_id: str = ""
    
    @property
    def area_normalized(self) -> float:
        """Calculate normalized area of detection box."""
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        return width * height
    
    @property
    def area_pixels(self) -> int:
        """Calculate pixel area of detection box."""
        width = self.bbox_pixels[2] - self.bbox_pixels[0]
        height = self.bbox_pixels[3] - self.bbox_pixels[1]
        return width * height


@dataclass
class InferenceResult:
    """Container for inference results from a single image."""
    image_path: str
    camera_id: str
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)  # height, width
    
    @property
    def total_time_ms(self) -> float:
        return self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms
    
    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


class InferenceEngine:
    """
    YOLOv5 Inference Engine for package damage detection.
    
    Supports:
    - PyTorch (.pt) models
    - TensorRT (.engine) models
    - CPU and GPU inference
    """
    
    # Class names for package damage detection
    CLASS_NAMES = [
        "structural_deformation",
        "surface_breach", 
        "contamination_stain",
        "compression_damage",
        "tape_seal_damage"
    ]
    
    def __init__(
        self,
        weights_path: str,
        device: str = "0",
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        half_precision: bool = True,
        max_detections: int = 100
    ):
        """
        Initialize the inference engine.
        
        Args:
            weights_path: Path to model weights (.pt or .engine)
            device: Device to run inference ("0" for GPU, "cpu" for CPU)
            input_size: Model input size (default 640)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            half_precision: Use FP16 inference (GPU only)
            max_detections: Maximum detections per image
        """
        self.weights_path = Path(weights_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.half_precision = half_precision and device != "cpu"
        self.max_detections = max_detections
        
        # Set device
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing inference engine on device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Warmup
        self._warmup()
        
        logger.info("Inference engine initialized successfully")
    
    def _load_model(self):
        """Load the model based on file extension."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        
        suffix = self.weights_path.suffix.lower()
        
        if suffix == ".pt":
            return self._load_pytorch_model()
        elif suffix == ".engine":
            return self._load_tensorrt_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        logger.info(f"Loading PyTorch model from {self.weights_path}")
        
        # Use torch.hub to load YOLOv5
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=str(self.weights_path),
            force_reload=False
        )
        
        model.to(self.device)
        
        if self.half_precision and self.device.type != "cpu":
            model.half()
        
        model.eval()
        
        # Set model parameters
        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        model.max_det = self.max_detections
        
        return model
    
    def _load_tensorrt_model(self):
        """Load TensorRT engine."""
        logger.info(f"Loading TensorRT engine from {self.weights_path}")
        
        # TensorRT loading requires additional setup
        # This is a placeholder - actual implementation depends on TensorRT version
        try:
            from utils.general import check_requirements
            check_requirements(['tensorrt'])
            
            # Load TensorRT engine using YOLOv5's DetectMultiBackend
            from models.common import DetectMultiBackend
            model = DetectMultiBackend(
                str(self.weights_path),
                device=self.device,
                fp16=self.half_precision
            )
            return model
        except ImportError:
            logger.error("TensorRT not available, falling back to PyTorch")
            raise RuntimeError("TensorRT not available")
    
    def _warmup(self, iterations: int = 3):
        """Warm up the model with dummy inputs."""
        logger.info("Warming up model...")
        
        dummy_input = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        
        for _ in range(iterations):
            self.infer(dummy_input, camera_id="warmup")
        
        logger.info("Model warmup complete")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Preprocessed image and original shape
        """
        original_shape = image.shape[:2]  # height, width
        
        # Letterbox resize
        img = self._letterbox(image, self.input_size)
        
        # BGR to RGB
        img = img[:, :, ::-1]
        
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        
        # Normalize and convert to contiguous array
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        
        return img, original_shape
    
    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> np.ndarray:
        """Resize image with letterboxing (maintain aspect ratio)."""
        shape = image.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape / shape[0], new_shape / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        
        # Divide padding into 2 sides
        dw /= 2
        dh /= 2
        
        # Resize
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        
        return image
    
    def infer(
        self,
        image: Union[np.ndarray, str],
        camera_id: str = ""
    ) -> InferenceResult:
        """
        Run inference on a single image.
        
        Args:
            image: BGR image as numpy array or path to image file
            camera_id: Identifier for the camera that captured the image
            
        Returns:
            InferenceResult with detections and timing info
        """
        # Load image if path provided
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            image_path = ""
        
        original_shape = image.shape[:2]
        
        # Preprocess
        t0 = time.perf_counter()
        processed, _ = self.preprocess(image)
        preprocess_time = (time.perf_counter() - t0) * 1000
        
        # Inference
        t1 = time.perf_counter()
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        if self.half_precision:
            tensor = tensor.half()
        
        # Run model
        with torch.no_grad():
            results = self.model(tensor)
        
        inference_time = (time.perf_counter() - t1) * 1000
        
        # Post-process
        t2 = time.perf_counter()
        detections = self._postprocess(results, original_shape, camera_id)
        postprocess_time = (time.perf_counter() - t2) * 1000
        
        return InferenceResult(
            image_path=image_path,
            camera_id=camera_id,
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time,
            image_shape=original_shape
        )
    
    def _postprocess(
        self,
        results,
        original_shape: Tuple[int, int],
        camera_id: str
    ) -> List[Detection]:
        """Process model output into Detection objects."""
        detections = []
        
        # Handle different result formats
        if hasattr(results, 'xyxy'):
            # torch.hub model output
            pred = results.xyxy[0].cpu().numpy()
        else:
            # Direct tensor output
            pred = results[0].cpu().numpy() if isinstance(results, tuple) else results.cpu().numpy()
        
        height, width = original_shape
        
        for det in pred:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls_id = det[:6]
                
                # Skip low confidence
                if conf < self.confidence_threshold:
                    continue
                
                cls_id = int(cls_id)
                
                # Get class name
                if cls_id < len(self.CLASS_NAMES):
                    class_name = self.CLASS_NAMES[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                # Normalize coordinates
                bbox_norm = (
                    float(x1 / width),
                    float(y1 / height),
                    float(x2 / width),
                    float(y2 / height)
                )
                
                # Pixel coordinates
                bbox_pixels = (
                    int(x1), int(y1), int(x2), int(y2)
                )
                
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=bbox_norm,
                    bbox_pixels=bbox_pixels,
                    camera_id=camera_id
                ))
        
        return detections
    
    def infer_batch(
        self,
        images: List[Tuple[np.ndarray, str]]
    ) -> List[InferenceResult]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of (image, camera_id) tuples
            
        Returns:
            List of InferenceResult objects
        """
        results = []
        for image, camera_id in images:
            result = self.infer(image, camera_id)
            results.append(result)
        return results
    
    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: BGR image
            detections: List of Detection objects
            show_labels: Whether to show class labels
            line_thickness: Box line thickness
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Color palette for classes
        colors = [
            (255, 165, 0),   # structural_deformation - orange
            (0, 0, 255),     # surface_breach - red
            (255, 0, 255),   # contamination_stain - magenta
            (0, 165, 255),   # compression_damage - orange-yellow
            (0, 255, 255),   # tape_seal_damage - yellow
        ]
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox_pixels
            color = colors[det.class_id % len(colors)]
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            
            if show_labels:
                label = f"{det.class_name}: {det.confidence:.2f}"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 5, y1),
                    color, -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated, label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )
        
        return annotated


def create_engine(config: Dict[str, Any]) -> InferenceEngine:
    """
    Factory function to create an InferenceEngine from config.
    
    Args:
        config: Configuration dictionary with model settings
        
    Returns:
        Configured InferenceEngine instance
    """
    model_config = config.get("model", {})
    
    # Try TensorRT first, fall back to PyTorch
    weights_path = model_config.get("tensorrt_engine", "")
    if not weights_path or not Path(weights_path).exists():
        weights_path = model_config.get("weights_path", "models/damage_detector.pt")
    
    return InferenceEngine(
        weights_path=weights_path,
        device=model_config.get("device", "0"),
        input_size=model_config.get("input_size", 640),
        confidence_threshold=model_config.get("confidence_threshold", 0.25),
        iou_threshold=model_config.get("iou_threshold", 0.45),
        half_precision=model_config.get("half_precision", True),
        max_detections=model_config.get("max_detections", 100)
    )


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
        
        # Load models using Ultralytics YOLO
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

