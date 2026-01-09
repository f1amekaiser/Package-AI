#!/usr/bin/env python3
"""
Package Damage Detection System - Main Application

Edge-Based Intelligent Package Damage Detection for Warehouse Receiving Docks

Usage:
    # Run single image inspection
    python main.py --image path/to/image.jpg
    
    # Run in demo mode (simulated inspections)
    python main.py --demo
    
    # Run web UI server
    python main.py --server
    
    # Run continuous demo mode
    python main.py --demo --interval 3
"""

import argparse
import sys
import signal
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.helpers import load_config, setup_logging, generate_package_id
from src.core.inference_engine import TwoStageInferenceEngine, TwoStageDetection

logger = logging.getLogger(__name__)


class PackageDamageDetector:
    """
    Main application class for the package damage detection system.
    Uses Two-Stage Pipeline: YOLO Detector + Damage Classifier
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        demo_mode: bool = False
    ):
        """
        Initialize the detector application.
        
        Args:
            config_path: Path to configuration file
            demo_mode: Use simulated inspections for testing
        """
        self.config_path = config_path
        self.demo_mode = demo_mode
        self.engine: Optional[TwoStageInferenceEngine] = None
        self._running = False
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        log_config = self.config.get("logging", {})
        setup_logging(
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            format_string=log_config.get("format")
        )
        
        logger.info("Package Damage Detector initializing...")
        logger.info(f"Demo mode: {demo_mode}")
    
    def initialize(self) -> bool:
        """
        Initialize the two-stage inference engine.
        
        Returns:
            True if initialization successful
        """
        if self.demo_mode:
            logger.info("Running in demo mode - no model loading required")
            return True
        
        try:
            logger.info("Initializing Two-Stage Inference Engine...")
            
            models_dir = Path(__file__).parent / "models"
            detector_path = models_dir / "best.pt"
            classifier_path = models_dir / "damaged_classifier_best.pt"
            
            if not detector_path.exists():
                raise FileNotFoundError(f"Detector model not found: {detector_path}")
            if not classifier_path.exists():
                raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
            
            self.engine = TwoStageInferenceEngine(
                detector_path=str(detector_path),
                classifier_path=str(classifier_path),
                detector_conf=0.05,
                device="cpu"
            )
            
            logger.info("Two-Stage Inference Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def inspect_image(self, image_path: str, package_id: Optional[str] = None) -> dict:
        """
        Run inspection on a single image using the two-stage pipeline.
        
        Args:
            image_path: Path to the image file
            package_id: Optional package ID
            
        Returns:
            Inspection result dictionary
        """
        if not package_id:
            package_id = generate_package_id()
        
        logger.info(f"Inspecting image: {image_path}")
        logger.info(f"Package ID: {package_id}")
        
        start_time = time.time()
        
        # Load image
        pil_img = Image.open(image_path).convert('RGB')
        img_array = np.array(pil_img)
        
        # Run two-stage inference
        decision, detections, reason = self.engine.infer_with_decision(img_array)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Format result
        result = {
            "inspection_id": f"INS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "package_id": package_id,
            "decision": decision,
            "reason": reason,
            "detections": len(detections),
            "inference_time_ms": inference_time,
            "details": []
        }
        
        for det in detections:
            result["details"].append({
                "label": det.classifier_label,
                "confidence": f"{det.classifier_confidence:.1%}",
                "yolo_conf": f"{det.yolo_confidence:.1%}"
            })
        
        return result
    
    def run_demo_inspection(self, package_id: str) -> dict:
        """
        Run a simulated inspection for demo mode.
        
        Args:
            package_id: Package ID
            
        Returns:
            Simulated inspection result
        """
        damage_types = [
            ("structural_deformation", "Dent on front panel"),
            ("surface_breach", "Tear on corner"),
            ("contamination_stain", "Water stain detected"),
            ("compression_damage", "Crushed corner"),
            ("tape_seal_damage", "Tape peeling"),
        ]
        
        # Randomly decide if damaged
        is_damaged = random.random() > 0.4
        
        detections = []
        if is_damaged:
            num_detections = random.randint(1, 3)
            for _ in range(num_detections):
                damage = random.choice(damage_types)
                detections.append({
                    "class_name": damage[0],
                    "description": damage[1],
                    "confidence": random.uniform(0.55, 0.95),
                    "severity": random.choice(["MINOR", "MODERATE", "SEVERE"]),
                    "camera": f"CAM-0{random.randint(1, 5)}"
                })
        
        # Calculate decision
        if not detections:
            decision = "ACCEPT"
            reason = "No damage detected"
        elif any(d["severity"] == "SEVERE" for d in detections):
            decision = "REJECT"
            reason = "Severe damage detected"
        elif any(d["severity"] == "MODERATE" for d in detections):
            decision = "REVIEW_REQUIRED"
            reason = "Moderate damage requires operator review"
        else:
            decision = "ACCEPT"
            reason = "Minor damage only - acceptable"
        
        return {
            "inspection_id": f"INS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "package_id": package_id,
            "decision": decision,
            "reason": reason,
            "detections": detections,
            "inference_time_ms": random.uniform(30, 80)
        }
    
    def run_single_inspection(self, package_id: Optional[str] = None, image_path: Optional[str] = None) -> bool:
        """
        Run a single package inspection.
        
        Args:
            package_id: Optional package ID (generated if not provided)
            image_path: Optional path to image file
            
        Returns:
            True if package accepted
        """
        if not package_id:
            package_id = generate_package_id()
        
        if self.demo_mode:
            result = self.run_demo_inspection(package_id)
        elif image_path:
            result = self.inspect_image(image_path, package_id)
        else:
            logger.error("Image path required for non-demo mode")
            return False
        
        # Log result
        logger.info("=" * 60)
        logger.info(f"INSPECTION COMPLETE: {result['inspection_id']}")
        logger.info(f"Package ID: {package_id}")
        logger.info(f"Decision: {result['decision']}")
        logger.info(f"Reason: {result['reason']}")
        
        if self.demo_mode:
            logger.info(f"Detections: {len(result['detections'])}")
            for i, det in enumerate(result['detections'], 1):
                severity_icon = {"MINOR": "🟡", "MODERATE": "🟠", "SEVERE": "🔴"}[det["severity"]]
                logger.info(f"  {i}. {det['class_name']} - {severity_icon} {det['severity']} ({det['confidence']:.1%})")
        else:
            logger.info(f"Detections: {result['detections']}")
            for det in result.get('details', []):
                logger.info(f"  - {det['label']}: {det['confidence']}")
        
        logger.info(f"Inference Time: {result['inference_time_ms']:.1f}ms")
        logger.info("=" * 60)
        
        return result['decision'] == "ACCEPT"
    
    def run_continuous(self, interval_seconds: float = 5.0):
        """
        Run continuous demo inspection loop.
        
        Args:
            interval_seconds: Seconds between inspections
        """
        if not self.demo_mode:
            logger.error("Continuous mode only available in demo mode")
            return
        
        self._running = True
        inspection_count = 0
        accept_count = 0
        reject_count = 0
        review_count = 0
        
        logger.info("Starting continuous inspection mode...")
        logger.info(f"Interval: {interval_seconds} seconds")
        logger.info("Press Ctrl+C to stop")
        
        while self._running:
            try:
                package_id = generate_package_id()
                result = self.run_demo_inspection(package_id)
                inspection_count += 1
                
                # Track stats
                if result['decision'] == "ACCEPT":
                    accept_count += 1
                    icon = "✅"
                elif result['decision'] == "REJECT":
                    reject_count += 1
                    icon = "❌"
                else:
                    review_count += 1
                    icon = "⚠️"
                
                # Log summary
                logger.info(
                    f"[{inspection_count}] {package_id}: {icon} {result['decision']} "
                    f"({len(result['detections'])} detections, {result['inference_time_ms']:.1f}ms)"
                )
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Inspection error: {e}")
                time.sleep(1)
        
        # Final stats
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"Total Inspections: {inspection_count}")
        logger.info(f"✅ Accepted: {accept_count} ({100*accept_count/max(1,inspection_count):.1f}%)")
        logger.info(f"❌ Rejected: {reject_count} ({100*reject_count/max(1,inspection_count):.1f}%)")
        logger.info(f"⚠️  Review Required: {review_count} ({100*review_count/max(1,inspection_count):.1f}%)")
        logger.info("=" * 60)
    
    def stop(self):
        """Stop the continuous loop."""
        self._running = False
        logger.info("Stopping...")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleanup complete")


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the web UI server."""
    from src.ui.server import run_ui_server
    run_ui_server(host=host, port=port, demo_mode=True, debug=debug)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Package Damage Detection System - Two-Stage Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with simulated inspections"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run the web UI server"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file for single inspection"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single inspection and exit"
    )
    parser.add_argument(
        "--package-id",
        type=str,
        default=None,
        help="Package ID for single inspection"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between inspections in continuous mode"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web server"
    )
    
    args = parser.parse_args()
    
    # Run web server if requested
    if args.server:
        print("\n" + "=" * 60)
        print("  📦 PACKAGE DAMAGE DETECTION SYSTEM - WEB UI")
        print("  Two-Stage Pipeline: YOLO Detector + Damage Classifier")
        print("=" * 60)
        print(f"\n  Starting server on http://localhost:{args.port}")
        print("  Press Ctrl+C to stop\n")
        run_server(port=args.port, debug=True)
        return
    
    # Create detector
    detector = PackageDamageDetector(
        config_path=args.config,
        demo_mode=args.demo
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        print("\nShutdown requested...")
        detector.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize
        if not detector.initialize():
            logger.error("Failed to initialize system")
            sys.exit(1)
        
        # Run
        if args.single or args.image:
            accepted = detector.run_single_inspection(args.package_id, args.image)
            sys.exit(0 if accepted else 1)
        else:
            detector.run_continuous(interval_seconds=args.interval)
    
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
