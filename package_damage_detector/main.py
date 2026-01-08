#!/usr/bin/env python3
"""
Package Damage Detection System - Main Application

Edge-Based Intelligent Package Damage Detection for Warehouse Receiving Docks

Usage:
    # Run with default config
    python main.py
    
    # Run with custom config
    python main.py --config path/to/config.yaml
    
    # Run in demo mode (simulated cameras)
    python main.py --demo
    
    # Run single inspection
    python main.py --single --package-id PKG-001
"""

import argparse
import sys
import signal
import time
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.helpers import load_config, setup_logging, generate_package_id, format_duration
from src.services.inspection_service import create_inspection_service, InspectionService
from src.core.decision_engine import DecisionType

logger = logging.getLogger(__name__)


class PackageDamageDetector:
    """
    Main application class for the package damage detection system.
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
            demo_mode: Use simulated cameras for testing
        """
        self.config_path = config_path
        self.demo_mode = demo_mode
        self.inspection_service: Optional[InspectionService] = None
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
        Initialize all system components.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing inspection service...")
            
            self.inspection_service = create_inspection_service(
                self.config,
                simulated_cameras=self.demo_mode
            )
            
            logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def run_single_inspection(self, package_id: Optional[str] = None) -> bool:
        """
        Run a single package inspection.
        
        Args:
            package_id: Optional package ID (generated if not provided)
            
        Returns:
            True if package accepted
        """
        if not self.inspection_service:
            logger.error("System not initialized")
            return False
        
        # Generate package ID if not provided
        if not package_id:
            package_id = generate_package_id()
        
        logger.info(f"Starting inspection for package: {package_id}")
        
        # Run inspection
        result = self.inspection_service.inspect_package(package_id)
        
        # Log result
        decision = result.decision
        logger.info("=" * 60)
        logger.info(f"INSPECTION COMPLETE: {result.inspection_id}")
        logger.info(f"Package ID: {package_id}")
        logger.info(f"Decision: {decision.decision_type.name}")
        logger.info(f"Rationale: {decision.rationale}")
        logger.info(f"Detections: {decision.total_detections}")
        logger.info(f"Max Severity: {decision.max_severity.name}")
        logger.info(f"Total Time: {format_duration(result.total_time_ms)}")
        logger.info("  - Capture: " + format_duration(result.capture_time_ms))
        logger.info("  - Inference: " + format_duration(result.inference_time_ms))
        logger.info("  - Decision: " + format_duration(result.decision_time_ms))
        logger.info("  - Evidence: " + format_duration(result.evidence_time_ms))
        logger.info("=" * 60)
        
        # Return accept status
        return result.is_accept
    
    def run_continuous(self, interval_seconds: float = 5.0):
        """
        Run continuous inspection loop.
        
        Simulates production mode with periodic inspections.
        
        Args:
            interval_seconds: Seconds between inspections
        """
        if not self.inspection_service:
            logger.error("System not initialized")
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
                # Generate package ID
                package_id = generate_package_id()
                
                # Run inspection
                result = self.inspection_service.inspect_package(package_id)
                inspection_count += 1
                
                # Track stats
                if result.decision.decision_type == DecisionType.ACCEPT:
                    accept_count += 1
                elif result.decision.decision_type == DecisionType.REJECT:
                    reject_count += 1
                else:
                    review_count += 1
                
                # Log summary
                logger.info(
                    f"[{inspection_count}] {package_id}: "
                    f"{result.decision.decision_type.name} "
                    f"({result.decision.total_detections} detections, "
                    f"{format_duration(result.total_time_ms)})"
                )
                
                # Wait for next inspection
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
        logger.info(f"Accepted: {accept_count} ({100*accept_count/max(1,inspection_count):.1f}%)")
        logger.info(f"Rejected: {reject_count} ({100*reject_count/max(1,inspection_count):.1f}%)")
        logger.info(f"Review Required: {review_count} ({100*review_count/max(1,inspection_count):.1f}%)")
        logger.info("=" * 60)
    
    def stop(self):
        """Stop the continuous loop."""
        self._running = False
        logger.info("Stopping...")
    
    def cleanup(self):
        """Clean up resources."""
        if self.inspection_service:
            self.inspection_service.camera_manager.release()
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Package Damage Detection System"
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
        help="Run in demo mode with simulated cameras"
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
    
    args = parser.parse_args()
    
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
        if args.single:
            accepted = detector.run_single_inspection(args.package_id)
            sys.exit(0 if accepted else 1)
        else:
            detector.run_continuous(interval_seconds=args.interval)
    
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
