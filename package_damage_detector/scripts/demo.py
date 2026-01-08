#!/usr/bin/env python3
"""
Demo Test Script

Quick demonstration of the package damage detection system.
Runs in demo mode without requiring cameras or trained model.
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("  📦 PACKAGE DAMAGE DETECTION SYSTEM - DEMO")
    print("  Edge-Based Intelligent Package Damage Detection")
    print("=" * 70 + "\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def simulate_camera_capture():
    """Simulate multi-camera capture."""
    cameras = ["CAM-01-TOP", "CAM-02-FRONT", "CAM-03-LEFT", "CAM-04-RIGHT", "CAM-05-BACK"]
    
    print("  Capturing from cameras...")
    for cam in cameras:
        time.sleep(0.1)
        print(f"    ✓ {cam} - 2592×1944 captured")
    
    return {cam: f"[Image data from {cam}]" for cam in cameras}


def simulate_inference():
    """Simulate YOLOv5 inference."""
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
    
    return detections


def calculate_decision(detections):
    """Calculate final decision."""
    if not detections:
        return "ACCEPT", "No damage detected"
    
    severities = [d["severity"] for d in detections]
    
    if "SEVERE" in severities:
        return "REJECT", "Severe damage detected"
    elif "MODERATE" in severities:
        return "REVIEW_REQUIRED", "Moderate damage requires operator review"
    elif len(detections) >= 3:
        return "REVIEW_REQUIRED", "Multiple minor damages detected"
    else:
        return "ACCEPT", "Minor damage only - acceptable"


def run_demo_inspection(package_id: str):
    """Run a single demo inspection."""
    print_section(f"INSPECTING: {package_id}")
    
    start_time = time.time()
    
    # Step 1: Capture
    print("  📷 STEP 1: Multi-Camera Capture")
    captures = simulate_camera_capture()
    capture_time = random.uniform(20, 40)
    print(f"    Capture time: {capture_time:.1f}ms")
    
    # Step 2: Inference
    print("\n  🔍 STEP 2: AI Inference")
    time.sleep(0.3)
    inference_time = random.uniform(30, 80)
    print(f"    Running YOLOv5 inference...")
    detections = simulate_inference()
    print(f"    Inference time: {inference_time:.1f}ms")
    print(f"    Detections found: {len(detections)}")
    
    # Step 3: Decision
    print("\n  ⚖️  STEP 3: Decision Logic")
    decision, rationale = calculate_decision(detections)
    decision_time = random.uniform(1, 5)
    print(f"    Decision time: {decision_time:.1f}ms")
    
    # Step 4: Evidence
    print("\n  💾 STEP 4: Evidence Storage")
    evidence_time = random.uniform(50, 100)
    evidence_id = f"INS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"    Evidence ID: {evidence_id}")
    print(f"    Storage time: {evidence_time:.1f}ms")
    print(f"    Hash chain updated: ✓")
    
    total_time = capture_time + inference_time + decision_time + evidence_time
    
    # Results
    print_section("INSPECTION RESULTS")
    
    # Decision display
    if decision == "ACCEPT":
        print(f"  ✅ DECISION: {decision}")
    elif decision == "REJECT":
        print(f"  ❌ DECISION: {decision}")
    else:
        print(f"  ⚠️  DECISION: {decision}")
    
    print(f"  📝 Rationale: {rationale}")
    print(f"  ⏱️  Total Time: {total_time:.1f}ms")
    
    # Detections
    if detections:
        print("\n  DETECTIONS:")
        for i, det in enumerate(detections, 1):
            severity_icon = {"MINOR": "🟡", "MODERATE": "🟠", "SEVERE": "🔴"}[det["severity"]]
            print(f"    {i}. {det['class_name']}")
            print(f"       Confidence: {det['confidence']:.1%}")
            print(f"       Severity: {severity_icon} {det['severity']}")
            print(f"       Camera: {det['camera']}")
    else:
        print("\n  DETECTIONS: None")
    
    return decision


def main():
    """Run demo."""
    print_header()
    
    print("This demo simulates the package damage detection system")
    print("without requiring cameras or a trained model.\n")
    
    # Run multiple inspections
    num_inspections = 5
    results = {"ACCEPT": 0, "REJECT": 0, "REVIEW_REQUIRED": 0}
    
    for i in range(num_inspections):
        package_id = f"PKG-DEMO-{i+1:04d}"
        decision = run_demo_inspection(package_id)
        results[decision] += 1
        
        if i < num_inspections - 1:
            print("\n" + "·" * 70)
            time.sleep(1)
    
    # Summary
    print_section("SESSION SUMMARY")
    print(f"  Total Inspections: {num_inspections}")
    print(f"  ✅ Accepted: {results['ACCEPT']}")
    print(f"  ❌ Rejected: {results['REJECT']}")
    print(f"  ⚠️  Review Required: {results['REVIEW_REQUIRED']}")
    
    print("\n" + "=" * 70)
    print("  Demo complete! Run with actual hardware for full functionality.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
