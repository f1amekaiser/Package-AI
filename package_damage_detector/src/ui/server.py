"""
Web UI Server

Flask-based web server for the operator console.
"""

import os
import io
import uuid
import base64
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, render_template, jsonify, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Import two-stage pipeline components
from ..core.inference_engine import TwoStageInferenceEngine, TwoStageDetection
from ..core.evidence_manager import TwoStageEvidenceRecorder
from ..core.decision_engine import compute_severity

logger = logging.getLogger(__name__)


def create_ui_app(
    inspection_service=None,
    config: Dict[str, Any] = None
) -> Flask:
    """
    Create Flask app for the operator UI.
    
    Args:
        inspection_service: InspectionService instance
        config: Configuration dictionary
        
    Returns:
        Configured Flask application
    """
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )
    
    app.config['inspection_service'] = inspection_service
    app.config['app_config'] = config or {}
    app.config['stats'] = {
        "total_inspections": 0,
        "accept_count": 0,
        "reject_count": 0,
        "review_count": 0,
        "total_inference_time": 0.0,
        "confidences": []
    }
    
    # Initialize two-stage inference engine
    models_dir = Path(__file__).parent.parent.parent / "models"
    detector_path = models_dir / "best.pt"
    classifier_path = models_dir / "damaged_classifier_best.pt"
    
    two_stage_engine = None
    evidence_recorder = None
    
    if detector_path.exists() and classifier_path.exists():
        try:
            two_stage_engine = TwoStageInferenceEngine(
                detector_path=str(detector_path),
                classifier_path=str(classifier_path),
                detector_conf=0.05
            )
            evidence_recorder = TwoStageEvidenceRecorder(
                storage_path=str(Path(__file__).parent.parent.parent / "evidence")
            )
            logger.info("Two-stage inference engine initialized")
        except Exception as e:
            logger.warning(f"Failed to load two-stage engine: {e}")
    else:
        logger.warning("Model files not found, running in demo mode")
    
    app.config['two_stage_engine'] = two_stage_engine
    app.config['evidence_recorder'] = evidence_recorder
    app.config['inspection_history'] = {}  # Store for operator overrides
    app.config['audit_logs'] = []  # Store for audit trail (enterprise compliance)
    
    # -------------------------------------------------------------------------
    # UI ROUTES
    # -------------------------------------------------------------------------
    
    @app.route('/')
    def index():
        """Serve the main operator console."""
        return render_template('index.html')
    
    # -------------------------------------------------------------------------
    # API ROUTES
    # -------------------------------------------------------------------------
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        service = app.config.get('inspection_service')
        
        cameras_status = {}
        model_loaded = False
        
        if service:
            cameras_status = service.camera_manager.get_camera_status()
            model_loaded = service.inference_engine.model is not None
        
        return jsonify({
            "status": "healthy" if model_loaded else "demo",
            "version": "1.0.0",
            "cameras_status": cameras_status,
            "model_loaded": model_loaded
        })
    
    @app.route('/stats')
    def stats():
        """Get session statistics."""
        stats_data = app.config['stats']
        total = max(stats_data["total_inspections"], 1)
        
        return jsonify({
            "total_inspections": stats_data["total_inspections"],
            "accept_count": stats_data["accept_count"],
            "reject_count": stats_data["reject_count"],
            "review_count": stats_data["review_count"],
            "avg_inference_time_ms": stats_data["total_inference_time"] / total
        })
    
    @app.route('/system/status')
    def system_status():
        """Return live system status for dashboard indicators."""
        stats_data = app.config['stats']
        total = max(stats_data["total_inspections"], 1)
        avg_latency = stats_data["total_inference_time"] / total
        
        # Determine system status based on metrics
        if two_stage_engine is None:
            status = "offline"
        elif avg_latency > 2000:
            status = "degraded"
        else:
            status = "operational"
        
        return jsonify({
            "status": status,
            "avg_latency_ms": round(avg_latency, 1),
            "queue_depth": stats_data.get("pending_inspections", 0)
        })
    
    @app.route('/api/dashboard/summary')
    def dashboard_summary():
        """Return real-time dashboard summary for live metrics."""
        stats_data = app.config['stats']
        total = max(stats_data["total_inspections"], 1)
        
        # Get inspection history for timeline
        history = app.config.get('inspection_history', {})
        timeline_data = []
        
        # Generate timeline from recent inspections (last 10)
        from datetime import datetime
        current_time = datetime.now()
        for i in range(10):
            hour = (current_time.hour - i) % 24
            minute = current_time.minute
            time_str = f"{hour:02d}:{minute:02d}"
            # Count inspections in this time window (simplified)
            count = len([k for k in history if time_str in str(k)]) if history else 0
            timeline_data.insert(0, {"time": time_str, "count": count})
        
        # Calculate average confidence
        confidences = stats_data.get("confidences", [])
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return jsonify({
            "total_inspected": stats_data["total_inspections"],
            "accepted": stats_data["accept_count"],
            "rejected": stats_data["reject_count"],
            "review_required": stats_data["review_count"],
            "avg_confidence": round(avg_confidence * 100, 1),
            "timeline": timeline_data,
            "decision_distribution": {
                "accept": stats_data["accept_count"],
                "reject": stats_data["reject_count"],
                "review": stats_data["review_count"]
            }
        })
    
    @app.route('/api/audit/logs')
    def get_audit_logs():
        """Return audit logs for enterprise compliance tracking."""
        audit_logs = app.config.get('audit_logs', [])
        # Return logs in reverse chronological order (newest first)
        return jsonify({
            "logs": list(reversed(audit_logs)),
            "total_count": len(audit_logs)
        })
    
    @app.route('/analyze-image', methods=['POST'])
    def analyze_image():
        """Analyze an uploaded image for package damage using two-stage pipeline."""
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        package_id = request.form.get('package_id', f'PKG-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
        try:
            import time
            start_time = time.time()
            
            # Load image
            pil_img = Image.open(file.stream).convert('RGB')
            img_array = np.array(pil_img)
            width, height = pil_img.size
            
            # Get two-stage engine
            engine = app.config.get('two_stage_engine')
            recorder = app.config.get('evidence_recorder')
            
            if engine:
                # Real inference using two-stage pipeline
                decision, detections, reason = engine.infer_with_decision(img_array)
                
                # Calculate definitive severity
                severity_info = compute_severity(detections)
                severity_score = severity_info["severity_score"]
                severity_label = severity_info["severity_label"]
                risk_level = severity_info["risk_level"]
                
                # Create annotated image
                annotated = img_array.copy()
                detection_list = []
                
                for det in detections:
                    x1 = int(det.bbox["x1"] * width)
                    y1 = int(det.bbox["y1"] * height)
                    x2 = int(det.bbox["x2"] * width)
                    y2 = int(det.bbox["y2"] * height)
                    
                    # Color based on classification
                    if det.classifier_label == "damaged":
                        color = (255, 0, 0)  # Red for damaged
                        severity = "SEVERE" if det.classifier_confidence >= 0.85 else "MODERATE"
                    else:
                        color = (0, 255, 0)  # Green for intact
                        severity = "MINOR"
                    
                    # Draw rectangle
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label
                    label = f"{det.classifier_label} {det.classifier_confidence:.0%}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    detection_list.append({
                        "class_name": det.classifier_label,
                        "confidence": det.classifier_confidence,
                        "yolo_confidence": det.yolo_confidence,
                        "severity_level": severity,
                        "severity_score": det.classifier_confidence * 10,
                        "bbox": [x1, y1, x2, y2]
                    })
                
                # Record evidence
                inspection_id = f"INS-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                if recorder:
                    try:
                        record = recorder.record_inspection(
                            original_image=img_array,
                            annotated_image=annotated,
                            detections=detections,
                            decision=decision,
                            reason=reason,
                            package_id=package_id
                        )
                        inspection_id = record.get("inspection_id", inspection_id)
                    except Exception as e:
                        logger.warning(f"Evidence recording failed: {e}")
                
                # Convert annotated to base64
                annotated_pil = Image.fromarray(annotated)
                buffered = io.BytesIO()
                annotated_pil.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                inference_time = (time.time() - start_time) * 1000
                
            else:
                # Demo mode fallback (existing code)
                damage_types = [
                    ('structural_deformation', (255, 165, 0)),
                    ('surface_breach', (255, 0, 0)),
                    ('contamination_stain', (255, 0, 255)),
                    ('compression_damage', (255, 140, 0)),
                    ('tape_seal_damage', (255, 255, 0))
                ]
                severities = ['MINOR', 'MODERATE', 'SEVERE']
                
                is_damaged = random.random() > 0.3
                detection_list = []
                
                if is_damaged:
                    num_detections = random.randint(1, 3)
                    for i in range(num_detections):
                        damage_type, color = random.choice(damage_types)
                        x1 = random.randint(int(width * 0.1), int(width * 0.5))
                        y1 = random.randint(int(height * 0.1), int(height * 0.5))
                        x2 = min(x1 + random.randint(int(width * 0.15), int(width * 0.35)), width - 10)
                        y2 = min(y1 + random.randint(int(height * 0.15), int(height * 0.35)), height - 10)
                        
                        detection_list.append({
                            "class_name": damage_type,
                            "confidence": random.uniform(0.55, 0.95),
                            "severity_level": random.choice(severities),
                            "severity_score": random.uniform(1, 8),
                            "bbox": [x1, y1, x2, y2],
                            "color": color
                        })
                
                draw = ImageDraw.Draw(pil_img)
                for det in detection_list:
                    x1, y1, x2, y2 = det['bbox']
                    color = det.get('color', (255, 0, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    if 'color' in det:
                        del det['color']
                
                if not detection_list:
                    decision = 'ACCEPT'
                    reason = 'No damage detected'
                elif any(d['severity_level'] == 'SEVERE' for d in detection_list):
                    decision = 'REJECT'
                    reason = 'Severe damage detected'
                else:
                    decision = 'REVIEW_REQUIRED'
                    reason = 'Damage requires review'
                
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                inspection_id = f"INS-{uuid.uuid4().hex[:8].upper()}"
                inference_time = random.uniform(30, 80)
            
            # Update stats
            stats_data = app.config['stats']
            stats_data["total_inspections"] += 1
            stats_data["total_inference_time"] += inference_time
            if decision == "ACCEPT":
                stats_data["accept_count"] += 1
            elif decision == "REJECT":
                stats_data["reject_count"] += 1
            else:
                stats_data["review_count"] += 1
            
            # Track confidence for avg calculation
            if detections:
                max_conf = max(d.classifier_confidence for d in detections)
                stats_data["confidences"].append(max_conf)
            else:
                stats_data["confidences"].append(1.0)  # 100% confidence for no-detection case
            
            # Store for potential override
            app.config['inspection_history'][inspection_id] = {
                "decision": decision,
                "package_id": package_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add audit log entry for compliance tracking
            max_confidence = 100.0
            if detection_list:
                max_confidence = max(d.get('confidence', 1.0) for d in detection_list) * 100
            
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "package_id": package_id,
                "action": "INSPECTED",
                "decision": decision,
                "severity": severity_score if 'severity_score' in dir() else 0,
                "confidence": round(max_confidence, 1),
                "source": "AI",
                "inspection_id": inspection_id
            }
            app.config['audit_logs'].append(audit_entry)
            
            # Add DECISION_MADE entry
            decision_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "package_id": package_id,
                "action": "DECISION_MADE",
                "decision": decision,
                "severity": severity_score if 'severity_score' in dir() else 0,
                "confidence": round(max_confidence, 1),
                "source": "AI",
                "inspection_id": inspection_id
            }
            app.config['audit_logs'].append(decision_entry)
            
            result = {
                "inspection_id": inspection_id,
                "package_id": package_id,
                "timestamp": datetime.utcnow().isoformat(),
                "decision": {
                    "decision": decision,
                    "rationale": reason,
                    "max_severity": severity_label,
                    "severity_score": severity_score,
                    "severity_label": severity_label,
                    "risk_level": risk_level,
                    "total_detections": len(detection_list)
                },
                "detections": detection_list,
                "annotated_image": f"data:image/jpeg;base64,{img_base64}",
                "timing": {
                    "inference_ms": inference_time,
                    "total_ms": (time.time() - start_time) * 1000
                },
                "mode": "real" if engine else "demo"
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    @app.route('/inspect', methods=['POST'])
    def run_inspection():
        """Run package inspection."""
        package_id = request.args.get('package_id', '')
        
        service = app.config.get('inspection_service')
        
        if not service:
            # Demo mode - return simulated result
            import random
            from datetime import datetime
            
            decisions = ['ACCEPT', 'ACCEPT', 'ACCEPT', 'REJECT', 'REVIEW_REQUIRED']
            decision = random.choice(decisions)
            
            detections = []
            if decision != 'ACCEPT':
                damage_types = ['structural_deformation', 'surface_breach', 
                               'contamination_stain', 'compression_damage']
                severities = ['MINOR', 'MODERATE', 'SEVERE']
                
                for _ in range(random.randint(1, 3)):
                    detections.append({
                        "class_id": random.randint(0, 4),
                        "class_name": random.choice(damage_types),
                        "confidence": random.uniform(0.5, 0.95),
                        "bbox": [0.1, 0.1, 0.3, 0.3],
                        "severity_score": random.uniform(1, 8),
                        "severity_level": random.choice(severities)
                    })
            
            result = {
                "inspection_id": f"INS-DEMO-{datetime.now().strftime('%H%M%S')}",
                "package_id": package_id,
                "timestamp": datetime.utcnow().isoformat(),
                "decision": {
                    "decision": decision,
                    "rationale": "No damage detected" if decision == "ACCEPT" else "Damage detected",
                    "max_severity": "NONE" if not detections else detections[0]["severity_level"],
                    "total_detections": len(detections)
                },
                "detections": detections,
                "timing": {
                    "capture_ms": random.uniform(10, 30),
                    "inference_ms": random.uniform(30, 80),
                    "decision_ms": random.uniform(1, 5),
                    "evidence_ms": random.uniform(50, 100),
                    "total_ms": random.uniform(100, 200)
                }
            }
            
            # Update stats
            stats_data = app.config['stats']
            stats_data["total_inspections"] += 1
            if decision == "ACCEPT":
                stats_data["accept_count"] += 1
            elif decision == "REJECT":
                stats_data["reject_count"] += 1
            else:
                stats_data["review_count"] += 1
            stats_data["total_inference_time"] += result["timing"]["inference_ms"]
            
            return jsonify(result)
        
        # Real inspection
        try:
            result = service.inspect_package(package_id)
            
            # Format response
            detections = []
            for scored in result.decision.detections:
                det = scored.detection
                detections.append({
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": list(det.bbox),
                    "severity_score": scored.severity_score,
                    "severity_level": scored.severity_level.name
                })
            
            response = {
                "inspection_id": result.inspection_id,
                "package_id": result.package_id,
                "timestamp": result.timestamp.isoformat(),
                "decision": {
                    "decision": result.decision.final_decision.name,
                    "rationale": result.decision.rationale,
                    "max_severity": result.decision.max_severity.name,
                    "total_detections": result.decision.total_detections
                },
                "detections": detections,
                "timing": {
                    "capture_ms": result.capture_time_ms,
                    "inference_ms": result.inference_time_ms,
                    "decision_ms": result.decision_time_ms,
                    "evidence_ms": result.evidence_time_ms,
                    "total_ms": result.total_time_ms
                }
            }
            
            # Update stats
            stats_data = app.config['stats']
            stats_data["total_inspections"] += 1
            decision_name = result.decision.decision_type.name
            if decision_name == "ACCEPT":
                stats_data["accept_count"] += 1
            elif decision_name == "REJECT":
                stats_data["reject_count"] += 1
            else:
                stats_data["review_count"] += 1
            stats_data["total_inference_time"] += result.inference_time_ms
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Inspection failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/inspect/<inspection_id>/decision', methods=['POST'])
    def submit_decision(inspection_id):
        """Submit operator decision override."""
        data = request.json
        
        if not data or 'decision' not in data:
            return jsonify({"error": "Decision required"}), 400
        
        new_decision = data.get("decision")
        if new_decision not in ["ACCEPT", "REJECT", "REVIEW_REQUIRED"]:
            return jsonify({"error": "Invalid decision"}), 400
        
        # Get original inspection from history
        history = app.config.get('inspection_history', {})
        original = history.get(inspection_id, {})
        original_decision = original.get('decision', 'UNKNOWN')
        
        # Log the override
        logger.info(f"Operator override for {inspection_id}: {original_decision} → {new_decision}")
        
        # Record the override (could be extended to save to evidence)
        override_record = {
            "inspection_id": inspection_id,
            "original_decision": original_decision,
            "operator_decision": new_decision,
            "operator_id": data.get("operator_id", "OPERATOR"),
            "notes": data.get("notes", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update history
        if inspection_id in history:
            history[inspection_id]["operator_override"] = override_record
            history[inspection_id]["final_decision"] = new_decision
        
        # Update stats if decision changed
        stats_data = app.config['stats']
        if original_decision != new_decision:
            # Decrement old decision count
            if original_decision == "ACCEPT":
                stats_data["accept_count"] = max(0, stats_data["accept_count"] - 1)
            elif original_decision == "REJECT":
                stats_data["reject_count"] = max(0, stats_data["reject_count"] - 1)
            else:
                stats_data["review_count"] = max(0, stats_data["review_count"] - 1)
            
            # Increment new decision count
            if new_decision == "ACCEPT":
                stats_data["accept_count"] += 1
            elif new_decision == "REJECT":
                stats_data["reject_count"] += 1
            else:
                stats_data["review_count"] += 1
        
        # Add audit log entry for operator override
        override_audit = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "package_id": original.get("package_id", "UNKNOWN"),
            "action": "REVIEW_OVERRIDE",
            "decision": new_decision,
            "severity": 0,
            "confidence": 100.0,
            "source": "Manual",
            "inspection_id": inspection_id
        }
        app.config['audit_logs'].append(override_audit)
        
        return jsonify({
            "status": "success",
            "inspection_id": inspection_id,
            "original_decision": original_decision,
            "operator_decision": new_decision,
            "operator_id": data.get("operator_id", "OPERATOR"),
            "message": f"Decision overridden from {original_decision} to {new_decision}"
        })
    
    return app


def run_ui_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    config_path: str = "config/config.yaml",
    demo_mode: bool = True,
    debug: bool = False
):
    """
    Run the UI server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        config_path: Path to configuration file
        demo_mode: Use simulated inspections
        debug: Enable debug mode
    """
    inspection_service = None
    config = {}
    
    if not demo_mode:
        try:
            from ..utils.helpers import load_config
            from ..services.inspection_service import create_inspection_service
            
            config = load_config(config_path)
            inspection_service = create_inspection_service(config, simulated_cameras=False)
        except Exception as e:
            logger.warning(f"Could not initialize inspection service: {e}")
            logger.info("Running in demo mode")
    
    app = create_ui_app(inspection_service, config)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_ui_server(demo_mode=True, debug=True)
