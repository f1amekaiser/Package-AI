"""
REST API Module

FastAPI-based REST API for external integration with the
Package Damage Detection System.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# =============================================================================
# API MODELS
# =============================================================================

class DetectionOut(BaseModel):
    """Detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(description="Normalized [x1, y1, x2, y2]")
    severity_score: float
    severity_level: str


class DecisionOut(BaseModel):
    """Decision result."""
    decision: str = Field(description="ACCEPT, REJECT, or REVIEW_REQUIRED")
    rationale: str
    max_severity: str
    total_detections: int


class InspectionResultOut(BaseModel):
    """Complete inspection result."""
    inspection_id: str
    package_id: str
    timestamp: str
    decision: DecisionOut
    detections: List[DetectionOut]
    timing: Dict[str, float]


class OperatorDecisionIn(BaseModel):
    """Operator decision input."""
    decision: str = Field(description="ACCEPT or REJECT")
    operator_id: str
    notes: Optional[str] = ""


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    cameras_status: Dict[str, bool]
    model_loaded: bool


class StatsResponse(BaseModel):
    """System statistics."""
    total_inspections: int
    accept_count: int
    reject_count: int
    review_count: int
    avg_inference_time_ms: float


# =============================================================================
# API APPLICATION
# =============================================================================

def create_api(
    inspection_service=None,
    config: Dict[str, Any] = None
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        inspection_service: InspectionService instance
        config: Configuration dictionary
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Package Damage Detection API",
        description="Edge-based intelligent package damage detection for warehouse receiving docks",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store service reference
    app.state.inspection_service = inspection_service
    app.state.config = config or {}
    app.state.stats = {
        "total_inspections": 0,
        "accept_count": 0,
        "reject_count": 0,
        "review_count": 0,
        "total_inference_time": 0.0
    }
    
    # -------------------------------------------------------------------------
    # HEALTH & INFO ENDPOINTS
    # -------------------------------------------------------------------------
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health status."""
        service = app.state.inspection_service
        
        cameras_status = {}
        model_loaded = False
        
        if service:
            cameras_status = service.camera_manager.get_camera_status()
            model_loaded = service.inference_engine.model is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            version="1.0.0",
            cameras_status=cameras_status,
            model_loaded=model_loaded
        )
    
    @app.get("/stats", response_model=StatsResponse, tags=["System"])
    async def get_stats():
        """Get system statistics."""
        stats = app.state.stats
        total = max(stats["total_inspections"], 1)
        
        return StatsResponse(
            total_inspections=stats["total_inspections"],
            accept_count=stats["accept_count"],
            reject_count=stats["reject_count"],
            review_count=stats["review_count"],
            avg_inference_time_ms=stats["total_inference_time"] / total
        )
    
    # -------------------------------------------------------------------------
    # INSPECTION ENDPOINTS
    # -------------------------------------------------------------------------
    
    @app.post("/inspect", response_model=InspectionResultOut, tags=["Inspection"])
    async def run_inspection(
        package_id: str = Query(..., description="Unique package identifier")
    ):
        """
        Run a full package inspection using all cameras.
        
        This is the primary endpoint for triggering an inspection.
        """
        service = app.state.inspection_service
        
        if not service:
            raise HTTPException(status_code=503, detail="Inspection service not available")
        
        try:
            result = service.inspect_package(package_id)
            
            # Update stats
            stats = app.state.stats
            stats["total_inspections"] += 1
            stats["total_inference_time"] += result.inference_time_ms
            
            decision_name = result.decision.decision_type.name
            if decision_name == "ACCEPT":
                stats["accept_count"] += 1
            elif decision_name == "REJECT":
                stats["reject_count"] += 1
            else:
                stats["review_count"] += 1
            
            return _format_result(result)
            
        except Exception as e:
            logger.error(f"Inspection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/inspect/image", response_model=InspectionResultOut, tags=["Inspection"])
    async def inspect_single_image(
        package_id: str = Query(..., description="Unique package identifier"),
        camera_id: str = Query("CAM-01", description="Camera identifier"),
        file: UploadFile = File(..., description="Image file to inspect")
    ):
        """
        Inspect a single uploaded image.
        
        Useful for testing or when images are captured externally.
        """
        service = app.state.inspection_service
        
        if not service:
            raise HTTPException(status_code=503, detail="Inspection service not available")
        
        try:
            # Read and decode image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Run inspection with single image
            result = service.inspect_with_images(
                package_id,
                {camera_id: image}
            )
            
            return _format_result(result)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Image inspection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/inspect/{inspection_id}/decision", tags=["Inspection"])
    async def submit_operator_decision(
        inspection_id: str,
        decision: OperatorDecisionIn
    ):
        """
        Submit operator decision for a REVIEW_REQUIRED inspection.
        """
        # In a full implementation, this would:
        # 1. Look up the inspection result
        # 2. Apply operator decision
        # 3. Update evidence record
        
        return {
            "status": "success",
            "inspection_id": inspection_id,
            "operator_decision": decision.decision,
            "operator_id": decision.operator_id
        }
    
    # -------------------------------------------------------------------------
    # EVIDENCE ENDPOINTS
    # -------------------------------------------------------------------------
    
    @app.get("/evidence/{inspection_id}", tags=["Evidence"])
    async def get_evidence(inspection_id: str):
        """
        Get evidence record for an inspection.
        """
        service = app.state.inspection_service
        
        if not service:
            raise HTTPException(status_code=503, detail="Service not available")
        
        record = service.evidence_manager.load_record(inspection_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        return record.to_dict()
    
    @app.get("/evidence/{inspection_id}/images/{camera_id}", tags=["Evidence"])
    async def get_evidence_image(
        inspection_id: str,
        camera_id: str,
        annotated: bool = Query(False, description="Get annotated image")
    ):
        """
        Get image from an evidence record.
        """
        service = app.state.inspection_service
        
        if not service:
            raise HTTPException(status_code=503, detail="Service not available")
        
        record = service.evidence_manager.load_record(inspection_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Evidence not found")
        
        # Find the capture for this camera
        suffix = "annotated" if annotated else "raw"
        
        for capture in record.captures:
            if capture.camera_id == camera_id and suffix in capture.image_path:
                image_path = Path(service.evidence_manager.storage_path) / capture.image_path
                
                if image_path.exists():
                    return FileResponse(image_path, media_type="image/jpeg")
        
        raise HTTPException(status_code=404, detail="Image not found")
    
    # -------------------------------------------------------------------------
    # HELPER FUNCTIONS
    # -------------------------------------------------------------------------
    
    def _format_result(result) -> InspectionResultOut:
        """Format inspection result for API response."""
        detections = []
        for scored in result.decision.detections:
            det = scored.detection
            detections.append(DetectionOut(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=list(det.bbox),
                severity_score=scored.severity_score,
                severity_level=scored.severity_level.name
            ))
        
        return InspectionResultOut(
            inspection_id=result.inspection_id,
            package_id=result.package_id,
            timestamp=result.timestamp.isoformat(),
            decision=DecisionOut(
                decision=result.decision.final_decision.name,
                rationale=result.decision.rationale,
                max_severity=result.decision.max_severity.name,
                total_detections=result.decision.total_detections
            ),
            detections=detections,
            timing={
                "capture_ms": result.capture_time_ms,
                "inference_ms": result.inference_time_ms,
                "decision_ms": result.decision_time_ms,
                "evidence_ms": result.evidence_time_ms,
                "total_ms": result.total_time_ms
            }
        )
    
    return app


# =============================================================================
# SERVER RUNNER
# =============================================================================

def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: str = "config/config.yaml",
    demo_mode: bool = False
):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        config_path: Path to configuration file
        demo_mode: Use simulated cameras
    """
    import uvicorn
    from ..utils.helpers import load_config
    from ..services.inspection_service import create_inspection_service
    
    # Load config
    config = load_config(config_path)
    
    # Create inspection service
    service = create_inspection_service(config, simulated_cameras=demo_mode)
    
    # Create API
    app = create_api(service, config)
    
    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server(demo_mode=True)
