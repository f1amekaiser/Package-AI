# Multi-Camera Fusion Logic Specification

## Edge-Based Package Damage Detection System

---

# 1. PACKAGE IDENTIFICATION & SYNCHRONIZATION

## Package Inspection Identity

| Element | Definition |
|---------|------------|
| **inspection_id** | `INS-{YYYYMMDD}-{HHMMSS}{ms}-{seq}` |
| **package_id** | Barcode scan or manual entry |
| **trigger_timestamp** | UTC timestamp when trigger activated |
| **station_id** | Fixed identifier for inspection station |

## Trigger Mechanism

| Component | Specification |
|-----------|---------------|
| **Primary trigger** | Photoelectric sensor at inspection zone entry |
| **Secondary trigger** | Conveyor encoder pulse (backup) |
| **Trigger debounce** | 500ms minimum between triggers |
| **Capture delay** | 50-100ms after trigger (package centering) |

## Synchronization Protocol

```
1. Photoelectric sensor breaks → trigger event generated
2. Controller broadcasts CAPTURE signal to all cameras
3. Each camera captures within 10ms window
4. All frames tagged with shared trigger_timestamp
5. Frames queued for inference
```

| Timing Requirement | Value |
|--------------------|-------|
| Camera-to-camera sync | ±5ms |
| Trigger-to-capture delay | <100ms |
| Total capture window | <150ms |

## Conveyor Assumptions

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Conveyor speed | 0.3-0.5 m/s | Typical warehouse speed |
| Package spacing | ≥1.0m | Prevents overlap in FOV |
| Inspection zone length | 0.8m | Package fully visible |
| Dwell time in zone | 1.6-2.6s | Time for full inspection |

## Package Overlap Prevention

| Rule | Implementation |
|------|----------------|
| Trigger lockout | Ignore triggers for 2s after capture |
| Zone clear check | Second sensor confirms package exit |
| Queue management | FIFO with max depth of 3 inspections |

---

# 2. PER-CAMERA INFERENCE PIPELINE

## Inference Trigger

| Event | Action |
|-------|--------|
| Frame received from camera | Add to inference queue |
| Queue depth > 0 | Dequeue and process |
| Processing mode | Sequential per camera, parallel across cameras |

## Preprocessing Steps

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Validate frame integrity | Reject corrupted data |
| 2 | Letterbox resize to 640×640 | Model input size |
| 3 | BGR → RGB conversion | Model expects RGB |
| 4 | Normalize to [0, 1] | Standard preprocessing |
| 5 | Convert to FP16 tensor | TensorRT optimization |

## Inference Execution

| Parameter | Value |
|-----------|-------|
| Model | YOLOv5s/m (TensorRT FP16) |
| Batch size | 1 (latency-optimized) |
| Input size | 640×640 |
| Expected latency | 15-30ms per frame |

## Per-Camera Outputs

| Field | Type | Description |
|-------|------|-------------|
| `camera_id` | string | Camera identifier |
| `frame_timestamp` | datetime | Capture time |
| `detections` | list | Detection objects |
| `inference_time_ms` | float | Processing time |
| `frame_quality` | enum | GOOD / BLUR / DARK |

### Detection Object

| Field | Type | Description |
|-------|------|-------------|
| `class_id` | int | 0-4 damage type |
| `class_name` | string | Damage type name |
| `confidence` | float | 0.0-1.0 |
| `bbox_normalized` | tuple | (x_center, y_center, w, h) |
| `bbox_pixels` | tuple | (x1, y1, x2, y2) |
| `area_ratio` | float | bbox area / image area |

## Local Confidence Filtering

| Threshold | Action |
|-----------|--------|
| confidence < 0.20 | Discard (noise) |
| confidence ≥ 0.20 | Pass to fusion |

**Note**: Low threshold intentional. Fusion layer decides final inclusion.

---

# 3. CROSS-CAMERA FUSION STRATEGY

## Primary Fusion Rule: OR-Based Union

```
IF any camera detects damage with confidence ≥ 0.50
   THEN include detection in final result
```

**Rationale**: Damage visible from one angle may not be visible from others. Missing damage (false negative) is worse than extra review (false positive).

## Detection Aggregation

| Step | Process |
|------|---------|
| 1 | Collect all detections from all cameras |
| 2 | Group by damage class |
| 3 | For each class, keep highest-confidence instance |
| 4 | Apply corroboration bonus |

## Corroboration Logic

| Condition | Corroboration Status | Effect |
|-----------|---------------------|--------|
| 1 camera detects | Uncorroborated | Normal weight |
| 2 cameras detect same class | Soft corroboration | +10% severity |
| 3+ cameras detect same class | Strong corroboration | +20% severity, auto-escalate |

## Low-Confidence Confirmation

| Condition | Resolution |
|-----------|------------|
| confidence 0.20-0.50, uncorroborated | Discard (likely FP) |
| confidence 0.20-0.50, corroborated | Promote to valid detection |
| confidence 0.50-0.70, uncorroborated | Include, mark uncertain |
| confidence ≥ 0.70, uncorroborated | Include, trust detection |

## Fusion Priority Matrix

| Priority | Damage Type | Reasoning |
|----------|-------------|-----------|
| 1 | surface_breach | Integrity compromised |
| 2 | tape_seal_damage | Tampering risk |
| 3 | contamination_stain | Product safety |
| 4 | compression_damage | Internal damage likely |
| 5 | structural_deformation | Often cosmetic |

## Duplicate Prevention

| Scenario | Rule |
|----------|------|
| Same class from multiple cameras | Use max confidence, count as 1 |
| Different classes from cameras | Count each class separately |
| Same class, multiple boxes, one camera | Count each box if IoU < 0.50 |

---

# 4. CONFLICT & AMBIGUITY HANDLING

## Single-Camera Detection

| Confidence | Camera Count | Action |
|------------|--------------|--------|
| ≥ 0.85 | 1 of 5 | Trust detection (high certainty) |
| 0.70-0.85 | 1 of 5 | Include, flag for review |
| 0.50-0.70 | 1 of 5 | Include, mark as uncertain |
| < 0.50 | 1 of 5 | Discard unless large area |

**Exception**: If single camera detects damage with area_ratio > 0.10, always include.

## Conflicting Damage Types

| Conflict | Resolution |
|----------|------------|
| Camera A: dent, Camera B: tear (same region) | Use higher-severity type (tear) |
| Camera A: stain, Camera B: compression | Both valid; different damage |
| Camera A: high conf dent, Camera B: low conf tear | Use higher-confidence detection |

**Resolution Rule**: When classes conflict for same visible surface, prefer:
1. Higher severity class
2. Higher confidence detection

## Partial Visibility / Occlusion

| Scenario | Handling |
|----------|----------|
| Package corner not in any FOV | Log coverage gap; proceed with available views |
| Damage at package edge (partial) | Accept detection if >30% of box visible |
| Camera blocked (operator hand) | Flag frame as invalid; use remaining cameras |

## Confidence Adjustment for Occlusion

| Visible Portion | Confidence Adjustment |
|-----------------|----------------------|
| >80% of bbox | No adjustment |
| 50-80% of bbox | Reduce confidence by 20% |
| 30-50% of bbox | Reduce confidence by 40% |
| <30% of bbox | Discard detection |

---

# 5. DECISION ESCALATION RULES

## Direct REJECT (Automatic)

```
REJECT IF:
  - surface_breach with confidence ≥ 0.90 AND corroborated
  - tape_seal_damage with confidence ≥ 0.90 AND corroborated
  - Any damage with severity_score ≥ 8.0 AND corroborated
  - 4+ distinct damage detections across package
```

## REVIEW_REQUIRED (Human Escalation)

```
REVIEW IF:
  - Any MAJOR severity damage (score ≥ 4.0)
  - surface_breach or tape_seal_damage with confidence 0.70-0.90
  - Conflicting detections across cameras
  - Single-camera detection with confidence 0.50-0.85
  - Uncorroborated damage with area_ratio > 0.10
  - Camera failure degraded inspection coverage
```

## Safe to ACCEPT

```
ACCEPT IF:
  - No detections with confidence ≥ 0.50
  - Only MINOR severity detections (score < 3.0)
  - All cameras operational, full coverage
  - No corroborated damage
```

## Escalation Priority

```
1. Check REJECT conditions → if any true → REJECT
2. Check REVIEW conditions → if any true → REVIEW
3. Otherwise → ACCEPT
```

---

# 6. PERFORMANCE CONSTRAINTS

## Latency Budget

| Phase | Target | Maximum |
|-------|--------|---------|
| Capture (all cameras) | 100ms | 150ms |
| Inference (per camera) | 30ms | 50ms |
| Fusion + decision | 20ms | 50ms |
| Evidence storage | 100ms | 200ms |
| **Total end-to-end** | **500ms** | **2000ms** |

## Camera Capacity

| Edge Device | Max Cameras | Inference Model |
|-------------|-------------|-----------------|
| Jetson Nano | 2 | YOLOv5n |
| Jetson Orin Nano | 4 | YOLOv5s |
| Jetson Orin NX | 6 | YOLOv5s/m |
| Jetson AGX Orin | 8 | YOLOv5m |

## Timeout Handling

| Condition | Action |
|-----------|--------|
| Inference > 50ms per camera | Log warning, continue |
| Total inspection > 2000ms | Force decision with available results |
| Total inspection > 5000ms | Abort, flag as SYSTEM_ERROR |

## Timeout Decision

```
IF timeout occurs:
  IF any REJECT-level detections found → REJECT
  ELSE → REVIEW (incomplete inspection)
```

---

# 7. FAILURE MODES & FALLBACKS

## Single Camera Failure

| Failure Type | Detection | Action |
|--------------|-----------|--------|
| Camera offline | No heartbeat | Mark camera unavailable |
| Frame corrupted | CRC/size check fails | Retry once, then skip |
| Inference error | Exception caught | Skip camera, log error |

**Decision Impact**: Continue with remaining cameras. Flag `degraded_coverage = true`.

## Multiple Camera Failure

| Cameras Down | Available | Action |
|--------------|-----------|--------|
| 1 of 5 | 4 | Normal operation, log gap |
| 2 of 5 | 3 | Proceed with warning, auto-REVIEW |
| 3 of 5 | 2 | Minimum viable, force REVIEW |
| 4+ of 5 | 1 or 0 | ABORT, human visual inspection required |

## Trigger Sensor Malfunction

| Symptom | Detection | Fallback |
|---------|-----------|----------|
| No trigger events | Timeout >30s | Alert operator, manual trigger |
| Rapid triggers | >10/minute | Debounce lockout |
| Trigger + no package | Frame similarity check | Ignore empty frame |

## Frame Quality Issues

| Issue | Detection | Action |
|-------|-----------|--------|
| Motion blur | Edge sharpness < threshold | Retry capture (1x) |
| Overexposure | Histogram saturation > 30% | Log, adjust exposure, continue |
| Underexposure | Mean brightness < 40 | Log, adjust exposure, continue |
| Complete black | Mean brightness < 10 | Mark camera failed |

## Graceful Degradation Summary

| Severity | System State | Decision Impact |
|----------|--------------|-----------------|
| Green | All cameras operational | Normal logic |
| Yellow | 1 camera down | Normal logic + coverage flag |
| Orange | 2 cameras down | Force REVIEW for any detection |
| Red | 3+ cameras down | ABORT, manual inspection |

---

# 8. OUTPUT CONTRACT

## Unified Inspection Result

```
InspectionResult:
  inspection_id: string
  package_id: string
  station_id: string
  timestamp_utc: datetime
  
  # Camera capture summary
  capture_summary:
    total_cameras: int
    cameras_captured: int
    cameras_failed: list[string]
    sync_quality: enum (EXCELLENT/GOOD/DEGRADED)
    
  # Per-camera results
  camera_results: list[CameraResult]
  
  # Fused detections
  fused_detections: list[FusedDetection]
  
  # Decision
  decision:
    outcome: enum (ACCEPT/REJECT/REVIEW_REQUIRED)
    severity: enum (NONE/MINOR/MAJOR)
    rationale: string
    confidence_level: enum (HIGH/MEDIUM/LOW)
    
  # Timing
  timing:
    capture_ms: float
    inference_total_ms: float
    fusion_ms: float
    decision_ms: float
    total_ms: float
    
  # Flags
  flags:
    degraded_coverage: bool
    timeout_occurred: bool
    manual_override_required: bool
```

## CameraResult Structure

```
CameraResult:
  camera_id: string
  position: string (TOP/FRONT/LEFT/RIGHT/BACK)
  capture_timestamp: datetime
  frame_quality: enum (GOOD/BLUR/DARK/INVALID)
  inference_time_ms: float
  detections: list[Detection]
```

## FusedDetection Structure

```
FusedDetection:
  detection_id: string
  class_id: int
  class_name: string
  max_confidence: float
  avg_confidence: float
  camera_sources: list[string]
  is_corroborated: bool
  corroboration_count: int
  severity_score: float
  severity_level: enum (MINOR/MAJOR)
  primary_camera: string
  bbox_primary: tuple
```

## Evidence Pipeline Contract

| Field | Required | Purpose |
|-------|----------|---------|
| All raw images | ✓ | Legal record |
| All annotated images | ✓ | Visual evidence |
| Detection list | ✓ | What was found |
| Fusion log | ✓ | How detections combined |
| Decision record | ✓ | Final outcome + reasoning |
| Timing breakdown | ✓ | Performance audit |
| Hash chain link | ✓ | Tamper detection |

## Operator Interface Contract

| Element | Content |
|---------|---------|
| Camera grid | 5 camera views with detection overlays |
| Decision banner | ACCEPT (green) / REJECT (red) / REVIEW (yellow) |
| Detection list | Class, confidence, camera source, severity |
| Action buttons | Override buttons for REVIEW cases |
| Timer | Countdown for operator decision (30s) |
| Package info | ID, timestamp, station |

---

## Summary: Fusion Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRIGGER EVENT                            │
│              (Photoelectric sensor)                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 SYNCHRONIZED CAPTURE                        │
│   CAM-01  CAM-02  CAM-03  CAM-04  CAM-05                   │
│     ↓       ↓       ↓       ↓       ↓                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            PARALLEL INFERENCE (per camera)                  │
│   [dets]   [dets]   [dets]   [dets]   [dets]               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FUSION LAYER                             │
│   • Collect all detections                                  │
│   • Apply corroboration                                     │
│   • Resolve conflicts                                       │
│   • Deduplicate                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DECISION ENGINE                            │
│   Severity scoring → Escalation rules → Final decision      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                   │
│   Decision + Evidence + Operator Display                    │
└─────────────────────────────────────────────────────────────┘
```
