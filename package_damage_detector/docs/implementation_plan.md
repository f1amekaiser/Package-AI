# Edge-Based Intelligent Package Damage Detection System

## Complete Implementation Plan

---

# PART A: SYSTEM ARCHITECTURE

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WAREHOUSE RECEIVING DOCK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │Camera 1 │  │Camera 2 │  │Camera 3 │  │Camera 4 │  │Camera 5 │           │
│  │  (Top)  │  │ (Front) │  │ (Left)  │  │ (Right) │  │ (Back)  │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │            │                 │
│       └────────────┴─────┬──────┴────────────┴────────────┘                 │
│                          │                                                  │
│                          ▼                                                  │
│              ┌───────────────────────┐                                      │
│              │   EDGE COMPUTE NODE   │                                      │
│              │   (NVIDIA Jetson)     │                                      │
│              │                       │                                      │
│              │  • Inference Engine   │                                      │
│              │  • Decision Logic     │                                      │
│              │  • Evidence Storage   │                                      │
│              └───────────┬───────────┘                                      │
│                          │                                                  │
│         ┌────────────────┼────────────────┐                                 │
│         ▼                ▼                ▼                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │   DISPLAY   │  │   ALERTS    │  │   STORAGE   │                         │
│  │  (Operator) │  │(Light/Sound)│  │   (Local)   │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Network (When Available)
                                    ▼
                    ┌───────────────────────────────┐
                    │      BACKEND SERVER           │
                    │  • Long-term Evidence Storage │
                    │  • Analytics Dashboard        │
                    │  • Model Updates              │
                    │  • Audit Reports              │
                    └───────────────────────────────┘
```

## Hardware Specification

### Edge Compute Node

| Component | Specification | Justification |
|-----------|---------------|---------------|
| **Device** | NVIDIA Jetson Orin NX 16GB | Balance of performance, power, cost |
| **GPU** | 1024 CUDA cores, 32 Tensor cores | Handles 6 cameras at <50ms total |
| **CPU** | 8-core ARM Cortex-A78AE | Runs decision logic, I/O handling |
| **RAM** | 16GB LPDDR5 | Multiple camera buffers + model |
| **Storage** | 256GB NVMe SSD | 7+ days of evidence at high resolution |
| **Power** | 15-25W | Industrial power supply |
| **Form Factor** | Fanless enclosure | Dust, vibration resistant |

### Camera Array

| Spec | Requirement | Recommendation |
|------|-------------|----------------|
| **Resolution** | 2MP minimum | 5MP (2592×1944) preferred |
| **Frame Rate** | 15 FPS minimum | 30 FPS for motion handling |
| **Interface** | GigE Vision or USB3 | GigE for cable length flexibility |
| **Lens** | 6-12mm, low distortion | Fixed focus, industrial grade |
| **Lighting** | Built-in or co-located LED | Diffused panels, 5000K daylight |
| **Quantity** | 4-6 per inspection station | 5 recommended (top, 4 sides) |

### Recommended Camera Models

| Option | Model | Price Range | Notes |
|--------|-------|-------------|-------|
| Budget | FLIR Blackfly S (GigE) | $400-600 | Good industrial features |
| Mid-range | Basler ace 2 | $500-800 | Excellent image quality |
| Premium | Allied Vision Alvium | $600-1000 | Superior low-light |

### Network & Infrastructure

| Component | Specification |
|-----------|---------------|
| **Local Network** | Gigabit Ethernet switch (PoE+ for cameras) |
| **WAN Connection** | Optional; system operates fully offline |
| **UPS** | 30-minute battery backup minimum |
| **Enclosure** | IP54 rated for dock environment |

---

## Software Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Operator UI │  │  REST API    │  │  Admin Panel │          │
│  │  (Local Web) │  │  (Optional)  │  │  (Config)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                       SERVICE LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Inspection   │  │  Decision    │  │  Evidence    │          │
│  │ Orchestrator │  │  Engine      │  │  Manager     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Camera       │  │  Alert       │  │  Sync        │          │
│  │ Manager      │  │  Manager     │  │  Manager     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                      INFERENCE LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐          │
│  │              YOLOv5 TensorRT Engine              │          │
│  │         (5-class damage detection model)         │          │
│  └──────────────────────────────────────────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                       DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  SQLite DB   │  │  Image Store │  │  Config      │          │
│  │  (Records)   │  │  (Evidence)  │  │  (YAML)      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Software Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Inference Engine** | YOLOv5 + TensorRT | Damage detection |
| **Camera SDK** | Aravis / Spinnaker | Camera control |
| **Image Processing** | OpenCV | Preprocessing, annotation |
| **Database** | SQLite | Local inspection records |
| **API Framework** | FastAPI | REST endpoints (optional) |
| **UI Framework** | PyQt6 or Web (Flask) | Operator interface |
| **Evidence Hashing** | hashlib (SHA-256) | Tamper-proof verification |
| **Sync Service** | Python + requests | Backend communication |

---

# PART B: DATASET STRATEGY

## Data Collection Plan

### Phase 1: Initial Dataset (MVP)

| Category | Target Count | Source |
|----------|--------------|--------|
| **Clean packages** | 3,000 images | Warehouse operations |
| **structural_deformation** | 800+ images | Warehouse + staged |
| **surface_breach** | 600+ images | Warehouse + staged |
| **contamination_stain** | 500+ images | Warehouse + staged |
| **compression_damage** | 600+ images | Warehouse + staged |
| **tape_seal_damage** | 500+ images | Warehouse + staged |
| **TOTAL** | ~6,000 images | |

### Collection Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Live collection** | Capture at actual dock | Real conditions | Slow, damage is rare |
| **Staged damage** | Intentionally damage packages | Fast data generation | May not match real damage |
| **Historic photos** | Use existing claim photos | Free data | May be poor quality |
| **Synthetic augmentation** | Apply damage overlays | Massive scale | Risk of unrealistic patterns |

### Recommended Approach

```
Week 1-2:  Install cameras, collect 1,000 clean package images
Week 3-4:  Begin staged damage collection (controlled conditions)
Week 5-6:  Collect real damage examples during operations
Week 7-8:  Label dataset, perform quality review
Week 9-10: Train initial model, evaluate gaps
Week 11-12: Targeted collection to fill gaps
```

## Labeling Infrastructure

### Labeling Tool Options

| Tool | Type | Cost | Recommendation |
|------|------|------|----------------|
| **Label Studio** | Self-hosted, open source | Free | ✅ Recommended |
| **CVAT** | Self-hosted, open source | Free | Good alternative |
| **Roboflow** | Cloud-based | $249+/mo | Fast but ongoing cost |
| **Labelbox** | Enterprise cloud | $$$$ | Overkill for this |

### Labeling Workflow

```
1. CAPTURE
   Raw images from cameras
        ↓
2. TRIAGE
   Quick review: damage present? Y/N
        ↓
3. LABEL
   Apply bounding boxes + class labels
        ↓
4. REVIEW
   Second annotator verifies labels
        ↓
5. EXPORT
   Generate YOLO format dataset
        ↓
6. VERSION
   Store dataset version with hash
```

### Labeling Quality Rules

| Rule | Implementation |
|------|----------------|
| **Double-labeling** | 20% of images labeled by two people |
| **Disagreement resolution** | Senior annotator breaks ties |
| **Edge case documentation** | Screenshot and document ambiguous examples |
| **Class balance check** | Weekly review of class distribution |
| **Negative mining** | Include hard negatives (normal wear) |

## Dataset Split

| Split | Percentage | Purpose |
|-------|------------|---------|
| **Training** | 70% | Model learning |
| **Validation** | 20% | Hyperparameter tuning |
| **Test** | 10% | Final evaluation (never touched during development) |

### Split Rules

- No package appears in multiple splits
- Each split has similar class distribution
- Test set includes challenging examples intentionally

---

# PART C: MULTI-CAMERA PIPELINE

## Inspection Station Layout

```
                    ┌─────────────────┐
                    │   Camera 1      │
                    │   (TOP VIEW)    │
                    └────────┬────────┘
                             │
                             ▼
        ┌────────┐    ┌─────────────┐    ┌────────┐
        │Camera 3│    │             │    │Camera 4│
        │ (LEFT) │───▶│   PACKAGE   │◀───│(RIGHT) │
        └────────┘    │             │    └────────┘
                      └─────────────┘
                       ▲           ▲
                       │           │
               ┌───────┘           └───────┐
               │                           │
        ┌──────┴──────┐             ┌──────┴──────┐
        │  Camera 2   │             │  Camera 5   │
        │  (FRONT)    │             │   (BACK)    │
        └─────────────┘             └─────────────┘

        ══════════════════════════════════════════
                    CONVEYOR DIRECTION →
```

## Capture Synchronization

### Trigger Options

| Method | Description | Latency | Reliability |
|--------|-------------|---------|-------------|
| **Sensor trigger** | Photoelectric sensor detects package | <10ms | ✅ High |
| **Software trigger** | Motion detection in primary camera | 50-100ms | Medium |
| **Timed capture** | Fixed interval captures | N/A | Low (may miss) |

### Recommended: Sensor-Triggered Capture

```
1. Package enters inspection zone
2. Photoelectric sensor breaks
3. Trigger signal sent to all cameras (hardware trigger line)
4. All cameras capture simultaneously (<1ms skew)
5. Images queued for processing
6. Conveyor pauses OR continues based on decision
```

### Frame Buffer Management

```python
# Conceptual flow (not implementation)

CAMERA_COUNT = 5
BUFFER_DEPTH = 3  # frames per camera

inspection_buffer = {
    camera_id: RingBuffer(size=BUFFER_DEPTH)
    for camera_id in range(CAMERA_COUNT)
}

# On trigger:
# 1. Capture from all cameras
# 2. Each image gets same inspection_id
# 3. Process as batch
```

## Detection Fusion Strategy

### Per-Camera Processing

```
For each camera image:
    1. Preprocess (resize to 640×640, normalize)
    2. Run YOLOv5 inference
    3. Apply NMS (conf_threshold=0.25, iou_threshold=0.45)
    4. Record detections with camera_id
```

### Cross-Camera Fusion

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Union (OR)** | Any camera detects damage → flagged | Default approach |
| **Intersection (AND)** | Multiple cameras must agree | Only for high-FP classes |
| **Weighted vote** | Cameras have confidence weights | Future enhancement |

### Recommended: Union with Confirmation

```
IF any camera detects damage with conf ≥ 0.70:
    → Damage confirmed
    
IF any camera detects damage with 0.40 ≤ conf < 0.70:
    → Check other cameras for corroboration
    → If another camera also detects: confirmed
    → If no corroboration: REVIEW_REQUIRED

IF all cameras conf < 0.40 or no detections:
    → No damage detected
```

### Duplicate Suppression

Same damage may be visible from multiple cameras. Prevent double-counting:

| Rule | Implementation |
|------|----------------|
| **Same class, same surface** | Count as 1 instance |
| **Same class, different surface** | Count separately |
| **Different class, same location** | Count separately (different damage) |

---

# PART D: DECISION LOGIC

## Decision Flowchart

```
                    ┌─────────────────────┐
                    │  Multi-Camera       │
                    │  Detection Results  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Any Detection?     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │ NO                              │ YES
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────────┐
    │     ACCEPT      │              │  Calculate Severity │
    │                 │              │  for each detection │
    └─────────────────┘              └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │  Max Severity?      │
                                     └──────────┬──────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │ MINOR                     │ MODERATE                  │ SEVERE
                    ▼                           ▼                           ▼
          ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
          │     ACCEPT      │        │ REVIEW_REQUIRED │        │     REJECT      │
          │  (with record)  │        │                 │        │                 │
          └─────────────────┘        └─────────────────┘        └─────────────────┘
```

## Severity Calculation

### Per-Detection Severity

Severity is computed from:
1. **Damage class** (some classes are inherently more severe)
2. **Detection size** (larger damage = more severe)
3. **Confidence** (higher confidence = more certain)

### Class Severity Weights

| Class | Base Severity | Reasoning |
|-------|---------------|-----------|
| `structural_deformation` | 2 | Often cosmetic, product usually safe |
| `surface_breach` | 4 | Packaging integrity compromised |
| `contamination_stain` | 3 | Unknown substance exposure |
| `compression_damage` | 3 | Product may be damaged |
| `tape_seal_damage` | 4 | Potential tampering |

### Size Factor

```
size_ratio = (detection_area / image_area)

IF size_ratio > 0.15:    size_factor = 2.0  (Large)
ELIF size_ratio > 0.05:  size_factor = 1.5  (Medium)
ELIF size_ratio > 0.02:  size_factor = 1.0  (Small)
ELSE:                    size_factor = 0.5  (Tiny)
```

### Confidence Factor

```
IF confidence ≥ 0.85:    conf_factor = 1.2  (High certainty)
ELIF confidence ≥ 0.70:  conf_factor = 1.0  (Good certainty)
ELIF confidence ≥ 0.50:  conf_factor = 0.8  (Moderate certainty)
ELSE:                    conf_factor = 0.5  (Low certainty)
```

### Final Severity Score

```
severity_score = base_severity × size_factor × conf_factor

IF severity_score ≥ 6.0:   → SEVERE
ELIF severity_score ≥ 3.0: → MODERATE  
ELSE:                      → MINOR
```

## Decision Rules

| Package Severity | Multi-Damage? | Decision | Action |
|------------------|---------------|----------|--------|
| No damage | N/A | ACCEPT | Log + proceed |
| MINOR only | No | ACCEPT | Log + proceed |
| MINOR only | Yes (3+) | REVIEW_REQUIRED | Operator check |
| MODERATE | Any | REVIEW_REQUIRED | Operator check |
| SEVERE | Any | REJECT | Block + alert |

## Operator Review Interface

When REVIEW_REQUIRED:

```
┌─────────────────────────────────────────────────────────────┐
│                    REVIEW REQUIRED                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Package ID: PKG-2024-01-06-0042                           │
│  Inspection Time: 14:32:45                                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │        [Annotated image with damage boxes]          │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Detections:                                                │
│  • surface_breach (conf: 0.72) - Camera 2                  │
│  • structural_deformation (conf: 0.65) - Camera 3          │
│                                                             │
│  Calculated Severity: MODERATE                              │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   ACCEPT    │    │   REJECT    │    │    SKIP     │    │
│  │   (Green)   │    │    (Red)    │    │   (Yellow)  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
│  Timeout: 30 seconds (defaults to REJECT)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# PART E: EVIDENCE PIPELINE

## Evidence Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Tamper-proof** | Cryptographic hash of image + metadata |
| **Timestamped** | Hardware RTC + NTP sync when available |
| **Complete** | All camera views for every inspection |
| **Retrievable** | Fast lookup by package ID, date, decision |
| **Durable** | Local retention + cloud backup |

## Evidence Record Structure

```json
{
  "inspection_id": "INS-20240106-143245-001",
  "package_id": "PKG-2024-01-06-0042",
  "timestamp_utc": "2024-01-06T09:02:45.123Z",
  "timestamp_local": "2024-01-06T14:32:45.123+05:30",
  "station_id": "DOCK-A-01",
  
  "captures": [
    {
      "camera_id": "CAM-01-TOP",
      "image_path": "evidence/2024/01/06/INS-20240106-143245-001/cam01.jpg",
      "image_hash_sha256": "a1b2c3d4e5f6...",
      "resolution": [2592, 1944],
      "exposure_ms": 8.0
    },
    // ... cameras 2-5
  ],
  
  "detections": [
    {
      "camera_id": "CAM-02-FRONT",
      "class": "surface_breach",
      "confidence": 0.72,
      "bbox_normalized": [0.45, 0.32, 0.12, 0.08],
      "severity_score": 4.8,
      "severity_level": "MODERATE"
    }
  ],
  
  "decision": {
    "automated_decision": "REVIEW_REQUIRED",
    "final_decision": "ACCEPT",
    "decided_by": "operator_jsmith",
    "decision_timestamp": "2024-01-06T09:03:12.456Z",
    "notes": "Minor surface scratch, product intact"
  },
  
  "integrity": {
    "record_hash_sha256": "f6e5d4c3b2a1...",
    "signed_at": "2024-01-06T09:03:12.789Z"
  }
}
```

## Hash Chain (Tamper Evidence)

```
Record N:
  • content_hash = SHA256(all fields except integrity)
  • record_hash = SHA256(content_hash + previous_record_hash)
  
Record N+1:
  • content_hash = SHA256(...)
  • record_hash = SHA256(content_hash + Record_N.record_hash)
  
...continuing chain...
```

Any modification to a past record breaks the chain, making tampering detectable.

## Storage Architecture

```
LOCAL STORAGE (Edge Device)
├── evidence/
│   └── 2024/
│       └── 01/
│           └── 06/
│               ├── INS-20240106-143245-001/
│               │   ├── cam01.jpg
│               │   ├── cam02.jpg
│               │   ├── cam03.jpg
│               │   ├── cam04.jpg
│               │   ├── cam05.jpg
│               │   ├── annotated_composite.jpg
│               │   └── record.json
│               └── INS-20240106-143301-002/
│                   └── ...
└── db/
    └── inspections.sqlite
```

### Retention Policy

| Location | Retention | Purpose |
|----------|-----------|---------|
| **Edge device** | 7-14 days | Recent access, offline operation |
| **Backend server** | 1-2 years | Audit, legal, insurance claims |
| **Cold storage** | 7+ years | Regulatory compliance |

## Sync to Backend

```
When network available:
    1. Query unsync'd records from local DB
    2. For each record:
       a. Upload images to cloud storage (S3/GCS/Azure Blob)
       b. POST record JSON to backend API
       c. Verify upload success
       d. Mark record as synced locally
    3. If offline for >24 hours, alert admin
```

---

# PART F: IMPLEMENTATION ROADMAP

## Phase Overview

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Foundation** | 4 weeks | Hardware installed, cameras working, basic capture |
| **Phase 2: Dataset** | 4 weeks | Labeled dataset, initial model trained |
| **Phase 3: Core System** | 4 weeks | Detection + decision + evidence pipeline |
| **Phase 4: Integration** | 2 weeks | Operator UI, alerts, WMS integration |
| **Phase 5: Validation** | 2 weeks | Testing, tuning, pilot operation |

## Detailed Phase Breakdown

### Phase 1: Foundation (Weeks 1-4)

| Week | Tasks |
|------|-------|
| **Week 1** | Procure hardware (Jetson, cameras, networking) |
| **Week 2** | Physical installation at pilot dock station |
| **Week 3** | Camera configuration, lighting setup, image quality validation |
| **Week 4** | Basic capture software, storage setup, infrastructure testing |

**Exit Criteria:**
- [ ] All 5 cameras capturing synchronized images
- [ ] Images stored correctly on local SSD
- [ ] Network connectivity verified
- [ ] Lighting produces consistent image quality

### Phase 2: Dataset (Weeks 5-8)

| Week | Tasks |
|------|-------|
| **Week 5** | Deploy Label Studio, begin clean package collection |
| **Week 6** | Staged damage collection, begin labeling |
| **Week 7** | Continue labeling, quality review process |
| **Week 8** | Dataset finalization, split creation, version control |

**Exit Criteria:**
- [ ] 5,000+ labeled images
- [ ] All 5 classes represented with 500+ examples each
- [ ] Train/val/test splits created
- [ ] Dataset versioned and backed up

### Phase 3: Core System (Weeks 9-12)

| Week | Tasks |
|------|-------|
| **Week 9** | Train YOLOv5 model, export to TensorRT |
| **Week 10** | Implement multi-camera inference pipeline |
| **Week 11** | Implement decision engine, severity calculation |
| **Week 12** | Implement evidence pipeline, hash chain |

**Exit Criteria:**
- [ ] Model achieves >85% mAP on test set
- [ ] Inference <50ms per frame on Jetson
- [ ] Decision logic produces correct outputs
- [ ] Evidence records are complete and verifiable

### Phase 4: Integration (Weeks 13-14)

| Week | Tasks |
|------|-------|
| **Week 13** | Operator UI development, alert integration |
| **Week 14** | Backend sync, WMS integration (if applicable) |

**Exit Criteria:**
- [ ] Operator can view and respond to REVIEW_REQUIRED
- [ ] Alerts trigger correctly
- [ ] Records sync to backend when connected

### Phase 5: Validation (Weeks 15-16)

| Week | Tasks |
|------|-------|
| **Week 15** | Pilot operation (shadow mode - decisions logged but not enforced) |
| **Week 16** | Performance tuning, threshold adjustment, go-live |

**Exit Criteria:**
- [ ] <5% false reject rate
- [ ] <2% false accept rate (for severe damage)
- [ ] Operators trained and comfortable
- [ ] System runs 24h without errors

---

## Verification Plan

### Model Verification

| Test | Method | Pass Criteria |
|------|--------|---------------|
| **Accuracy** | Evaluate on held-out test set | mAP@0.5 > 0.85 |
| **Per-class performance** | Class-wise precision/recall | Each class P&R > 0.80 |
| **Speed** | Benchmark on target hardware | <50ms per 640×640 image |
| **Edge cases** | Curated hard examples | Human review of failures |

### System Verification

| Test | Method | Pass Criteria |
|------|--------|---------------|
| **Multi-camera sync** | Measure capture timestamp skew | <10ms between cameras |
| **Decision accuracy** | Compare to human labels on 500 packages | >95% agreement |
| **Evidence integrity** | Verify hash chain after 1000 records | All hashes valid |
| **Offline operation** | Disconnect network for 24h | All functions work |
| **Recovery** | Kill process, verify restart | Auto-restart, no data loss |

### Operational Verification (Pilot)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Throughput** | ≥20 packages/minute | Time study |
| **False reject rate** | <5% | Operator overrides |
| **False accept rate** | <2% (severe damage) | Post-inspection audit |
| **Operator satisfaction** | Positive | Survey |
| **System uptime** | >99% | Monitoring logs |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient damage data** | Medium | High | Staged damage collection + augmentation |
| **Model underperforms** | Medium | High | Iterative training, threshold tuning |
| **Hardware failure** | Low | High | Spare unit on-site, auto-restart |
| **Operator resistance** | Medium | Medium | Training, shadow mode first |
| **Lighting changes** | Medium | Medium | Dedicated lighting, train on variations |
| **New damage types** | Certain | Medium | Quarterly model updates |

---

## Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| **Damage detection rate** | 0% (no system) | 90% | 95% |
| **Processing time** | N/A | <5 seconds | <3 seconds |
| **False accept (severe)** | Unknown | <2% | <1% |
| **False reject** | Unknown | <5% | <3% |
| **Evidence availability** | 0% | 100% | 100% |
| **System uptime** | N/A | 99% | 99.9% |

---

## Next Steps

1. **Hardware procurement** — Order Jetson Orin NX, cameras, networking
2. **Site preparation** — Plan physical installation at pilot dock
3. **Team assignment** — Identify roles (ML engineer, integration, operators)
4. **Labeling setup** — Deploy Label Studio, train annotators
5. **Kickoff** — Begin Phase 1

---

*Document Version: 1.0*  
*Created: 2026-01-06*  
*Status: Pending Review*
