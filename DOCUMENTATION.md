# Package Damage Detection System

## Project Overview

Edge-based AI system for automated package damage detection using a two-stage inference pipeline.

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| Two-Stage Engine | `src/core/inference_engine.py` | YOLO detector + classifier pipeline |
| Evidence Recorder | `src/core/evidence_manager.py` | SHA-256 hashed immutable records |
| Flask Server | `src/ui/server.py` | Web UI with real inference |
| Decision Engine | `src/core/decision_engine.py` | Accept/Reject/Review logic |

## Models

| Model | Purpose | Classes |
|-------|---------|---------|
| `models/best.pt` | Detect damage regions | 1 (damage) |
| `models/damaged_classifier_best.pt` | Classify detected regions | 2 (damaged/intact) |

## Decision Thresholds

```
Classifier confidence ≥ 85% + "damaged" → REJECT
Classifier confidence ∈ [50%, 85%) + "damaged" → REVIEW_REQUIRED  
All detections "intact" → ACCEPT
No detections → ACCEPT
```

## Running the System

```bash
cd package_damage_detector
source venv/bin/activate
python -m src.ui.server
```

Server: http://localhost:5000

## API Endpoints

- `POST /analyze-image` — Upload and analyze image
- `POST /inspect/<id>/decision` — Operator override
- `GET /stats` — Session statistics
- `GET /health` — System health

## Evidence Storage

```
evidence/YYYY/MM/DD/INSPECTION_ID/
├── original.jpg (read-only)
├── annotated.jpg (read-only)
└── record.json (SHA-256 hashed)
```
