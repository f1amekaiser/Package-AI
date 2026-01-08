# Package Damage Detection System

Edge-Based Intelligent Package Damage Detection for Warehouse Receiving Docks

## Overview

This system provides real-time, AI-powered detection of visible damage on sealed packages at warehouse receiving docks. It uses YOLOv5 object detection to identify damage types, makes automated accept/reject decisions, and stores tamper-proof photographic evidence.

## Features

- **Multi-camera inspection** - Synchronized capture from 2-6 cameras
- **Real-time detection** - <50ms inference per camera on edge hardware
- **5 damage classes** - Structural deformation, surface breach, contamination, compression, tape/seal damage
- **Severity scoring** - Automated severity calculation with configurable thresholds
- **Decision logic** - ACCEPT / REJECT / REVIEW_REQUIRED with operator override
- **Tamper-proof evidence** - SHA-256 hash chain for integrity verification
- **Offline operation** - Full functionality without network connectivity

## Project Structure

```
package_damage_detector/
├── config/
│   ├── config.yaml          # Main configuration
│   └── damage.yaml           # YOLO dataset config
├── data/
│   ├── images/               # Training images
│   ├── labels/               # YOLO format labels
│   └── hyps/                 # Training hyperparameters
├── src/
│   ├── core/
│   │   ├── inference_engine.py   # YOLOv5/TensorRT inference
│   │   ├── decision_engine.py    # Severity and decision logic
│   │   └── evidence_manager.py   # Tamper-proof storage
│   ├── services/
│   │   ├── camera_manager.py     # Multi-camera control
│   │   └── inspection_service.py # Orchestration
│   └── utils/
│       └── helpers.py            # Utility functions
├── models/                   # Trained model weights
├── evidence/                 # Stored inspection evidence
├── logs/                     # Application logs
├── main.py                   # Application entry point
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
cd package_damage_detector
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# From yolov5 directory
python train.py \
    --data ../package_damage_detector/config/damage.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16 \
    --hyp ../package_damage_detector/data/hyps/hyp.damage.yaml
```

### 3. Copy Trained Weights

```bash
cp runs/train/exp/weights/best.pt ../package_damage_detector/models/damage_detector.pt
```

### 4. Run Demo Mode

```bash
python main.py --demo --single
```

### 5. Run Continuous Mode

```bash
python main.py --demo --interval 3
```

## Configuration

Edit `config/config.yaml` to customize:

- **Model settings** - Weights path, confidence threshold, device
- **Camera settings** - Number of cameras, resolution, FPS
- **Decision rules** - Severity thresholds, auto-reject rules
- **Evidence storage** - Retention period, hash chain settings

## Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | structural_deformation | Dents, bent corners, warped panels |
| 1 | surface_breach | Tears, punctures, holes |
| 2 | contamination_stain | Water marks, oil stains, mold |
| 3 | compression_damage | Crushed corners, collapsed edges |
| 4 | tape_seal_damage | Torn tape, peeling labels |

## Decision Logic

```
Severity Score = Base Weight × Size Factor × Confidence Factor

SEVERE (score ≥ 6.0)   → REJECT
MODERATE (score ≥ 3.0) → REVIEW_REQUIRED
MINOR (score < 3.0)    → ACCEPT (unless multiple)
```

## API Usage

```python
from src.services.inspection_service import create_inspection_service
from src.utils.helpers import load_config

# Load config
config = load_config("config/config.yaml")

# Create service
service = create_inspection_service(config, simulated_cameras=True)

# Run inspection
result = service.inspect_package("PKG-001")

print(f"Decision: {result.decision.decision_type.name}")
print(f"Detections: {result.decision.total_detections}")
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Edge Device | Jetson Nano | Jetson Orin NX |
| RAM | 4GB | 16GB |
| Storage | 64GB SSD | 256GB NVMe |
| Cameras | 2× 2MP | 5× 5MP |

## License

Proprietary - Internal Use Only
