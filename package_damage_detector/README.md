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
- **PostgreSQL database** - Production-ready with SQLite fallback for development
- **Web dashboard** - Real-time monitoring and inspection history
- **Offline operation** - Full functionality without network connectivity

## Project Structure

```
package_damage_detector/
├── config/
│   ├── config.yaml           # Main configuration
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
│   ├── db/
│   │   ├── connection.py         # PostgreSQL/SQLite connection
│   │   └── models.py             # SQLAlchemy models
│   ├── services/
│   │   ├── camera_manager.py     # Multi-camera control
│   │   └── inspection_service.py # Orchestration
│   ├── ui/
│   │   └── server.py             # Flask web dashboard
│   └── api/
│       └── routes.py             # FastAPI REST endpoints
├── models/                   # Trained model weights
├── evidence/                 # Stored inspection evidence
├── logs/                     # Application logs
├── main.py                   # Application entry point
├── setup_postgresql.py       # Database setup script
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
cd package_damage_detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Database Setup

#### Option A: PostgreSQL (Recommended for Production)

```bash
# Install PostgreSQL (macOS)
brew install postgresql@17
brew services start postgresql@17

# Create database and user
python setup_postgresql.py --create-db --init-tables
```

#### Option B: SQLite (Development)

No setup required - SQLite database is created automatically at `data/packageai.db`.

### 3. Run Web Dashboard

```bash
source venv/bin/activate
python -m src.ui.server
```

Open http://localhost:5000 in your browser.

### 4. Run Demo Mode

```bash
python main.py --demo --single
```

### 5. Run Continuous Mode

```bash
python main.py --demo --interval 3
```

## Database Configuration

The system supports both PostgreSQL (production) and SQLite (development).

### Environment Variables

```bash
# Full connection URL
export DATABASE_URL=postgresql://packageai:password@localhost:5432/packageai_db

# Or individual settings
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=packageai_db
export POSTGRES_USER=packageai
export POSTGRES_PASSWORD=your_password
```

### Configuration File

Edit `config/config.yaml`:

```yaml
database:
  type: "postgresql"  # or "sqlite"
  postgresql:
    host: "localhost"
    port: 5432
    database: "packageai_db"
    username: "packageai"
    password: "packageai_secure_password"
    pool:
      max_size: 10
      pool_timeout: 30
```

## Configuration

Edit `config/config.yaml` to customize:

- **Model settings** - Weights path, confidence threshold, device
- **Camera settings** - Number of cameras, resolution, FPS
- **Decision rules** - Severity thresholds, auto-reject rules
- **Database settings** - PostgreSQL or SQLite configuration
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

## API Endpoints

### REST API (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/health/db` | Database health check |
| GET | `/stats` | Inspection statistics |
| POST | `/inspect?package_id=XXX` | Run inspection |
| GET | `/evidence/{id}` | Get evidence record |

### Web Dashboard

| Route | Description |
|-------|-------------|
| `/` | Main dashboard with real-time stats |
| `/api/stats` | JSON statistics |
| `/api/history` | Inspection history |
| `/api/health` | Health status |

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
| Database | SQLite | PostgreSQL 17+ |

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
flake8 src/
mypy src/
```

### Database Migrations

```bash
# Check connection
python setup_postgresql.py --info

# Initialize tables
python setup_postgresql.py --init-tables

# Reset database (caution!)
python setup_postgresql.py --create-db --drop-existing
```

## License

Proprietary - Internal Use Only
