# Edge vs Cloud Architecture Specification

## Package Damage Detection System

---

# 1. EDGE RESPONSIBILITIES (MANDATORY)

All real-time operations run on edge. No cloud dependency.

| Component | Location | Why Edge? |
|-----------|----------|-----------|
| **Camera capture** | Edge | Hardware-attached, microsecond timing |
| **Synchronization** | Edge | <5ms sync impossible over network |
| **Preprocessing** | Edge | Reduces data before inference |
| **AI inference** | Edge | <50ms latency required |
| **Multi-camera fusion** | Edge | Real-time decision depends on it |
| **Severity scoring** | Edge | Part of decision pipeline |
| **Decision logic** | Edge | Must be instant, cannot wait |
| **Evidence generation** | Edge | Images captured here |
| **Local storage** | Edge | Offline operation required |
| **Operator interface** | Edge | Dock workers use local display |
| **Alerts** | Edge | Audio/visual at station |

## Edge Device Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Compute | Jetson Orin Nano | Jetson Orin NX |
| RAM | 8GB | 16GB |
| Storage | 256GB NVMe | 512GB NVMe |
| Network | Optional WiFi/Ethernet | Gigabit Ethernet |

## Critical Principle

```
IF network is down:
    System operates at 100% capability
    All inspections complete normally
    All evidence stored locally
    Zero degradation in decision quality
```

---

# 2. CLOUD / BACKEND RESPONSIBILITIES (OPTIONAL)

Backend handles non-real-time operations only.

| Component | Location | Why Backend? |
|-----------|----------|--------------|
| **Model training** | Backend | GPU cluster, hours/days |
| **Model versioning** | Backend | Central registry |
| **Long-term storage** | Backend | Years of evidence, cheaper |
| **Analytics dashboard** | Backend | Aggregated fleet view |
| **Reporting** | Backend | Compliance reports |
| **Audit tools** | Backend | Cross-station analysis |
| **Alerting (fleet-wide)** | Backend | SMS/email to management |

## Why Not Required for Real-Time

| Component | Independence Reason |
|-----------|---------------------|
| Training | Models updated offline, pushed to edge |
| Long-term storage | Local stores 14 days; sync later |
| Analytics | Historical, not real-time |
| Reporting | Generated on-demand, not during inspection |
| Audit | Forensic, after the fact |

## Backend Optional = True

```
IF backend unreachable for 30 days:
    Edge continues normal operation
    Local storage fills (14 days retained)
    Oldest records auto-purged (configurable)
    System logs connectivity failure
    Alerts operator to sync when possible
```

---

# 3. DATA FLOW DESIGN

## Edge → Cloud (Upload)

| Data Type | Frequency | Priority | Size |
|-----------|-----------|----------|------|
| Evidence records | Hourly batch | Medium | ~5MB/inspection |
| Decision logs | Hourly batch | Medium | ~1KB/inspection |
| System health | Every 5 min | Low | ~1KB |
| Alert events | Immediate | High | ~2KB |

## Cloud → Edge (Download)

| Data Type | Frequency | Priority | Size |
|-----------|-----------|----------|------|
| Model updates | On-demand | Medium | ~50MB |
| Config updates | On-demand | Low | ~10KB |
| Threshold adjustments | On-demand | Low | ~1KB |

## Sync Timing

| Trigger | Action |
|---------|--------|
| Network becomes available | Queue sync, batch upload |
| Scheduled (hourly) | Check for pending uploads |
| Model update available | Download when idle |
| Manual request | Operator-triggered sync |

## Bandwidth Optimization

| Technique | Savings |
|-----------|---------|
| JPEG compression (quality 85) | ~70% vs raw |
| Upload only rejected/review cases | ~60% reduction |
| Differential config sync | ~95% reduction |
| Compressed JSON for logs | ~80% reduction |

## Typical Daily Bandwidth

| Scenario | Daily Upload |
|----------|--------------|
| 500 inspections, 20% flagged | ~500MB |
| 500 inspections, all uploaded | ~2.5GB |
| Health + logs only | ~10MB |

---

# 4. OFFLINE-FIRST GUARANTEES

## Zero-Network Behavior

| Function | Offline Capability |
|----------|-------------------|
| Inspect packages | ✅ Full |
| Make decisions | ✅ Full |
| Store evidence | ✅ Full (local) |
| Operator UI | ✅ Full |
| Alerts | ✅ Full (local) |
| Analytics | ❌ Not available |
| Fleet view | ❌ Not available |

## Local Retention Policy

| Storage | Capacity | Retention |
|---------|----------|-----------|
| Evidence images | 400GB | 14 days |
| Decision logs | 10GB | 90 days |
| System logs | 5GB | 30 days |

## Auto-Purge Rules

```
WHEN storage > 90% full:
    Delete evidence older than 14 days
    Keep decision logs (small)
    Alert operator about sync needed

WHEN storage > 95% full:
    Delete evidence older than 7 days
    Critical alert to operator
```

## Sync Resume Protocol

| Step | Action |
|------|--------|
| 1 | Detect network available |
| 2 | Authenticate with backend |
| 3 | Check last successful sync timestamp |
| 4 | Upload records newer than last sync |
| 5 | Confirm receipt (hash verification) |
| 6 | Mark records as synced (do not re-upload) |
| 7 | Download pending model/config updates |

## Idempotent Sync

```
Each record has unique inspection_id
Backend deduplicates on inspection_id
Re-upload same record = no duplicate
Safe to retry failed uploads
```

---

# 5. FAILURE & RECOVERY SCENARIOS

## Backend Unreachable

| Duration | System Behavior |
|----------|-----------------|
| 0-1 hour | Normal, queue uploads |
| 1-24 hours | Normal, warn operator |
| 1-7 days | Normal, alert management |
| 7-14 days | Normal, storage warning |
| 14+ days | Auto-purge oldest, continue |

**Impact on operations: NONE**

## Partial Upload Failure

| Failure | Recovery |
|---------|----------|
| Network drop mid-upload | Retry from last chunk |
| Server timeout | Exponential backoff retry |
| Authentication expired | Re-authenticate, retry |
| Partial batch uploaded | Resume from failed record |

**Tracking**: Each record has `sync_status: pending|uploaded|failed`

## Model Update Failure

| Failure | Recovery |
|---------|----------|
| Download interrupted | Resume download (range request) |
| Checksum mismatch | Re-download entire model |
| Model load fails | Rollback to previous version |
| No previous version | Continue with current model |

**Rollback Policy**:
```
Keep last 2 model versions on edge
New model tested on 10 cached images before activation
If accuracy drops >10%, auto-rollback
```

## Edge Device Reboot/Crash

| Phase | Recovery Action |
|-------|-----------------|
| Boot | Load last known good config |
| Camera init | Auto-detect connected cameras |
| Model load | Load from local cache |
| Evidence | Scan storage, rebuild index |
| Sync state | Load pending queue from disk |
| Resume | Ready in <60 seconds |

**Persistent State**:
- Evidence stored immediately to NVMe
- Decision logs flushed every 10 records
- Sync queue persisted to disk

---

# 6. SECURITY BOUNDARIES

## Trust Zones

```
┌─────────────────────────────────────────────────┐
│                 TRUSTED ZONE                    │
│  ┌─────────────────────────────────────────┐   │
│  │           EDGE DEVICE                    │   │
│  │  • All user data                         │   │
│  │  • All decisions                         │   │
│  │  • All evidence                          │   │
│  │  • Local operator access                 │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
            │
            │ Encrypted Channel (mTLS)
            ▼
┌─────────────────────────────────────────────────┐
│              SEMI-TRUSTED ZONE                  │
│  ┌─────────────────────────────────────────┐   │
│  │           BACKEND                        │   │
│  │  • Aggregated data only                  │   │
│  │  • Cannot modify edge decisions          │   │
│  │  • Can push model updates (signed)       │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Authentication

| Direction | Method |
|-----------|--------|
| Edge → Backend | Device certificate (mTLS) |
| Backend → Edge | Server certificate (TLS) |
| Operator → Edge | Local password/badge |
| Admin → Backend | SSO/MFA |

## Data Integrity

| Data | Integrity Mechanism |
|------|---------------------|
| Evidence records | SHA-256 hash chain |
| Uploads | TLS + content hash |
| Model files | Code-signed by vendor |
| Config updates | Signed manifest |

## What Backend CANNOT Do

| Action | Allowed? | Reason |
|--------|----------|--------|
| View evidence | ✅ | Authorized access |
| Modify past decisions | ❌ | Tamper-proof logs |
| Force real-time decision | ❌ | Edge is autonomous |
| Delete edge evidence | ❌ | Local retention policy |
| Downgrade model unsigned | ❌ | Signature required |

---

# 7. SCALABILITY CONSIDERATIONS

## Adding New Docks

| Step | Action |
|------|--------|
| 1 | Install edge device, cameras |
| 2 | Flash standard OS image |
| 3 | Configure station_id, network |
| 4 | Device registers with backend |
| 5 | Backend pushes latest model |
| 6 | Device operational |

**Time to deploy**: <2 hours per station

## Model Update Propagation

| Method | Description |
|--------|-------------|
| **Push** | Backend notifies devices of new version |
| **Pull** | Devices check for updates every 4 hours |
| **Staged rollout** | 10% → 50% → 100% over 3 days |
| **Rollback trigger** | Accuracy drop detected, auto-revert |

## Fleet Model Versions

```
All devices SHOULD run same model version
Acceptable drift: 1 version behind
Unacceptable: >2 versions behind (alert)
```

## Configuration Drift Prevention

| Mechanism | Purpose |
|-----------|---------|
| Central config template | Single source of truth |
| Config hash in health report | Detect drift |
| Auto-sync on drift | Pull correct config |
| Immutable base image | Reproducible deploys |
| Config versioning | Track changes |

## Monitoring at Scale

| Metric | Aggregation |
|--------|-------------|
| Inspections/hour | Per station, fleet total |
| Decision distribution | Per station, trend |
| Model accuracy | Per station, compare |
| Latency percentiles | Per station, flag outliers |
| Storage utilization | Per station, alert thresholds |

---

## Summary: Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                     EDGE DEVICE (per dock)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Cameras  │→│ Inference│→│ Decision │→│ Evidence │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│       ↓                                      ↓                 │
│  ┌──────────┐                          ┌──────────┐           │
│  │Operator UI│                         │Local Store│           │
│  └──────────┘                          └──────────┘           │
│                                              ↓                │
│                                    [Batch Sync Queue]          │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ When network available
                              │ (mTLS, compressed, batched)
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                         BACKEND                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Long-term│ │ Analytics│ │  Model   │ │  Fleet   │          │
│  │ Storage  │ │Dashboard │ │ Registry │ │ Monitor  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└────────────────────────────────────────────────────────────────┘
```

**Golden Rule**: Edge never waits for cloud. Cloud enhances, never blocks.
