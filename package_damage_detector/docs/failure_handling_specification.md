# Failure Handling & Safety Mechanisms

## Package Damage Detection System

**Principle: Safety > Throughput. When uncertain, escalate.**

---

# 1. AI MODEL FAILURE MODES

## Low-Confidence Predictions

| Aspect | Specification |
|--------|---------------|
| **Detection** | Confidence < 0.50 on detected damage |
| **Impact** | Uncertain damage assessment |
| **Mitigation** | → REVIEW_REQUIRED (never auto-ACCEPT) |

## False Negatives on Severe Damage

| Aspect | Specification |
|--------|---------------|
| **Detection** | Model misses visible damage |
| **Impact** | Damaged package accepted |
| **Mitigation** | Multi-camera fusion (any camera detection counts), anomaly detection layer (future), operator spot-checks |

## Unexpected Input (New Packaging)

| Aspect | Specification |
|--------|---------------|
| **Detection** | High entropy in feature maps, unusual aspect ratios |
| **Impact** | Model performance unknown |
| **Mitigation** | Log anomaly, → REVIEW_REQUIRED, flag for retraining |

## Model Drift Over Time

| Aspect | Specification |
|--------|---------------|
| **Detection** | Accuracy metrics drop >5% over 7 days |
| **Impact** | Degraded detection quality |
| **Mitigation** | Weekly accuracy audit, alert on threshold breach, scheduled retraining |

### Model Drift Monitoring

| Metric | Threshold | Action |
|--------|-----------|--------|
| False negative rate | >2% increase | Alert supervisor |
| Confidence distribution shift | Significant | Flag for review |
| Override rate increase | >10% | Investigate model |

---

# 2. CAMERA & SENSOR FAILURES

## Single Camera Failure

| Aspect | Specification |
|--------|---------------|
| **Inspection proceeds?** | ✅ Yes |
| **Operator review?** | ❌ No (if 4+ cameras ok) |
| **System halt?** | ❌ No |
| **Logging** | Flag `degraded_coverage: true` |
| **Decision impact** | Normal logic, coverage gap logged |

## Multiple Camera Failures (2-3)

| Aspect | Specification |
|--------|---------------|
| **Inspection proceeds?** | ✅ Yes |
| **Operator review?** | ⚠️ Force REVIEW for any detection |
| **System halt?** | ❌ No |
| **Logging** | List failed cameras |
| **Decision impact** | Any detection → REVIEW_REQUIRED |

## Critical Camera Failures (4+)

| Aspect | Specification |
|--------|---------------|
| **Inspection proceeds?** | ❌ No |
| **Operator review?** | ✅ Manual inspection required |
| **System halt?** | ✅ Alert, await repair |
| **Logging** | CRITICAL alert |
| **Decision impact** | System cannot decide |

## Trigger Sensor Failure

| Symptom | Detection | Fallback |
|---------|-----------|----------|
| No triggers for 30s | Timeout | Alert operator, enable manual trigger |
| Continuous triggers | >10/min | Debounce lockout, alert |
| Trigger without package | Empty frame check | Ignore capture |

## Image Corruption or Blur

| Issue | Detection | Action |
|-------|-----------|--------|
| JPEG decode fails | Exception | Retry capture (1×), then skip camera |
| Motion blur | Edge sharpness < threshold | Retry (1×), proceed if still blurry |
| Overexposure | Histogram saturation | Log, adjust exposure, proceed |
| Black frame | Mean brightness < 10 | Mark camera failed |

---

# 3. EDGE DEVICE FAILURES

## GPU Overload

| Level | Detection | Action |
|-------|-----------|--------|
| Warning (>85%) | Utilization monitor | Reduce to 3 priority cameras |
| Critical (>95%) | Utilization monitor | Queue inspections (max 3) |
| Saturated | Queue full | REJECT all pending, alert |

## Memory Exhaustion

| Level | Detection | Action |
|-------|-----------|--------|
| Warning (>80%) | Memory monitor | Clear old buffers |
| Critical (>90%) | Memory monitor | Skip evidence images, log only |
| OOM imminent | Allocation fails | Graceful shutdown, restart |

## Process Crash

| Component | Detection | Recovery |
|-----------|-----------|----------|
| Inference process | Watchdog timeout | Auto-restart within 5s |
| Camera process | Heartbeat miss | Reconnect cameras |
| Storage process | Write fails | Buffer to memory, retry |
| Main orchestrator | Supervisor detects | Full service restart |

**Recovery time target**: <30s to operational state

## Power Loss / Reboot

| Phase | Action |
|-------|--------|
| Shutdown | Flush all pending writes |
| Boot | Load last known config |
| Model init | Load from local cache (<5s) |
| Camera init | Auto-detect, reconnect |
| Evidence recovery | Scan storage, rebuild index |
| Resume | Ready in <60s total |

### Data Integrity During Recovery

| Data | Protection |
|------|------------|
| Evidence images | Flushed to NVMe immediately |
| Decision logs | Flushed every 10 records |
| Hash chain | Append-only, recoverable |
| Pending sync queue | Persisted to disk |

---

# 4. NETWORK & BACKEND FAILURES

## Total Network Outage

| Aspect | Behavior |
|--------|----------|
| Inspections | ✅ Continue normally (100%) |
| Evidence storage | ✅ Local (14 days) |
| Sync | ❌ Queue for later |
| Model updates | ❌ Use current model |

## Sync Retry Logic

| Attempt | Delay | Action |
|---------|-------|--------|
| 1 | Immediate | First try |
| 2 | 30s | Retry |
| 3 | 2 min | Retry |
| 4 | 10 min | Retry |
| 5+ | 30 min | Exponential backoff (max 4h) |

## Partial Upload Handling

| Failure Point | Recovery |
|---------------|----------|
| Image upload interrupted | Resume from byte offset |
| Metadata rejected | Log error, retry with fixes |
| Batch partial success | Continue from failed record |
| Duplicate uploaded | Backend deduplicates |

## Prolonged Disconnection Alerts

| Duration | Alert Level | Action |
|----------|-------------|--------|
| 1 hour | Info | Log |
| 24 hours | Warning | Notify operator |
| 7 days | Alert | Email management |
| 14 days | Critical | Storage pressure warning |

---

# 5. OPERATOR INTERACTION FAILURES

## No Response to REVIEW_REQUIRED

| Timeout | Action |
|---------|--------|
| 30s | Audio alert |
| 60s | Screen flash |
| 120s | Default to REJECT |
| - | Log as `operator_timeout` |

**Default**: REJECT (safe for liability)

## Repeated Overrides

| Pattern | Detection | Action |
|---------|-----------|--------|
| >5 overrides/shift | Counter | Supervisor notification |
| >20% override rate | Stats | Weekly audit flag |
| Fast decisions (<3s avg) | Timer | Training review |
| Same notes repeated | Text compare | Flag rubber-stamping |

## Misuse / Bypass Attempts

| Attempt | Prevention |
|---------|------------|
| Skip inspection | Trigger lockout until complete |
| Modify decision | Records immutable after seal |
| Access without auth | Badge/PIN required |
| Bulk approve | One-at-a-time only |

### Operator Accountability

| Every Decision Logs |
|---------------------|
| Operator ID |
| Auth method used |
| Time to decide |
| Override notes (mandatory) |
| Station ID |

---

# 6. FAIL-SAFE DECISION POLICY

## Core Rule

```
WHEN system is uncertain:
  → NEVER auto-ACCEPT
  → Escalate to REVIEW_REQUIRED or REJECT
```

## Confidence-Based Defaults

| Confidence | Severity | Default Action |
|------------|----------|----------------|
| ≥0.85 | MAJOR | REJECT |
| ≥0.85 | MINOR | ACCEPT |
| 0.50-0.85 | Any | REVIEW_REQUIRED |
| <0.50 | MAJOR | REVIEW_REQUIRED |
| <0.50 | MINOR | ACCEPT (log) |
| No detection | - | ACCEPT |

## Forced REJECT Conditions

```
REJECT IF:
  - 4+ cameras failed (insufficient coverage)
  - Operator timeout on REVIEW_REQUIRED
  - Surface breach + corroborated + conf ≥0.90
  - Tape seal damage + corroborated + conf ≥0.90
  - System error during inspection
```

## Forced Manual Inspection

```
MANUAL INSPECTION IF:
  - All cameras failed
  - GPU/memory critical failure
  - Model not loaded
  - Trigger sensor failed
```

## Liability Minimization

| Decision | Liability Position |
|----------|-------------------|
| False ACCEPT | Warehouse liable for damage |
| False REJECT | Carrier dispute (defendable) |
| REVIEW timeout → REJECT | Conservative, documented |
| System error → REJECT | Safe default, logged |

**Conservative strategy**: False REJECT is preferable to False ACCEPT.

---

# 7. LOGGING & INCIDENT TRACEABILITY

## Logged Failure Events

| Event Type | Logged Data |
|------------|-------------|
| Camera failure | camera_id, failure_type, timestamp |
| Inference error | error_message, input_hash, stack_trace |
| Timeout | phase, expected_ms, actual_ms |
| Memory pressure | level, used_mb, available_mb |
| GPU overload | utilization_pct, action_taken |
| Operator timeout | inspection_id, wait_time, default_action |
| Sync failure | attempt_count, error, next_retry |
| Model anomaly | input_hash, confidence_distribution |

## Log Structure

```
{
  "timestamp": "2026-01-06T14:30:25.123Z",
  "level": "ERROR",
  "event_type": "CAMERA_FAILURE",
  "station_id": "DOCK-A-01",
  "inspection_id": "INS-...",
  "component": "camera_manager",
  "details": {
    "camera_id": "CAM-03",
    "failure_type": "CONNECTION_LOST",
    "retry_count": 3
  },
  "action_taken": "SKIPPED_CAMERA",
  "impact": "DEGRADED_COVERAGE"
}
```

## Incident Correlation

| Correlation Key | Purpose |
|-----------------|---------|
| `inspection_id` | Group all events for one inspection |
| `station_id` | Isolate station-specific issues |
| `timestamp` (±5s) | Correlate related events |
| `component` | Trace through pipeline |

## Failure Reconstruction

| Requirement | Implementation |
|-------------|----------------|
| Complete timeline | Events ordered by timestamp |
| Causal chain | Parent event IDs linked |
| System state | Periodic snapshots |
| Evidence | Images preserved |
| Reproducibility | Input data hashed |

## Audit Support

| Audit Query | Supported |
|-------------|-----------|
| "What happened to package X?" | ✅ Full inspection record |
| "Why was this rejected?" | ✅ Decision + evidence |
| "What failures occurred today?" | ✅ Event log filter |
| "Who overrode this decision?" | ✅ Operator audit trail |
| "How often does CAM-03 fail?" | ✅ Aggregate stats |

---

## Summary: Failure Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     FAILURE DETECTED                        │
└─────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌──────────────┐          ┌──────────────┐
       │ Recoverable  │          │ Critical     │
       └──────────────┘          └──────────────┘
              │                         │
              ▼                         ▼
       ┌──────────────┐          ┌──────────────┐
       │ Auto-recover │          │ Safe default │
       │ Log + Alert  │          │ REJECT/HALT  │
       │ Continue     │          │ Alert        │
       └──────────────┘          └──────────────┘
```

**Golden Rule**: If the system cannot make a confident, evidence-backed decision, it must escalate to human or default to REJECT. Never silently fail. Never auto-ACCEPT under uncertainty.
