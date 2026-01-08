# Evidence, Security & Audit Framework

## Package Damage Detection System

---

# 1. EVIDENCE REQUIREMENTS

## Per-Inspection Evidence Package

| Category | Item | Format | Required |
|----------|------|--------|----------|
| **Images** | Raw image per camera | JPEG (quality 95) | ✓ |
| **Images** | Annotated image per camera | JPEG (quality 85) | ✓ |
| **Images** | Composite view | JPEG | Optional |
| **Detections** | All detections (pre-fusion) | JSON | ✓ |
| **Detections** | Fused detections | JSON | ✓ |
| **Calculations** | Severity scores per detection | JSON | ✓ |
| **Decision** | Automated decision | JSON | ✓ |
| **Decision** | Operator override (if any) | JSON | Conditional |
| **Metadata** | Timestamps, IDs, station | JSON | ✓ |

## Evidence Record Structure

```
EvidenceRecord:
  inspection_id: "INS-20260106-143025123-0042"
  package_id: "PKG-2026010614302"
  station_id: "DOCK-A-01"
  
  timestamps:
    trigger: "2026-01-06T14:30:25.123Z"
    capture_complete: "2026-01-06T14:30:25.223Z"
    decision_made: "2026-01-06T14:30:25.512Z"
    evidence_stored: "2026-01-06T14:30:25.718Z"
    
  captures:
    - camera_id: "CAM-01"
      raw_image: "captures/CAM-01_raw.jpg"
      raw_hash: "sha256:a1b2c3..."
      annotated_image: "captures/CAM-01_annotated.jpg"
      annotated_hash: "sha256:d4e5f6..."
      
  detections:
    pre_fusion: [...]
    post_fusion: [...]
    
  severity:
    calculations: [...]
    max_score: 5.2
    level: "MAJOR"
    
  decision:
    automated: "REVIEW_REQUIRED"
    rationale: "Moderate damage detected"
    operator_override: "ACCEPT"
    operator_id: "OP-12345"
    operator_timestamp: "2026-01-06T14:30:42.000Z"
    operator_notes: "Damage is pre-existing label wear"
    
  integrity:
    content_hash: "sha256:..."
    previous_hash: "sha256:..."
    record_hash: "sha256:..."
```

---

# 2. EVIDENCE INTEGRITY & TAMPER RESISTANCE

## Image Protection

| Mechanism | Implementation |
|-----------|----------------|
| Immediate hashing | SHA-256 hash computed within 1s of capture |
| Read-only storage | Images written to append-only partition |
| No re-encoding | Original JPEG bytes preserved |
| Metadata embedded | EXIF contains inspection_id, timestamp |

## Hash Chain Design

```
Record N:
  content_hash = SHA256(images + detections + decision)
  previous_hash = record_hash of Record N-1
  record_hash = SHA256(content_hash + previous_hash)
```

| Property | Guarantee |
|----------|-----------|
| Forward integrity | Cannot insert record without breaking chain |
| Backward integrity | Cannot modify old record without breaking chain |
| Independent verification | Any party can verify with hashes |

## Tamper Detection

| Check | When | Alert |
|-------|------|-------|
| Hash verification | On every record access | Mismatch → TAMPER_ALERT |
| Chain continuity | Daily batch check | Gap → CHAIN_BROKEN |
| File integrity | On sync upload | Modified file → REJECT_UPLOAD |

## Immutability Rules

| Record State | Allowed Operations |
|--------------|-------------------|
| Open (inspection in progress) | Append only |
| Sealed (decision finalized) | Read only |
| Sealed + Synced | Read only, deletion after retention |

**Exception**: Operator override creates NEW record linked to original. Original never modified.

---

# 3. TIMESTAMPING & TRACEABILITY

## Timestamp Sources

| Priority | Source | Accuracy | Fallback Trigger |
|----------|--------|----------|------------------|
| 1 | NTP-synced RTC | ±10ms | - |
| 2 | Hardware RTC | ±1s/day drift | NTP unreachable |
| 3 | Boot-relative counter | Monotonic | RTC failure |

## Time Sync Protocol

```
On boot:
  IF network available:
    Sync with NTP (pool.ntp.org or internal)
    Set hardware RTC
  ELSE:
    Use hardware RTC
    Flag: time_sync = "RTC_ONLY"
    
Every 4 hours:
  IF network available:
    Re-sync NTP
    Correct drift
```

## Timezone Handling

| Field | Format | Timezone |
|-------|--------|----------|
| All stored timestamps | ISO 8601 | UTC |
| Display to operator | Local | Configured per station |
| Logs | ISO 8601 | UTC |

## Inspection Ordering

| Mechanism | Purpose |
|-----------|---------|
| Monotonic sequence number | Order within station |
| UTC timestamp | Order across stations |
| inspection_id format | Encodes date-time-sequence |

## Unique Identification

```
inspection_id = INS-{YYYYMMDD}-{HHMMSSmmm}-{seq}

Example: INS-20260106-143025123-0042
         │    │        │         │
         │    │        │         └─ Sequence (4 digits)
         │    │        └─ Time with milliseconds
         │    └─ Date
         └─ Prefix
```

**Collision probability**: Zero (sequence resets daily, ms precision)

---

# 4. ACCESS CONTROL & ROLE SEPARATION

## Role Definitions

| Role | Description | Scope |
|------|-------------|-------|
| Operator | Dock worker running inspections | Single station |
| Supervisor | Shift lead, reviews decisions | Station group |
| Auditor | Compliance review, read-only | All stations |
| Administrator | System configuration | All stations |

## Permission Matrix

| Action | Operator | Supervisor | Auditor | Admin |
|--------|----------|------------|---------|-------|
| Run inspection | ✓ | ✓ | ✗ | ✗ |
| View own evidence | ✓ | ✓ | ✓ | ✓ |
| View all evidence | ✗ | ✓ | ✓ | ✓ |
| Override decision | ✓* | ✓ | ✗ | ✗ |
| Add notes | ✓ | ✓ | ✓ | ✗ |
| Export evidence | ✗ | ✓ | ✓ | ✓ |
| Configure system | ✗ | ✗ | ✗ | ✓ |
| Delete records | ✗ | ✗ | ✗ | ✓** |

*Operator override only during inspection (30s window)
**Deletion only after retention period, requires justification

## Authentication

| Access Point | Method |
|--------------|--------|
| Edge UI (local) | Badge scan or PIN |
| Edge UI (Supervisor) | Badge + PIN |
| Backend dashboard | SSO with MFA |
| API access | Service account + certificate |

---

# 5. OPERATOR OVERRIDES & AUDIT TRAIL

## When Overrides Are Allowed

| System Decision | Override Allowed? | Time Window |
|-----------------|-------------------|-------------|
| ACCEPT | ✗ (no need) | - |
| REJECT | ✓ (to ACCEPT only) | 30 seconds |
| REVIEW_REQUIRED | ✓ (ACCEPT or REJECT) | 30 seconds |

## Override Record Requirements

| Field | Required | Description |
|-------|----------|-------------|
| original_decision | ✓ | What system decided |
| override_decision | ✓ | What operator chose |
| operator_id | ✓ | Who made override |
| operator_auth_method | ✓ | Badge, PIN, both |
| timestamp | ✓ | When override made |
| notes | ✓ | Mandatory justification |
| time_to_decide | ✓ | Seconds to make decision |

## Override Audit Log

```
OverrideEvent:
  inspection_id: "INS-..."
  original: "REJECT"
  override: "ACCEPT"
  operator: "OP-12345"
  auth: "BADGE+PIN"
  timestamp: "2026-01-06T14:30:42Z"
  notes: "Pre-existing damage documented on intake form"
  decision_time_seconds: 12.3
  supervisor_notified: true
```

## Post-Hoc Review

| Trigger | Review Action |
|---------|---------------|
| Any REJECT→ACCEPT override | Supervisor notification |
| Operator >5 overrides/shift | Supervisor alert |
| Override rate >20% for station | Weekly audit flag |
| Pattern of fast decisions (<5s) | Potential rubber-stamping |

## Abuse Detection

| Pattern | Indicator | Action |
|---------|-----------|--------|
| High override rate | >30% of decisions | Automatic escalation |
| Short decision times | Avg <3 seconds | Training review |
| Same operator, same pattern | Repeated override notes | Supervisor review |
| Override without notes | Empty justification | System prevents |

---

# 6. DATA RETENTION & STORAGE POLICY

## Retention Tiers

| Tier | Location | Duration | Contents |
|------|----------|----------|----------|
| Hot | Edge NVMe | 14 days | Everything |
| Warm | Backend storage | 2 years | Everything |
| Cold | Archive/tape | 7 years | Metadata + flagged images |
| Deleted | - | After cold | Per policy |

## Retention by Decision Type

| Decision | Edge Retention | Backend Retention |
|----------|----------------|-------------------|
| ACCEPT (clean) | 7 days | 1 year |
| ACCEPT (with detections) | 14 days | 2 years |
| REJECT | 14 days | 7 years |
| REVIEW (any outcome) | 14 days | 5 years |
| Disputed/claimed | Indefinite hold | 10 years |

## Auto-Purge Rules

```
Edge purge:
  IF storage > 85%:
    Delete ACCEPT records older than 7 days
  IF storage > 90%:
    Delete ACCEPT records older than 3 days
  IF storage > 95%:
    Alert supervisor, delete oldest synced records
    
NEVER auto-purge:
  - Unsynced records
  - REJECT decisions
  - Disputed records
```

## Compliance Considerations

| Regulation | Requirement | How Met |
|------------|-------------|---------|
| Record retention | Keep damage records | 2-7 year retention |
| Data integrity | Tamper-evident | Hash chain |
| Audit trail | Track access | All actions logged |
| Deletion rights | Right to delete | After legal hold period |

---

# 7. SYNC & BACKUP MECHANISMS

## Sync Protocol

```
1. Check network connectivity
2. Authenticate with backend
3. Query: last_sync_timestamp for this station
4. Collect: all records newer than last_sync
5. For each record:
   a. Verify local hashes
   b. Upload images (resumable)
   c. Upload metadata JSON
   d. Receive confirmation + backend record_id
   e. Mark local record as synced
6. Update last_sync_timestamp
```

## Partial Upload Handling

| Failure Point | Recovery |
|---------------|----------|
| Network drop during image | Resume from byte offset |
| Timeout on metadata | Retry with same inspection_id |
| Backend rejects (validation) | Log error, alert, retry later |
| Backend confirms | Mark synced, continue |

## Duplicate Prevention

| Mechanism | Purpose |
|-----------|---------|
| inspection_id as unique key | Backend deduplicates |
| Idempotent uploads | Re-upload = no duplicate |
| sync_status field | Track per-record state |

### Sync States

| State | Meaning |
|-------|---------|
| PENDING | Captured, not yet synced |
| UPLOADING | In progress |
| SYNCED | Backend confirmed |
| FAILED | Retry needed |

## Backup Failure Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Backend down | Connection timeout | Retry exponentially |
| Partial upload | Missing confirmation | Resume upload |
| Corrupted local file | Hash mismatch | Mark corrupt, alert |
| Full edge storage | Capacity monitoring | Prioritize sync, then purge |

## Backup Integrity Verification

```
Weekly:
  Backend selects 10 random records per station
  Edge re-computes hashes
  Compare with stored hashes
  Any mismatch → INTEGRITY_ALERT
```

---

# 8. THREAT MODEL & LIMITATIONS

## What This System Protects Against

| Threat | Protection |
|--------|------------|
| Accidental modification | Hash verification fails |
| Intentional tampering (post-capture) | Hash chain breaks |
| Unauthorized access | Role-based access control |
| Clock manipulation | NTP sync, drift detection |
| Evidence deletion | Retention policy, sync redundancy |
| Operator rubber-stamping | Override audit, pattern detection |
| Disputed claims | Complete evidence package |

## What This System Does NOT Protect Against

| Threat | Why Not Protected | Mitigation |
|--------|-------------------|------------|
| Malicious firmware | Full device compromise | Physical security, secure boot |
| Camera manipulation (physical) | Hardware tampering | Tamper-evident enclosures |
| Root access abuse | Admin can bypass anything | Separation of duties, logging |
| Collusion (operator + admin) | Social attack | External audit, rotation |
| Pre-capture staging | Damage faked before inspection | Out of scope |
| Sophisticated insider | Determined attacker | Accepts residual risk |

## Residual Risks (Accepted)

| Risk | Likelihood | Impact | Acceptance Rationale |
|------|------------|--------|---------------------|
| Hardware RTC drift | Medium | Low | ±1s/day acceptable for disputes |
| Image quality limits detection | Low | Medium | Model + human review |
| Network down during dispute | Low | Medium | Local evidence sufficient |
| Storage failure | Very Low | High | RAID + sync backup |

## Trust Assumptions

| Assumption | If Violated |
|------------|-------------|
| Edge device physically secure | All guarantees void |
| Operators act in good faith | Audit detects patterns |
| Cameras capture truthfully | System cannot detect |
| Network between edge-cloud secure | mTLS protects |

---

## Summary: Evidence Flow

```
┌───────────────────────────────────────────────────────────────┐
│                      CAPTURE                                  │
│   Images captured → Immediately hashed → Stored append-only   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                      PROCESSING                               │
│   Inference → Fusion → Severity → Decision                    │
│   (All steps logged with timestamps)                          │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                      SEALING                                  │
│   content_hash = SHA256(all_data)                            │
│   record_hash = SHA256(content_hash + previous_hash)         │
│   Append to chain → Immutable                                 │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                      SYNC (when available)                    │
│   Verify hashes → Upload → Confirm → Mark synced             │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                      RETENTION                                │
│   Edge: 14 days → Backend: 2-7 years → Archive → Delete      │
└───────────────────────────────────────────────────────────────┘
```

**Guiding Principle**: Every inspection creates a complete, self-contained, tamper-evident evidence package that can stand alone in any dispute.
