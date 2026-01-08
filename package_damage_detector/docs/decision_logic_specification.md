# Severity & Decision Logic Specification

## Edge-Based Package Damage Detection System

---

# 1. INPUT SIGNALS

## Primary Inputs (Per Detection)

| Input | Source | Value Range | Description |
|-------|--------|-------------|-------------|
| `damage_type` | Model output | 0-4 (class ID) | Type of damage detected |
| `confidence` | Model output | 0.0 - 1.0 | Model certainty |
| `bbox_area_ratio` | Computed | 0.0 - 1.0 | Detection area ÷ image area |
| `bbox_location` | Computed | corner/edge/center | Position on package |

## Aggregated Inputs (Per Package)

| Input | Source | Description |
|-------|--------|-------------|
| `detection_count` | Fusion | Total detections across cameras |
| `camera_count` | System | Number of cameras that captured |
| `cameras_with_damage` | Fusion | Cameras where damage detected |
| `max_confidence` | Fusion | Highest confidence among detections |
| `max_area_ratio` | Fusion | Largest detection area ratio |
| `damage_types_present` | Fusion | Set of unique damage types |

## Derived Inputs

| Input | Computation | Purpose |
|-------|-------------|---------|
| `is_corroborated` | Same class detected by ≥2 cameras | Multi-view confirmation |
| `location_penalty` | corner=1.5, edge=1.2, center=1.0 | Corners are structurally critical |
| `type_weight` | Per-class weight table | Inherent severity of damage type |

---

# 2. SEVERITY CLASSIFICATION

## Severity Levels

| Level | Code | Meaning |
|-------|------|---------|
| **NONE** | 0 | No damage detected |
| **MINOR** | 1 | Cosmetic damage, product likely unaffected |
| **MAJOR** | 2 | Significant damage, product may be compromised |

## Per-Detection Severity Score

```
severity_score = type_weight × size_factor × confidence_factor × location_penalty
```

### Type Weights (Inherent Danger)

| Damage Type | Weight | Reasoning |
|-------------|--------|-----------|
| structural_deformation | 2 | Often cosmetic, box structure intact |
| surface_breach | 4 | Packaging integrity broken, contamination risk |
| contamination_stain | 3 | Unknown substance, product exposure risk |
| compression_damage | 3 | May indicate product damage inside |
| tape_seal_damage | 4 | Potential tampering, security concern |

### Size Factor (Damage Extent)

| Condition | Factor | Reasoning |
|-----------|--------|-----------|
| bbox_area_ratio ≥ 0.15 | 2.0 | Large damage (>15% of visible area) |
| bbox_area_ratio ≥ 0.05 | 1.5 | Medium damage (5-15%) |
| bbox_area_ratio ≥ 0.02 | 1.0 | Small damage (2-5%) |
| bbox_area_ratio < 0.02 | 0.5 | Tiny damage (<2%) |

### Confidence Factor (Detection Certainty)

| Condition | Factor | Reasoning |
|-----------|--------|-----------|
| confidence ≥ 0.85 | 1.2 | High certainty, trust fully |
| confidence ≥ 0.70 | 1.0 | Good certainty, normal weight |
| confidence ≥ 0.50 | 0.8 | Moderate certainty, reduce weight |
| confidence < 0.50 | 0.5 | Low certainty, heavily discount |

### Location Penalty (Structural Importance)

| Location | Penalty | Reasoning |
|----------|---------|-----------|
| Corner | 1.5 | Corners are load-bearing |
| Edge | 1.2 | Edges provide structural support |
| Center/Surface | 1.0 | Less structural impact |

## Classification Rules

### NONE (No Damage)
```
IF detection_count = 0
   OR all detections have confidence < 0.25
THEN severity = NONE
```

### MINOR Damage
```
IF max severity_score < 4.0
   AND no surface_breach with confidence ≥ 0.70
   AND no tape_seal_damage with confidence ≥ 0.70
   AND detection_count ≤ 3
THEN severity = MINOR
```

### MAJOR Damage
```
IF any severity_score ≥ 4.0
   OR surface_breach with confidence ≥ 0.70
   OR tape_seal_damage with confidence ≥ 0.70
   OR detection_count > 3 (clustered damage)
   OR contamination_stain with area_ratio ≥ 0.10
THEN severity = MAJOR
```

---

# 3. ACCEPT / REJECT DECISION LOGIC

## Decision Outcomes

| Decision | Meaning | Operator Action |
|----------|---------|-----------------|
| **ACCEPT** | Package cleared | Sign delivery receipt |
| **REJECT** | Package refused | Do not sign; return to carrier |
| **REVIEW** | Human review needed | Operator inspects and decides |

## Decision Matrix

| Severity | Corroborated? | Max Confidence | Decision |
|----------|---------------|----------------|----------|
| NONE | - | - | ACCEPT |
| MINOR | No | < 0.70 | ACCEPT |
| MINOR | No | ≥ 0.70 | ACCEPT |
| MINOR | Yes | Any | ACCEPT (log) |
| MAJOR | No | < 0.70 | REVIEW |
| MAJOR | No | ≥ 0.70 | REVIEW |
| MAJOR | Yes | < 0.85 | REVIEW |
| MAJOR | Yes | ≥ 0.85 | REJECT |

## Decision Rules (Deterministic)

### Auto-ACCEPT
```
IF severity = NONE
   THEN → ACCEPT
   Reason: "No damage detected"

IF severity = MINOR
   AND max_severity_score < 3.0
   THEN → ACCEPT
   Reason: "Minor cosmetic damage only"
```

### Auto-REJECT
```
IF severity = MAJOR
   AND is_corroborated = TRUE
   AND max_confidence ≥ 0.85
   THEN → REJECT
   Reason: "Confirmed major damage"

IF surface_breach detected
   AND confidence ≥ 0.90
   AND is_corroborated = TRUE
   THEN → REJECT
   Reason: "Confirmed packaging breach"

IF tape_seal_damage detected
   AND confidence ≥ 0.90
   AND is_corroborated = TRUE
   THEN → REJECT
   Reason: "Confirmed seal tampering"
```

### REVIEW (Human Decision)
```
ALL OTHER CASES → REVIEW
   Reason: "Damage detected but requires human verification"
```

## Liability Justification

| Decision | Justification |
|----------|---------------|
| Auto-ACCEPT | No detection or minor cosmetic damage that does not affect product |
| Auto-REJECT | High-confidence, multi-camera confirmed major damage is indefensible to accept |
| REVIEW | Uncertain cases require human judgment; system provides evidence |

---

# 4. CONFLICT RESOLUTION

## Scenario: Conflicting Camera Detections

| Situation | Resolution |
|-----------|------------|
| Camera A: damage detected, Camera B: no damage | Trust Camera A; damage may be angle-specific |
| Camera A: tear, Camera B: dent (same area) | Label as highest-severity type |
| Camera A: high conf, Camera B: low conf | Use highest confidence |

**Rule**: Union approach — if ANY camera sees damage, it counts.

## Scenario: Low Confidence + Large Area

| Condition | Resolution |
|-----------|------------|
| confidence < 0.50 AND area_ratio > 0.10 | → REVIEW |
| confidence < 0.50 AND area_ratio < 0.05 | → Discount (likely false positive) |

**Reasoning**: Large uncertain detections warrant human verification; small uncertain ones are noise.

## Scenario: High Confidence + Very Small Area

| Condition | Resolution |
|-----------|------------|
| confidence ≥ 0.85 AND area_ratio < 0.01 | Log detection, classify as MINOR |
| confidence ≥ 0.85 AND area_ratio < 0.005 | Ignore (likely artifact) |

**Reasoning**: Very small high-confidence detections may be valid but are typically cosmetic.

## Scenario: Corroboration Mismatch

| Condition | Resolution |
|-----------|------------|
| 1 camera detects, 4 cameras do not | → REVIEW (possible angle-specific damage) |
| 3+ cameras detect same class | → Strong corroboration, trust detection |

---

# 5. FALSE POSITIVE / FALSE NEGATIVE CONTROL

## Error Types

| Error | Definition | Warehouse Impact |
|-------|------------|------------------|
| **False Positive** | System reports damage that isn't there | Unnecessary rejection, carrier dispute |
| **False Negative** | System misses real damage | Damaged goods accepted, liability |

## Which Error is More Dangerous?

**FALSE NEGATIVE is more dangerous**.

| Reason | Impact |
|--------|--------|
| Liability | Accepting damaged goods = warehouse responsibility |
| Legal | Cannot claim carrier damage if signed for |
| Insurance | No evidence to support claim |
| Product | Damaged product reaches customer |

## Threshold Tuning Strategy

| Goal | Implementation |
|------|----------------|
| Reduce false negatives | Use lower confidence threshold (0.25 for detection) |
| Control false positives | Use REVIEW tier for uncertain cases |
| Balance | High-confidence damage → REJECT; medium → REVIEW; low → log only |

## Confidence Threshold Settings

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `detection_threshold` | 0.25 | Capture all potential damage |
| `review_threshold` | 0.50 | Trigger human review |
| `reject_threshold` | 0.85 | Automatic rejection |

## Behavior Under Uncertainty

| Uncertainty Level | Behavior |
|-------------------|----------|
| Very uncertain (conf < 0.40) | Log but don't act; ACCEPT |
| Moderately uncertain (0.40-0.70) | → REVIEW |
| Reasonably certain (0.70-0.85) | → REVIEW (unless minor) |
| Highly certain (≥ 0.85) | Trust detection |

**Principle**: When in doubt, escalate to human. Never auto-accept uncertain major damage.

---

# 6. EXPLAINABILITY REQUIREMENTS

## Decision Explanation Structure

Every decision must include:

```
DECISION: [ACCEPT | REJECT | REVIEW]

SUMMARY:
  - Damage detected: [Yes/No]
  - Severity: [NONE | MINOR | MAJOR]
  - Confidence level: [Low | Medium | High]
  
DETECTIONS:
  1. [damage_type] on [camera_id]
     - Confidence: XX%
     - Size: X% of visible area
     - Location: [corner/edge/center]
     - Severity score: X.XX
     
REASONING:
  [Human-readable explanation of why this decision was made]
  
RECOMMENDATION:
  [For REVIEW cases: suggested action based on evidence]
```

## Operator Display Requirements

| Element | Required | Purpose |
|---------|----------|---------|
| Annotated images | ✓ | Show damage boxes on each camera view |
| Confidence bars | ✓ | Visual certainty indicator |
| Severity badge | ✓ | Color-coded MINOR/MAJOR |
| Decision rationale | ✓ | Plain-language explanation |
| Override buttons | ✓ | ACCEPT/REJECT for REVIEW cases |
| Timer | ✓ | Countdown for operator decision |

## Evidence Before Sign-Off

| Decision | Required Evidence |
|----------|-------------------|
| ACCEPT (no damage) | All camera images (unannotated) |
| ACCEPT (minor) | All camera images + detection annotations |
| REJECT | All camera images + annotations + severity breakdown |
| REVIEW→ACCEPT | Operator ID + notes mandatory |
| REVIEW→REJECT | Operator ID + notes mandatory |

## Audit Log Fields

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 UTC timestamp |
| `package_id` | Unique package identifier |
| `inspection_id` | Unique inspection identifier |
| `camera_captures` | List of image hashes |
| `detections` | Full detection list with scores |
| `severity_calculation` | Step-by-step score computation |
| `automated_decision` | System's initial decision |
| `final_decision` | Actual outcome (with override) |
| `operator_id` | Who made final decision (if human) |
| `operator_notes` | Free-text justification |
| `integrity_hash` | SHA-256 of entire record |

## Legal Defensibility

| Requirement | Implementation |
|-------------|----------------|
| Complete evidence | All images retained for 2+ years |
| Tamper-proof | Hash chain prevents retroactive modification |
| Traceable decisions | Every step logged with reasoning |
| Human accountability | Operator ID recorded for overrides |
| Time-stamped | Hardware RTC + NTP sync |

---

## Summary: Decision Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     MODEL OUTPUT                              │
│  [detections with class, confidence, bbox]                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  SEVERITY SCORING                            │
│  severity_score = type × size × confidence × location        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│               SEVERITY CLASSIFICATION                        │
│  NONE (score=0) | MINOR (score<4) | MAJOR (score≥4)         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 DECISION LOGIC                               │
│                                                              │
│  NONE → ACCEPT                                               │
│  MINOR → ACCEPT (log)                                        │
│  MAJOR + corroborated + high conf → REJECT                   │
│  All other MAJOR → REVIEW                                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│               EVIDENCE & LOGGING                             │
│  Images + detections + reasoning → tamper-proof storage      │
└──────────────────────────────────────────────────────────────┘
```
