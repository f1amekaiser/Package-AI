# Dataset & Annotation Strategy

## Edge-Based Intelligent Package Damage Detection System

---

# 1. DATASET COMPOSITION

## Recommended Dataset Size

| Stage | Total Images | Damaged | Clean | Notes |
|-------|-------------|---------|-------|-------|
| **MVP (Pilot)** | 5,000-8,000 | 2,500-4,000 | 2,500-4,000 | Minimum viable for deployment |
| **Production V1** | 10,000-15,000 | 5,000-7,500 | 5,000-7,500 | Robust production model |
| **Mature System** | 25,000+ | 12,000+ | 12,000+ | Continuous improvement |

## Class Distribution

| Class | Target % of Damaged Subset | Reasoning |
|-------|---------------------------|-----------|
| structural_deformation | 25-30% | Most common warehouse damage |
| surface_breach | 20-25% | High severity, must detect reliably |
| contamination_stain | 15-20% | Moderate frequency |
| compression_damage | 20-25% | Common in stacking scenarios |
| tape_seal_damage | 10-15% | Less frequent but security-critical |

## Handling Dataset Imbalance

| Problem | Solution |
|---------|----------|
| **Rare damage types** | Oversample rare classes during training (not in dataset) |
| **Clean vs damaged ratio** | Maintain ~1:1 ratio; do NOT over-represent clean |
| **Rare variations** | Targeted collection campaigns for specific patterns |
| **Class-wise evaluation** | Report per-class metrics, not just overall mAP |

## Realistic Collection Methods

| Method | Pros | Cons | Recommended Use |
|--------|------|------|-----------------|
| **Live dock capture** | Real conditions, real damage | Slow, damage is rare | 60% of dataset |
| **Staged damage** | Fast, controlled | May not match real patterns | 30% of dataset |
| **Historical claims** | Free data, real damage | Variable quality, old conditions | 10% of dataset |

### Collection Protocol

```
Week 1-2:  Install cameras, capture 2,000 clean package images
Week 3-4:  Begin staged damage (controlled, documented)
Week 5-8:  Live capture during normal operations
Week 9-10: Targeted collection for underrepresented classes
Week 11-12: Quality review, remove duplicates, finalize splits
```

### Staged Damage Guidelines

| Damage Type | How to Create Realistically |
|-------------|----------------------------|
| Dent | Apply controlled pressure with blunt object |
| Tear | Use blade to create clean cut, then tear naturally |
| Crush | Stack weight on corners, let compress naturally |
| Water | Spray water, let dry naturally (creates tide marks) |
| Tape damage | Peel tape partially, reseal poorly |

> **IMPORTANT**: Document all staged damage with metadata (type, severity, method). Never mix staged and real damage in same image.

---

# 2. DAMAGE TYPE COVERAGE

## structural_deformation (Dents, Bends)

| Metric | Requirement |
|--------|-------------|
| **Minimum samples** | 800-1,200 |
| **Visual variations** | Shallow dents, deep dents, corner bends, panel warping |
| **Common patterns** | Impact marks, forklift damage, drop damage |
| **Rare cases** | Very shallow dents (require side lighting), multi-dent clusters |

### Capture Guidance
- Capture with varying light angles (shadows reveal depth)
- Include both matte and glossy packaging surfaces
- Capture on brown, white, and printed cardboard

## surface_breach (Tears, Punctures)

| Metric | Requirement |
|--------|-------------|
| **Minimum samples** | 600-1,000 |
| **Visual variations** | Clean cuts, ragged tears, punctures, burst seams |
| **Common patterns** | Linear tears along edges, punctures from sharp objects |
| **Rare cases** | Hairline tears, tears hidden by tape, internal layer exposure |

### Capture Guidance
- Include tears of varying lengths (1cm to 20cm+)
- Capture both tape-covered and exposed tears
- Include color contrast (torn area vs interior material)

## contamination_stain (Water, Oil, Chemical)

| Metric | Requirement |
|--------|-------------|
| **Minimum samples** | 500-800 |
| **Visual variations** | Water marks, oil spots, chemical burns, mold |
| **Common patterns** | Dried tide marks, localized spots, edge discoloration |
| **Rare cases** | Clear water (no stain yet), very faint marks, stains matching printed patterns |

### Capture Guidance
- Capture at different drying stages (wet vs dried)
- Include various stain colors (brown water, black oil, yellow chemical)
- Capture on both plain and printed surfaces

## compression_damage (Crushed Corners)

| Metric | Requirement |
|--------|-------------|
| **Minimum samples** | 600-1,000 |
| **Visual variations** | Collapsed corners, accordion folds, buckled panels |
| **Common patterns** | Diagonal crush lines, stacking damage |
| **Rare cases** | Partial crush (one side only), distributed micro-crushing |

### Capture Guidance
- Capture from multiple angles (top view vs side view)
- Include severity gradient (slight push to full collapse)
- Capture both single and multiple crushed corners

## tape_seal_damage (Tampering, Wear)

| Metric | Requirement |
|--------|-------------|
| **Minimum samples** | 400-600 |
| **Visual variations** | Torn tape, peeling tape, missing tape, resealed tape |
| **Common patterns** | Edge lifting, cut tape, double-layer tape (retaping) |
| **Rare cases** | Security tape triggered, label damage, strap breaks |

### Capture Guidance
- Include different tape types (clear, brown, printed security)
- Capture signs of resealing (misalignment, bubbles, overlap)
- Include label damage as sub-variant

---

# 3. ANNOTATION GUIDELINES

## When to Label Damage

| Condition | Action |
|-----------|--------|
| Damage clearly visible in image | ✅ Label |
| Damage affects packaging integrity | ✅ Label |
| Damage visible from at least one angle | ✅ Label |
| Minor damage that could worsen | ✅ Label (as minor) |

## When NOT to Label Damage

| Condition | Action |
|-----------|--------|
| Normal manufacturing seams | ❌ Do not label |
| Intentional design features (perforations, vents) | ❌ Do not label |
| Dust or removable dirt | ❌ Do not label |
| Printed graphics that look like damage | ❌ Do not label |
| Very minor scuffs (normal handling) | ❌ Do not label |
| Damage outside the frame | ❌ Do not label |
| Heavily occluded damage (<20% visible) | ❌ Do not label |

## Ambiguous Case Guidelines

| Situation | Rule |
|-----------|------|
| **Not sure if damage or design** | Do NOT label; collect more examples |
| **Damage severity unclear** | Label conservatively (mark visible area only) |
| **Multiple damage types overlap** | Label dominant type; add second box if clearly distinct |
| **Very small damage (<1% of image)** | Label only if clearly visible; note size |
| **Motion blur on damage** | Do NOT label; exclude from dataset |

## Avoiding Over-Labeling

| Mistake | Correction |
|---------|------------|
| Labeling normal wear as damage | Define clear threshold (>5mm deviation) |
| Labeling shadows as dents | Verify with multiple images/angles |
| Multiple boxes for same damage | One box per damage instance |
| Box much larger than damage | Tight box around visible damage only |
| Labeling uncertain areas | When in doubt, leave it out |

## Bounding Box Guidelines

| Rule | Specification |
|------|---------------|
| **Tightness** | Box should tightly enclose visible damage |
| **Margin** | 2-5 pixels margin maximum |
| **Occlusion** | If damage partially occluded, box visible portion only |
| **Clustering** | Multiple nearby damages of same class → one box if <2cm apart |
| **Large damage** | Single damage spanning large area → one box |

---

# 4. BOUNDING BOX vs SEGMENTATION

## Recommendation: Bounding Boxes Only

| Damage Type | Annotation Type | Justification |
|-------------|-----------------|---------------|
| structural_deformation | **Bounding Box** | Irregular 3D shadows; exact boundary impractical |
| surface_breach | **Bounding Box** | Edge is clear but irregular; box sufficient for detection |
| contamination_stain | **Bounding Box** | Stains have fuzzy edges; pixel-perfect boundary impossible |
| compression_damage | **Bounding Box** | Complex geometry; box identifies affected region |
| tape_seal_damage | **Bounding Box** | Localized; box captures damaged seal area |

## Trade-off Analysis

| Factor | Bounding Box | Segmentation |
|--------|--------------|--------------|
| **Labeling time per image** | 10-15 seconds | 60-120 seconds |
| **Labeler skill required** | Low | High |
| **Inter-annotator agreement** | High (80%+) | Moderate (60-70%) |
| **Model training time** | Faster | Slower |
| **Inference speed** | Faster | Slower |
| **Decision value** | Sufficient | Marginal improvement |

## When Segmentation Might Be Needed (Future)

| Scenario | Reason |
|----------|--------|
| Insurance claim documentation | Precise damage area (cm²) calculation |
| Regulatory compliance | Exact contamination spread measurement |
| Severity grading V2 | Pixel-based severity instead of box-based |

**Recommendation**: Start with bounding boxes. Add segmentation only if business requires precise area measurement.

---

# 5. DATA AUGMENTATION STRATEGY

## Allowed Augmentations

| Augmentation | Parameters | Simulates |
|--------------|------------|-----------|
| **Horizontal flip** | 50% probability | Camera on either side |
| **Brightness shift** | ±20% | Lighting variation |
| **Contrast adjustment** | ±15% | Camera exposure differences |
| **Saturation shift** | ±20% | Color variation across cameras |
| **Blur (Gaussian)** | kernel 0-3px | Slight motion or focus issues |
| **Noise (Gaussian)** | σ = 5-15 | Sensor noise |
| **HSV hue shift** | ±5° | Color calibration differences |
| **Scale** | 0.8-1.2× | Package distance variation |
| **Translation** | ±10% | Package position variation |

## Augmentations to Avoid

| Augmentation | Why to Avoid |
|--------------|--------------|
| **Vertical flip** | Packages not upside-down in warehouse |
| **Rotation >10°** | Packages mostly axis-aligned |
| **Extreme color shifts** | Unrealistic; creates domain mismatch |
| **Heavy blur (>5px)** | Real system would discard such frames |
| **Cutout/erasing** | May erase damage, confusing model |
| **Mosaic (aggressive)** | Creates unrealistic package combinations |
| **MixUp** | Blends damage patterns unrealistically |

## Warehouse-Realistic Conditions

| Condition | How to Augment |
|-----------|----------------|
| **Morning vs evening light** | Brightness ±30%, color temp shift |
| **Dock door open/closed** | High contrast augmentation |
| **Shadow from forklift** | Synthetic shadow overlay (advanced) |
| **Dirty lens** | Edge vignetting, slight haze |
| **Conveyor vibration** | Mild motion blur (1-2px) |

## YOLOv5 Hyperparameter Settings

```yaml
# Recommended for package damage detection
hsv_h: 0.010   # Minimal hue shift (consistent lighting)
hsv_s: 0.50    # Moderate saturation variation
hsv_v: 0.30    # Moderate brightness variation
degrees: 5.0   # Minimal rotation
translate: 0.1 # Slight translation
scale: 0.3     # Moderate scale variation
shear: 2.0     # Minimal shear
flipud: 0.0    # NO vertical flip
fliplr: 0.5    # Horizontal flip allowed
mosaic: 0.8    # Mosaic augmentation
mixup: 0.1     # Minimal mixup
```

---

# 6. TRAIN / VALIDATION / TEST SPLIT

## Split Ratios

| Split | Percentage | Purpose |
|-------|------------|---------|
| **Training** | 70% | Model learning |
| **Validation** | 20% | Hyperparameter tuning, early stopping |
| **Test** | 10% | Final evaluation (never touch during dev) |

## Splitting Rules to Avoid Data Leakage

### Rule 1: Camera-Based Splitting

```
If 5 cameras: 
  - Train: Cameras 1, 2, 3
  - Val: Camera 4
  - Test: Camera 5
```

**Why**: Same package visible from multiple cameras creates leakage if different views are in train vs val.

### Rule 2: Time-Based Splitting

```
If continuous collection over 4 weeks:
  - Train: Week 1-2
  - Val: Week 3
  - Test: Week 4
```

**Why**: Model sees future patterns if time not separated.

### Rule 3: Package-Based Splitting

```
Each package_id appears in ONLY ONE split
Split by package, not by image
```

**Why**: Same package from different angles = leakage.

### Rule 4: Staged vs Real Separation

```
Staged damage: Max 40% of any split
Real damage: Min 60% of val and test
```

**Why**: Validate on real-world patterns, not artificial ones.

## Recommended Approach

| Priority | Method |
|----------|--------|
| 1 | Split by package_id (no package in multiple splits) |
| 2 | Ensure class balance across all splits |
| 3 | Ensure staged/real balance across splits |
| 4 | Ensure camera diversity in each split |

---

# 7. ROLE OF ANOMALY DETECTION

## Purpose

Anomaly detection is a **safety net** for damage types not in the training set.

| Detection | Anomaly Detection |
|-----------|-------------------|
| Finds known damage | Finds unknown damage |
| Fast inference | Slower inference |
| Classified output | Binary: normal/abnormal |
| Primary system | Secondary/backup system |

## When to Use

| Scenario | AD Role |
|----------|---------|
| Detection confidence < 0.50 and AD flags anomaly | Escalate to REVIEW_REQUIRED |
| New damage type appears (first time) | AD catches it, system updates |
| Model drift (performance drops) | AD provides fallback |

## Training Data for AD

| Data | Use |
|------|-----|
| Clean packages only | Train AD on "normal" |
| No damaged packages | AD learns normal distribution |
| Maximum variation | Include all package types, lighting conditions |

| Requirement | Amount |
|-------------|--------|
| **Clean samples for AD** | 5,000-10,000 (more is better) |
| **Package type coverage** | All types received at dock |
| **Lighting coverage** | All shifts, seasons |

## Integration Logic

```
IF detection_confidence ≥ 0.70:
    → Use detection result (ignore AD)
    
ELIF detection_confidence < 0.70 AND ad_score > threshold:
    → REVIEW_REQUIRED (AD triggered)
    
ELIF detection_confidence < 0.70 AND ad_score ≤ threshold:
    → ACCEPT (no detection, no anomaly)
```

## Phase Plan

| Phase | AD Status |
|-------|-----------|
| Phase 1 (MVP) | Not implemented; focus on detection |
| Phase 2 | AD runs in shadow mode (logging only) |
| Phase 3 | AD integrated into decision logic |

---

# 8. COMMON DATASET FAILURES TO AVOID

## Label Noise Risks

| Issue | Impact | Prevention |
|-------|--------|------------|
| Inconsistent labelers | Model confusion | Clear guidelines, double-labeling |
| Missed damage | False negatives | Quality review, audit samples |
| Wrong class | Class confusion | Clear class definitions, examples |
| Sloppy boxes | Poor localization | Bounding box review, automated checks |
| Unlabeled damage in "clean" | Model learns wrong | Careful clean verification |

## Bias Issues

| Bias | How It Happens | Prevention |
|------|----------------|------------|
| **Package type bias** | Only one brand in training | Collect from multiple suppliers |
| **Lighting bias** | All images from same time of day | Collect across shifts |
| **Camera bias** | All images from one camera | Use all cameras equally |
| **Size bias** | Only large packages | Include small, medium, large |
| **Color bias** | Only brown cardboard | Include white, printed, plastic |

## Domain Shift Issues

| Shift | Cause | Prevention |
|-------|-------|------------|
| **Camera change** | New camera installed | Recollect validation set |
| **Lighting change** | Dock renovation | Augment for lighting variation |
| **Supplier change** | New packaging type | Flag unknown packages |
| **Seasonal** | Humidity affects stains | Collect year-round |

## Data Quality Checklist

- [ ] No duplicate images in dataset
- [ ] No near-duplicate images across splits
- [ ] No corrupted/unreadable images
- [ ] All images properly labeled
- [ ] Class distribution documented
- [ ] Staged vs real ratio documented
- [ ] Camera source documented
- [ ] Collection date documented
- [ ] Minimum 2 reviewers per batch
- [ ] Test set never touched during development

## Red Flags During Training

| Warning Sign | Likely Cause |
|--------------|--------------|
| Val accuracy >> train accuracy | Data leakage |
| Perfect accuracy early | Leakage or memorization |
| Large train/val performance gap | Overfitting or domain shift |
| One class consistently poor | Class imbalance or label noise |
| Performance drops on new data | Domain shift |

---

## Summary Table

| Aspect | Recommendation |
|--------|----------------|
| Dataset size | 8,000-15,000 images |
| Damaged:clean ratio | ~1:1 |
| Annotation type | Bounding boxes only |
| Splitting | By package_id, not by image |
| Augmentation | Conservative; match warehouse reality |
| Anomaly detection | Phase 2 addition; not MVP |
| Quality control | Double-labeling, audit reviews |
