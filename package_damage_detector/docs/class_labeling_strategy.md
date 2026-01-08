# Detection Class & Labeling Strategy

## Edge-Based Intelligent Package Damage Detection System

---

## 1. FINAL DETECTION CLASSES

### Primary Classes (Phase 1)

| Class ID | Class Name | Visual Pattern | Examples |
|----------|------------|----------------|----------|
| **0** | `structural_deformation` | 3D shape distortion visible as shadows, irregular edges, bent surfaces | Dents, bent corners, warped panels, pushed-in surfaces |
| **1** | `surface_breach` | Broken packaging integrity exposing inner layers or contents | Tears, punctures, holes, rips, slashes, burst seams |
| **2** | `contamination_stain` | Discoloration or residue on packaging surface | Water stains, oil marks, chemical spills, mold/mildew marks |
| **3** | `compression_damage` | Crushed or flattened structures with visible material failure | Crushed corners, collapsed edges, accordion folds, buckling |
| **4** | `tape_seal_damage` | Compromised closure mechanisms or sealing elements | Torn tape, missing tape, resealed tape, peeling labels, broken straps |

### Why 5 Classes

| Rationale | Explanation |
|-----------|-------------|
| **Minimal** | Fewer classes = faster training, less labeling confusion, better model convergence |
| **Complete** | Covers all common visible damage patterns in warehouse receiving |
| **Visually distinct** | Each class has different visual signatures; low inter-class confusion |
| **Actionable** | Each class maps to real damage that affects accept/reject decisions |

---

## 2. WHAT SHOULD NOT BE A CLASS

### ❌ Do NOT Train These as Classes

| Concept | Why NOT a Class | How to Handle |
|---------|-----------------|---------------|
| **Damage severity** (minor/moderate/severe) | Subjective; varies by package type; creates labeling inconsistency | Compute post-detection from box size relative to package |
| **Accept / Reject / Review** | Business decision, not visual pattern | Apply decision logic after detection |
| **Confidence threshold** | Model output, not trainable class | Use model's native confidence scores |
| **Package type** (box, envelope, pallet) | Not damage; separate classification task | Either ignore or use separate model |
| **"No damage" / "Clean"** | Absence is not a class; detect presence only | No detection = no damage |
| **Brand / Supplier** | Not damage-related | Out of scope |
| **Contents description** | Cannot see inside sealed packages | Out of scope |
| **Old damage vs new damage** | Cannot determine from visual alone | Out of scope |
| **Location on package** (corner/side/top) | Inferred from box coordinates | Compute post-detection |

### Why This Matters

Training severity or decisions as classes:
- Increases labeling subjectivity → inconsistent dataset
- Forces model to learn non-visual concepts → poor generalization
- Reduces samples per class → underfitting on rare cases

---

## 3. LABELING RULES

### Class 0: `structural_deformation`

#### ✅ INCLUDE (Label as this class)

| Pattern | Visual Indicator |
|---------|------------------|
| Dents | Depressed area with shadow/highlight contrast |
| Bent corners | Corner angle deviates from 90° |
| Warped panels | Flat surface shows curvature |
| Impact marks | Localized depression with radial distortion |
| Pushed-in areas | Surface recessed relative to surrounding |

#### ❌ EXCLUDE (Do NOT label as this class)

| Pattern | Reasoning |
|---------|-----------|
| Intentional folds | Design feature, not damage |
| Creases from stacking | Normal wear, no structural compromise |
| Printed shadows | Graphic design, not actual deformation |
| Packaging seams | Manufacturing feature |
| Slightly soft corners on new boxes | Normal manufacturing variance |

#### ⚠️ EDGE CASES (Labeler must decide carefully)

| Case | Guidance |
|------|----------|
| Very shallow dent | Include if visible under standard lighting; exclude if only visible from extreme angles |
| Corner slightly pushed in | Include if >5mm deviation; exclude if within normal box flex |
| Multiple small dents | Label each separately if distinct; one box if cluster |

---

### Class 1: `surface_breach`

#### ✅ INCLUDE (Label as this class)

| Pattern | Visual Indicator |
|---------|------------------|
| Tears | Visible rip in packaging material |
| Punctures | Hole penetrating packaging |
| Cuts/slashes | Linear opening in material |
| Burst seams | Seam separation exposing interior |
| Rips exposing contents | Any breach showing inner material |

#### ❌ EXCLUDE (Do NOT label as this class)

| Pattern | Reasoning |
|---------|-----------|
| Perforated tear strips | Intentional design feature |
| Ventilation holes | Manufacturing feature |
| Clear windows in packaging | Intentional design |
| Scuff marks (surface only) | No breach; use `contamination_stain` if visible mark |

#### ⚠️ EDGE CASES (Labeler must decide carefully)

| Case | Guidance |
|------|----------|
| Very small puncture (<3mm) | Include if inner layer visible; exclude if surface scratch only |
| Tape covering a breach | Label the breach, not the tape (unless tape is also damaged) |
| Stress whitening (not torn yet) | Exclude; material stressed but not breached |

---

### Class 2: `contamination_stain`

#### ✅ INCLUDE (Label as this class)

| Pattern | Visual Indicator |
|---------|------------------|
| Water stains | Dried water marks, waterlines, tide marks |
| Oil/grease marks | Dark spots with irregular edges |
| Chemical spills | Discoloration with burn or bleach patterns |
| Mold/mildew | Organic growth patterns, dark spots |
| Dirt/mud stains | Ground contamination marks |
| Leakage from inside | Stains originating from package interior |

#### ❌ EXCLUDE (Do NOT label as this class)

| Pattern | Reasoning |
|---------|-----------|
| Printed patterns/colors | Design feature |
| Normal tape adhesive marks | Common, not contamination |
| Dust (removable) | Not permanent staining |
| Scuff marks from handling | Normal wear, no contamination |
| Faded printing | Aging, not contamination |

#### ⚠️ EDGE CASES (Labeler must decide carefully)

| Case | Guidance |
|------|----------|
| Very light water mark | Include if distinctly visible; exclude if barely perceptible |
| Active wetness vs dried stain | Both included; active wet = contamination occurring |
| Stain vs shadow | Verify stain doesn't move with lighting angle |

---

### Class 3: `compression_damage`

#### ✅ INCLUDE (Label as this class)

| Pattern | Visual Indicator |
|---------|------------------|
| Crushed corners | Corner collapsed with material folding |
| Collapsed edges | Edge line compressed inward |
| Accordion folds | Multiple parallel creases from compression |
| Buckling | Wall surfaces bulging under weight stress |
| Flattened sections | Originally 3D area now 2D |

#### ❌ EXCLUDE (Do NOT label as this class)

| Pattern | Reasoning |
|---------|-----------|
| Simple dents | Use `structural_deformation` instead |
| Intentional flat areas | Design feature |
| Soft corners on padded mailers | Normal for flexible packaging |
| Minor edge wear | Normal handling, no compression failure |

#### ⚠️ EDGE CASES (Labeler must decide carefully)

| Case | Guidance |
|------|----------|
| Dent vs compression | Compression shows directional force; dent is localized impact. If in doubt, use `structural_deformation` |
| Corner slightly pushed in vs crushed | Crushed shows material failure (creases, folds); pushed shows only displacement |

---

### Class 4: `tape_seal_damage`

#### ✅ INCLUDE (Label as this class)

| Pattern | Visual Indicator |
|---------|------------------|
| Torn tape | Tape ripped, exposing gap in seal |
| Missing tape | Seal area without expected tape |
| Resealed tape | Double layer or misaligned tape (tampering indicator) |
| Peeling tape | Tape lifting from surface |
| Peeling labels | Shipping labels separating |
| Broken straps | Banding straps cut or snapped |

#### ❌ EXCLUDE (Do NOT label as this class)

| Pattern | Reasoning |
|---------|-----------|
| Tape lifting at very edges | Normal for some tape types |
| Multiple labels (normal stacking) | Normal restickering |
| Security tape functioning correctly | No damage |
| Slightly crooked tape | Poor application, not damage |

#### ⚠️ EDGE CASES (Labeler must decide carefully)

| Case | Guidance |
|------|----------|
| Tape bubble vs tape damage | Bubble = cosmetic; lifted edge = potential tampering |
| "OPENED" security tape triggered | Include if seal is broken; may indicate tampering |

---

## 4. BOUNDING BOX vs SEGMENTATION DECISION

| Class | Annotation Type | Justification |
|-------|-----------------|---------------|
| `structural_deformation` | **Bounding Box** | Dents/bends have irregular 3D shadows; precise boundary unnecessary for accept/reject decision |
| `surface_breach` | **Bounding Box** | Box captures damaged region; exact edge tracing not needed for decision |
| `contamination_stain` | **Bounding Box** | Stains have fuzzy edges; pixel-perfect boundary impractical and unnecessary |
| `compression_damage` | **Bounding Box** | Crush patterns have complex geometry; box is sufficient for detection |
| `tape_seal_damage` | **Bounding Box** | Tape damage is localized; box identifies affected area |

### Why Not Segmentation?

| Reason | Explanation |
|--------|-------------|
| **Decision parity** | Accept/reject decision doesn't change based on exact pixel boundary |
| **Labeling speed** | Boxes are 5-10x faster to label than masks |
| **Edge deployment** | Detection is 2-3x faster than segmentation |
| **Noise reduction** | Segmentation masks on textured packaging produce noisy edges |

### When Segmentation Might Be Added (Future Phase)

- If precise damage area calculation becomes business requirement (insurance claims)
- If damage percentage of package surface is needed for grading

---

## 5. CLASS MERGING / SPLITTING

### Phase 1: Initial Deployment (5 Classes)

Use the 5 classes defined above. Keep it simple.

### Merge Candidates (Keep Merged)

| Potential Split | Recommendation | Reasoning |
|-----------------|----------------|-----------|
| Dent vs Bend | **Keep merged as `structural_deformation`** | Visually similar; same business impact |
| Tear vs Puncture | **Keep merged as `surface_breach`** | Both are breaches; distinction rarely changes decision |
| Water vs Oil stain | **Keep merged as `contamination_stain`** | Both are contamination; source rarely matters |

### Split Candidates (Consider for Phase 2)

| Current Class | Potential Split | Trigger to Split |
|---------------|-----------------|------------------|
| `surface_breach` | `tear`, `puncture` | If puncture requires different handling than tears |
| `contamination_stain` | `water_damage`, `chemical_damage` | If chemical exposure requires immediate quarantine |
| `tape_seal_damage` | `tampered_seal`, `worn_seal` | If security/tampering is separate business process |

### Phase 2 Addition Candidates

| New Class | Trigger to Add |
|-----------|---------------|
| `label_damage` | If shipping label legibility affects routing |
| `pallet_damage` | If palletized shipments inspected (not individual packages) |
| `shrink_wrap_damage` | If shrink-wrapped pallets inspected |

---

## 6. OUTPUT FORMAT COMPATIBILITY

### YOLO Training Format

```yaml
# damage.yaml (dataset config)
path: ../datasets/package_damage
train: images/train
val: images/val
test: images/test

names:
  0: structural_deformation
  1: surface_breach
  2: contamination_stain
  3: compression_damage
  4: tape_seal_damage
```

### Label File Format

```
# Standard YOLO format: class x_center y_center width height
# All values normalized 0-1 relative to image dimensions

# Example: labels/train/image001.txt
0 0.456 0.312 0.089 0.124
2 0.721 0.543 0.156 0.087
4 0.234 0.891 0.067 0.043
```

### Compatibility Verified

| Requirement | Status |
|-------------|--------|
| Integer class IDs | ✅ 0-4 |
| Standard detection head | ✅ No custom heads needed |
| Normalized coordinates | ✅ YOLO native format |
| Single class per box | ✅ Each damage is one class |
| Multi-label per image | ✅ Multiple damages allowed |

---

## Summary

| Aspect | Decision |
|--------|----------|
| **Number of classes** | 5 (minimal but complete) |
| **Annotation type** | Bounding boxes only |
| **Severity handling** | Post-detection logic, NOT a class |
| **Decision handling** | Post-detection logic, NOT a class |
| **Format** | Standard YOLO-compatible |
| **Expansion path** | Split contamination/breach if business requires |
