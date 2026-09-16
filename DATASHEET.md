# OffRoad-OOD — Benchmark Datasheet

This document specifies how the OffRoad-OOD benchmark is constructed from four
existing public off-road datasets, so that the task, splits, and evaluation are
fully reproducible. It is the detailed companion to the paper's Section III.

---

## 1. What this benchmark is (and is not)

OffRoad-OOD is a **task, protocol, and evaluation layer**, not a new dataset of
raw sensor data. We do **not** collect or redistribute images. We define, over
four existing public datasets:

1. an **ID/OOD ontology** — which semantic classes count as in-distribution
   background vs. out-of-distribution objects;
2. **leakage-free splits** — train/val/test partitions that do not leak
   background between sides;
3. an **evaluation suite** — pixel- and object-level metrics with confidence
   intervals and fairness controls.

**Scope (honest).** The task is *semantic OOD of discrete objects*: detecting
unexpected, discrete objects (people, animals, vehicles, structures, debris)
against the natural off-road background. This is an **upstream perception
component**, not a complete traversability or hazard detector. Naturally
occurring hazards that are part of the terrain (negative obstacles, deep water,
soft ground, steep slopes) are largely *natural* and are absorbed into the
in-distribution background; they are out of scope here. We make no causal claim
about downstream navigation safety.

---

## 2. Source datasets and licenses

| Dataset | Reference venue | License | Redistributed here |
|---|---|---|---|
| GOOSE | ICRA 2024 | CC BY-SA 4.0 | splits + ontology only |
| RELLIS-3D | ICRA 2021 | CC BY-NC-SA 3.0 | splits + ontology only |
| WildScenes | IJRR 2025 | CC BY (CSIRO DAP) | splits + ontology only |
| RUGD | IROS 2019 | research use | splits + ontology only |

Only **derived artifacts** (frame lists in `configs/*_split.json` and the class
mapping in `configs/meta_ontology.yaml`) are included in this repository. The
images must be downloaded from the original sources (see `README.md` §2) and
remain under the licenses above. RELLIS-3D is non-commercial; treat all derived
splits accordingly.

---

## 3. The meta-ontology

Every dataset's native classes are first mapped to a shared meta-ontology, then
partitioned into ID / OOD / ignore. This cross-dataset consistency is the core
standardization the benchmark adds. The full mapping is in
`configs/meta_ontology.yaml` (the single source of truth, parsed by
`ontology.py`).

**Roles.**
- **ID** = the expected natural background a robot normally traverses:
  natural terrain (soil, sand, gravel, mud, grass, moss, snow, cobble),
  vegetation (bush, tree, forest, foliage, trunk, crops), sky, and water.
- **OOD** = unexpected, discrete objects, grouped into three tiers:
  - **T1** — salient foreign objects: vehicles, man-made structures, and people
    (matches the classes used by prior off-road work, for comparability).
  - **T2** — small/scattered objects and signage: debris, poles, pipes, wires,
    cones, signs. This is the hardest, most novel tier (the main terrain
    false-positive regime).
  - **T3** — animals (rare, deformable).
- **ignore** — ego-vehicle, undefined/void, and ambiguous urban road furniture
  (curb, sidewalk, road markings, crossings), excluded from both ID and OOD to
  avoid contaminating either.

**Per-dataset mapping (summary; exact class names in the YAML).**
- **GOOSE (64 classes):** 20 ID (5 terrain + 12 vegetation + sky + water),
  35 OOD (21 T1 + 13 T2 + 1 T3), 9 ignore.
- **RELLIS-3D:** 11 ID, 6 OOD (4 T1 + 2 T2).
- **WildScenes:** 12 ID, 5 OOD (3 T1 + 2 T2); no person/animal classes, so it
  covers only structure/object OOD (stated as a limitation).
- **RUGD:** 14 ID, 9 OOD (6 T1 + 3 T2). We additionally reproduce the exact
  16-ID / 8-OOD split of the prior off-road method for a protocol-matched
  comparison.

---

## 4. Leakage-free splits

**Why re-split.** Several off-road datasets are collected as continuous
traversals, so their official *frame-level* train/val splits place
near-identical backgrounds on both sides. A detector can then memorize terrain
rather than learn genuine novelty, inflating scores. We measured this directly:
on GOOSE, training a detector with vs. without same-site background present
changes pixel AP from **0.35 to 0.67** and FPR95 from **0.66 to 0.43** — an
apparent gain that is entirely leakage (`leakage_ablation.py`).

**Split strategy (per dataset).**

| Dataset | train / val / test frames | Strategy |
|---|---|---|
| GOOSE | 6105 / 991 / 1711 | **Site-level** — whole locations kept on one side; semi-urban sites (campus) go to train to keep test purely off-road. |
| RELLIS-3D | 4134 / 900 / 1200 | **Sequence-level** — whole traversal sequences kept on one side. |
| WildScenes | 6051 / 283 / 2133 | **Official 45 m spatial buffer** (already leakage-safe) — adopted as-is. |
| RUGD | 5670 / 646 / 1120 | **Official scene-level** split — adopted as-is. |

Split membership is shipped as frame lists in `configs/*_split.json`. We report
that train and test sites/sequences are disjoint.

---

## 5. Leave-class-out construction

The closed-set segmenter is trained only on ID classes:

- **Training:** OOD pixels are mapped to `ignore` (pixel-masking) and do not
  contribute to the loss. Whole-frame exclusion is not used, because pure-ID
  frames are too scarce off-road (e.g. some GOOSE sites have essentially none).
  Pixel-masking is the standard leave-class-out construction (StreetHazards,
  BDD-Anomaly, MUAD). Residual training-context leakage (the model sees the
  masked region's surroundings) is a known, accepted limitation.
- **Testing:** each pixel is labeled OOD (positive), ID (negative), or ignore
  (not scored). Anomaly scores are produced per pixel by each method in its own
  paradigm (feature distance, logit score, mask score, …).

---

## 6. Borderline-class adjudication

Ambiguous classes are decided with a fixed reference segmenter and a confusion
analysis (`confusion_matrix.py`), and documented rather than hidden:

- **rock / log:** natural presence vs. hard obstacle — default **ID** (to keep
  OOD strictly "foreign"), with a sensitivity note. On GOOSE, `rock` is
  self-predicted only ~37% of the time (mostly confused with other terrain),
  which is recorded.
- **debris / rubble:** man-made debris → OOD-T2; natural rubble is adjudicated
  by confusion.
- **urban road furniture** (curb, sidewalk, road markings): ambiguous off-road
  → **ignore**.
- **snow / puddle:** seasonal natural → ID (usable as a domain-shift probe).

**Validity check.** For each OOD class we verify with the ID-only reference
model that it is genuinely confusing (low confidence / high anomaly score); a
class that is confidently and stably assigned to some ID class is flagged as a
weak anomaly and discussed. A per-class min-support threshold decides whether a
class is reported individually or folded into its tier.

---

## 7. Evaluation protocol

- **Pixel-level:** average precision (**AP**, primary, robust to imbalance),
  AUROC, and FPR95. AUROC is reported but is optimistic under the extreme
  foreground/background imbalance of off-road scenes; we rank on AP.
- **Object-level:** `F1*`, the threshold-averaged connected-component F1 of
  SegmentMeIfYouCan (minimum-area and sIoU thresholds fixed and published).
- **Statistics:** 95% frame-level bootstrap confidence intervals; training-based
  methods run over multiple seeds (mean ± std).
- **Fairness controls:** every method fits its ID statistics on the *same* train
  split; outlier exposure is a *separate track* with a fixed COCO subset;
  frozen backbones are unified to DINOv2; **any threshold is selected on
  validation and never on test**; inference cost (ms/frame, params) is reported
  alongside accuracy.
- **Operating point:** we additionally evaluate deployable, ID-anchored
  thresholds via terrain-conditional conformal calibration
  (`conformal_headroom.py`), which needs no OOD labels.

---

## 8. Known limitations

1. Semantic OOD of discrete objects ≠ complete traversability/hazard detection;
   natural hazards are absorbed into ID.
2. Held-out known classes are a proxy for genuine open-world novelty.
3. Training-context leakage from pixel-masking is accepted (see §5).
4. WildScenes lacks person/animal OOD classes (structure/object only).
5. Evaluation is offline; there is no real-robot closed-loop study.

---

## 9. Reproducing the benchmark artifacts

- `plan_split.py`, `audit_goose.py`, `audit_wildscenes.py` — build and audit the
  splits and check feasibility/leakage.
- `ontology.py` — parse `meta_ontology.yaml`; the loaders apply pixel-masking.
- `leakage_ablation.py` — the controlled leakage experiment of §4.
- `confusion_matrix.py` — the borderline-class validity check of §6.

See `README.md` §3 for the full end-to-end reproduction sequence.
