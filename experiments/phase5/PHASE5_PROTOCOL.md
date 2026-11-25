# Phase 5 protocol: verified four-band composition model

**Version 1.1, frozen 2026-11-25** (supersedes 1.0; see §10). Machine-readable twin: [`protocol.json`](protocol.json). Its SHA-256 is
stamped into every snapshot, split file and checkpoint. Changing a threshold requires a new protocol
version and owner approval, decided **before** looking at results.

## 1. Question and limits
* **Can an RGB photograph place a fabric in a cotton band (Low 30–40 %, Medium 50–66 %, High 80–95 %,
  Very high 98–100 %), for physical fabrics never seen in training?**
* The model estimates appearance. It never chemically verifies fibre composition, and its output must not be
  used for valuation or token allocation.

## 2. Ground truth
* Only Phase 4 fabrics with status **VERIFIED under verification policy v2** count as ground truth:
  * accepted laboratory evidence, or exact-product manufacturer/supplier documentation;
  * plus an independent second review.
* Garment care labels never count.
* CottonFabricImageBD and any other unverified labels are never ground truth.

## 3. Unit of independence
The **provenance group** is the unit. Fabrics are joined into one group if they share a `provenance_group`, a
source product/lot identifier, or an evidence product identifier. Every photograph of every fabric in a group
stays in the same split. Photographs are never counted as independent samples.

## 4. Data pipeline (in this order; each step refuses to run if the previous failed)
1. The Phase 4 validator reports **zero errors**.
2. **Snapshot:**
   * contents: frozen copies of the registries, fabric status, manifest and audit;
   * hashes: SHA-256 of every CSV and every image, plus the protocol hash;
   * versioning: a version ID, re-verified before every use.
3. **Readiness gates R1–R9** (§6) on the snapshot. The result is READY, PILOT_ONLY or NOT_READY.
4. **Lockbox:** about 20 % of provenance groups per band, stratified, with a fixed seed. It is sealed by hash
   before any baseline or model sees the data, and opened once, at the end, for the selected model.
5. **Development splits:** repeated stratified group k-fold (3 × 5) on the non-lockbox groups. Each training
   fold holds back an inner validation split, by group, for early stopping. The test fold is never used for
   model selection.

## 5. Experiments
| ID | What | Purpose |
|---|---|---|
| E0 | Full pipeline on **synthetic** data | Tests the pipeline only. Results are **not** model performance. |
| E1 | Majority, colour-histogram and metadata-only baselines | Floors and confound detectors |
| E2 | Frozen ImageNet ResNet-50 features + logistic regression | Small-data baseline |
| E3 | Fine-tuned ResNet-50 (Phase 2/3 recipe, strong colour augmentation) | Reference model |
| E4 | EfficientNet-B0, MobileNetV3-Large (identical folds/seeds) | Lighter alternatives |
| E5 | Macro-only vs all photos; grayscale | How much the model relies on colour |
| E6 | (optional) ordinal/regression on verified cotton % | Uses exact labels; scored by MAE |

**Selection:** the simplest model whose paired balanced-accuracy difference to the best has a 95 % CI that
includes 0. Lighter models are preferred when equivalent.

## 6. Readiness gates (all must pass for READY)
| Gate | Requirement |
|---|---|
| R1 | ≥ 30 VERIFIED, in-band, error-free fabrics per band. 20–29 per band permits a clearly labelled pilot only. |
| R2 | Phase 4 validator: 0 errors; dataset on verification policy v2 |
| R3 | Every fabric has ≥ 4 photographs, including `full_fabric` and `macro_texture` |
| R4 | No identifier shared across provenance groups; ≥ 5 distinct sources per band; no source > 30 % of a band |
| R5 | ≥ 4 colour families per band; woven/knit mix reported |
| R6 | Cramér's V between band and capture device / lighting / capture session < 0.30; the metadata-only baseline's balanced accuracy is within 15 points of chance |
| R7 | ≤ 20 % of fabrics per band at **adjacent-band risk**: distance from the verified cotton % to the nearest *other* band ≤ 3 points (§6.1) |
| R8 | Every cross-group pair of photographs flagged as look-alike has a recorded human review decision, and none is "same fabric". Flags: colour/structure similarity ≥ 0.92 **or** cached ImageNet ResNet-50 embedding similarity ≥ 0.85 (§6.2). |
| R9 | Snapshot hashes verify, and the lockbox is sealed before any experiment |

### 6.1 R7 definition (v1.1)
For a fabric with verified cotton percentage *c* in band *b*:

    dist(c, [lo, hi]) = max(lo − c, 0, c − hi)
    at_risk(c) = min over bands b' ≠ b of dist(c, b') ≤ 3

* The bands are separated by gaps of 10, 14 and 3 points (40 | 50, 66 | 80, 95 | 98). With the 3-point distance, only the
  High/Very high boundary carries risk:
  * a High fabric at exactly 95 % is 3 points from Very high;
  * a Very high fabric at exactly 98 % is 3 points from High;
  * 99 % and 100 % are 4–5 points from High and are **not** at risk.
* Edges facing an empty gap or the physical limits 0/100 % do not make a label ambiguous between two classes. Being near
  them is reported for information (`gap_edge_proximity`) but does not count toward the gate.
* Why v1.0 failed: it measured distance to every band endpoint. The Very high band is 2 points wide, so every Very high
  value lay within 3 points of 98 and READY was unreachable.

### 6.2 R8 review flags (v1.1)
Each cross-group image pair is flagged when:
* colour/structure similarity ≥ 0.92; **or**
* the cosine similarity of ImageNet ResNet-50 (IMAGENET1K_V2) embeddings ≥ 0.85.

The embedding model loads only from the local torch cache and is never downloaded; if it is unavailable, the report
says so and only the colour/structure flag is used. Flags list both scores and the reason. **Nothing is merged or
declared identical automatically.** A reviewer records `distinct` or `same` (per image pair, or per fabric pair). Any
`same` decision blocks READY until the fabrics are given one `provenance_group` in Phase 4.

## 7. Evaluation
* **Primary:** fabric-level metrics (mean of image probabilities per fabric):
  * balanced accuracy, macro F1, per-band precision and recall;
  * confusion matrix and far-band error rate (Low ↔ Very high);
  * expected calibration error.
* **Uncertainty:** 2,000 bootstrap resamples over fabrics give 95 % CIs. Paired comparisons use identical folds
  and seeds.
* **Secondary:** image-level metrics, plus breakdowns by colour family, weave and source.

## 8. Success criteria (all must hold; even then the model stays "experimental")
| | Criterion |
|---|---|
| S1 | Fabric balanced accuracy: mean ≥ 0.55 and 95 % CI lower bound ≥ 0.40 (chance 0.25) |
| S2 | ≥ 10 points above both the colour-only and the metadata-only baselines (paired, majority of folds) |
| S3 | Every band: recall ≥ 0.40 and precision ≥ 0.35 |
| S4 | Far-band errors (Low ↔ Very high) ≤ 5 % |
| S5 | Std of balanced accuracy across repeats ≤ 0.10 |
| S6 | Lockbox balanced accuracy within the CV 95 % CI |
| S7 | Expected calibration error ≤ 0.10 (after calibration on validation data) if probabilities are shown |

## 9. Guards in code
* Training on a non-synthetic snapshot is refused unless:
  * readiness is READY;
  * the snapshot and lockbox verify;
  * the protocol hash matches;
  * an explicit approval flag is given.
* Synthetic snapshots are stamped SYNTHETIC in every output: "PIPELINE TEST, not model performance".

## 10. Version history
| Version | Date | Change | Record |
|---|---|---|---|
| 1.0 | 2026-11-25 | Initial frozen protocol | `protocol_history/protocol_v1.0.json` (SHA-256 `634ab240…839e82c`), `protocol_history/PHASE5_PROTOCOL_v1.0.md` |
| 1.1 | 2026-11-25 | R7 redefined as adjacent-band distance (3 points and 20 % unchanged); R8 adds a cached ResNet-50 embedding flag (0.85). Approved by the project owner. | this file + `protocol.json` |

Snapshots record the protocol version and SHA-256 they were created under. `verify_snapshot` reports a mismatch with the
current protocol, and real-data training refuses it. Records made under an earlier version are never rewritten; a new
snapshot is required.
