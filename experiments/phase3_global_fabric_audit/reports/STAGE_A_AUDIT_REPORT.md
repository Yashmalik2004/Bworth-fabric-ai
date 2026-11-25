# Phase 3B, Stage A: dataset-wide physical-fabric identity audit

**Status: Stage A complete. Stopped for review. No model was trained.**
Existing Phase 1, Phase 2 and Phase 3 files are unchanged: SHA-256 check of 263 files, see
`audit/preservation/`.

## Method
1. **Inventory**: all 28,605 files (1,300 Original + 27,305 Augmented). Recorded for each: SHA-256, size,
   dimensions, perceptual hash (Original), and prior Phase 1/2/3 membership.
   `audit/complete_image_inventory.csv`.
2. **Same photograph**: identical SHA-256; or dHash ≤ 8/256 **and** mean absolute pixel difference ≤ 6/255,
   with complete linkage (no chaining). 1,300 files → 889 unique hashes → **883 photographs**.
   `mappings/image_to_photograph.csv`, `audit/near_duplicate_photo_pairs.csv`.
3. **Shortlisting across all classes**:
   * Agglomerative clustering of all 883 photographs together (ImageNet ResNet-50 embedding 0.7 + HSV
     colour histogram 0.3, average-linkage, distance 0.30) gave **150 visual families**.
   * Plus a list of 505 cross-family, cross-band look-alike pairs.
   * Similarity was used only to shortlist, never as proof.
4. **Visual review** of every family: 35 contact sheets, plus the cross-family pair sheet.
   * Each photograph got CONFIRMED_MATCH, POSSIBLE_MATCH, DISTINCT_FABRIC or UNCERTAIN, with written evidence.
   * CONFIRMED requires a specific feature: identical motif, stripe sequence, stain, plaid, knit pattern, or
     byte-identical photographs.
   * Decisions: `audit/identity_review_decisions.json`.
5. **Safety check**: every kept photograph compared with its most similar photograph in another band. Six
   groups were moved to UNCERTAIN as a result.
6. **Label audit**: a fabric's bands are the union over every file of every photograph in it, so a photo copied
   into another class counts.

## Findings

| | Count |
|---|---|
| Images audited (Original) | 1,300 files / 883 photographs |
| Global physical-fabric IDs | **100** (60 CONFIRMED_MATCH, 21 POSSIBLE_MATCH, 19 DISTINCT_FABRIC) |
| Photographs by identity status | 303 confirmed, 114 possible, 19 distinct, **447 uncertain** |
| Fabrics with consistent band | 73 |
| **Fabrics with contradictory bands** | **27 (339 files)** |
| Unresolved remainders | 77 groups (447 photographs, 601 files) |

Notable contradictory fabrics (`audit/contradictory_fabrics.csv`):
* Terracotta open mesh: 30, 40, 53, 58, 65, 66 %.
* Green mesh: 50, 53, 58, 80, 95 %.
* Orange honeycomb: 30, 53, 58, 95 %.
* Grey oval mesh: 30, 40, 58, 80, 99 %.
* Red slub plain weave: 30, 65, 80 %.
* Pink flower-watermark fabric: 53, 58, 80 %.
* Royal blue: 50, 58, 98, 99 %.
* 14 of the 27 contradictory fabrics are proven by a byte-identical photograph filed under more than one
  band: 10 grey/blue denim-type fabrics (50–63 % vs 98–99 %), plus royal blue, grey oval mesh, red slub
  and orange weave.

The **chevron knit garment** (65 % + 66 %) is a single fabric ID within one band. It is consistent and
eligible, and is now kept in one fold.

The Phase 2 fold assignment is therefore not trustworthy:
* the Phase 2 clean set contains photographs of all 27 contradictory fabrics;
* Phase 2 grouping only compared images within a class.

## Corrected dataset and folds
See `datasets/dataset_cleaning_report.md` and `splits/fold_integrity_report.md`.

| | Strict (primary) | Family-safe (alternative) |
|---|---|---|
| Eligible photographs / fabrics | 133 / 54 | 236 / 73 |
| Low / Medium / High / Very-high fabrics | **2** / 24 / 20 / 8 | **3** / 38 / 23 / 9 |
| Low / Medium / High / Very-high photographs | **4** / 59 / 56 / 14 | **13** / 126 / 70 / 27 |
| Fabrics per fold | 11, 11, 11, 11, 10 | 14, 13, 15, 16, 15 |
| Zero fabric leakage across folds | **PASS** | **PASS** |
| Low band present in every fold | **FAIL** (3 folds empty) | **FAIL** (2 folds empty) |

## Unresolved integrity issues
1. **Low band is not evaluable.** Only 2–3 independent low-cotton fabrics survive, because almost every
   30/40 % fabric is also filed under a medium or high band. A four-band five-fold result would be
   misleading, so the brief's stop condition applies.
2. **Identity calls are visual and single-reviewer.** CONFIRMED calls rest on specific features. POSSIBLE and
   UNCERTAIN calls are judgement on 140–150 px thumbnails. A second reviewer, ideally with access to the
   physical samples or the dataset authors, would strengthen them.
3. **Label truth is unknown even for consistent fabrics.** Band consistency only means the dataset never
   contradicts itself for that fabric, not that the composition is right.
4. **Phase 8 needs redefinition.** The global audit already classifies the 14 "recoverable" images (6 strict,
   8 family-safe) along with every other former conflict photograph (24 strict / 79 family-safe are now
   included). There is no longer a separate "expanded" dataset to compare against.

## Decisions needed before Stage B
* **A.** Policy: strict (133 photos) or family-safe (236 photos).
* **B.** Task definition, given the Low band:
  1. three bands (Medium / High / Very high) with five-fold CV, reporting Low as not evaluable;
  2. keep four bands but report Low only descriptively, with no balanced accuracy;
  3. leave-one-fabric-out evaluation (more runs, honest but high variance);
  4. stop modelling on this dataset and collect verified samples.
* **C.** Whether Phase 8 becomes "corrected data with vs without the former label-conflict photographs"
  (the closest meaningful version of the original question).
