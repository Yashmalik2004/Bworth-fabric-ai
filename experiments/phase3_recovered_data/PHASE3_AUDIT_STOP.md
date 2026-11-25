# Phase 3 — audit stopped before training

**Status:** stopped at the physical-swatch audit (step B) because of a data-integrity failure.
No model was trained. No Phase 1 or Phase 2 file was modified (MD5 check of 211 existing files).

## What was done

1. **Audit A (labels and duplicate identity)**, all 240 conflict clusters —
   `audit/audit_a_all_conflict_clusters.csv`
   * 216 clusters have labels that all fall in one band. 24 span two bands and are rejected.
   * Of the 216, **214 pass**: 213 are byte-identical copies, and one is a re-encoded copy with
     dHash ≤ 6 and mean absolute pixel difference ≤ 4.6.
   * 2 are rejected: copies close (dHash 2–5) but pixel difference 7.6 > 6, so not provably the same photograph.
2. **Audit B (physical-swatch mapping)**, the 214 survivors —
   `audit/audit_b_candidates.csv`, `audit/review_sheet_0{1,2,3}.jpg`
   * Recovered images were pre-clustered within each band (30 pre-clusters).
   * Each was visually compared with its nearest clean swatch groups, searching **all** bands.

## Why it stopped

**1. The same physical fabric appears under different cotton bands in the *clean* dataset.**
   * A pink fabric with a distinctive flower watermark is in 58 % g01 (Medium) and 80 % g11 (High).
   * The red fabric with white slub lines appears in 30 % (Low), 65 % (Medium), 80 % and 95 % (High).
   * Evidence: `audit/cross_band_check_warm_fabrics.jpg`.
   * The 240 byte-identical cross-label copies already showed that photos were copied between class
     folders. This shows the problem extends to different photos of the same fabric, so it is not removed
     by de-duplication.

**2. Phase 2's swatch grouping missed cross-class matches within a band, and one leaks across CV folds.**
   * The recovered pair R_medium_03 links the blue/cream chevron knit in 65 % g03 (fold 0) with 66 % g03+g04
     (fold 4). Evidence: `audit/within_band_cross_fold_chevron.jpg`.
   * This affects 4 clean images. Phase 1 and 2 grouping only clustered within a class, so other cases may
     exist that the recovered images did not happen to reveal.

**3. The recovered images cannot be mapped to swatches safely.**
   * 152 of the 214 sit in five large pre-clusters: the warm plain-weave family, generic grey denims and
     green/teal weaves.
   * Their nearest clean fabrics span several classes, folds and often **bands**. Attaching them to one group
     would either leak across folds or assign a band to a fabric the dataset also labels with another band.
   * Treated conservatively as instructed, **only 14 images (6 pre-clusters) are provisionally recoverable**,
     each with a distinctive match to a single same-band group in a single fold.
   * The other 200 are rejected as uncertain. Per-cluster reasons:
     `audit/recoverable_groups_audit_PROVISIONAL.csv`.

## Consequences

* Training with +14 images (+2 %) could not show a measurable effect: fold-to-fold spread is ±6 points.
* The larger issue is the labels. If one physical fabric carries several cotton bands, the dataset cannot
  support composition learning for that fabric, and **the Phase 2 numbers are themselves affected**:
  * cross-band duplicates in different folds lower measured accuracy (contradictory labels);
  * within-band cross-class duplicates in different folds raise it (leakage).
  The net direction is unknown.

## Options (your decision)

A. **Proceed minimally:** add the 14 provisional images and run the planned CV (expanded vs Phase 2,
   two seeds). Expected outcome: no detectable change. Mainly documents that the recovery yields little.
B. **Re-audit swatch identity across classes first (recommended):**
   * Cluster all 638 clean and 214 recovered images jointly across classes, review galleries per band and
     across bands, and record a dataset-wide fabric ID.
   * Then (i) drop fabrics that carry labels from more than one band, (ii) rebuild CV folds by fabric ID,
     (iii) re-run Phase 2 and Phase 3 on the corrected folds.
   * This is the only way to get trustworthy numbers from this dataset. It changes the Phase 2 fold
     assignments, so comparisons would be re-run rather than reused.
C. **Stop using this dataset for composition** beyond the demo and collect a small, controlled set with
   verified composition labels (one physical fabric = one label, several photos each).

Code for the full experiment (build with integrity checks, CV, fixed-split model, report) is in
`training/composition/phase3_recover.py`. `build` refuses to run without a reviewed
`audit/swatch_decisions.json`, so nothing can be trained by accident.
