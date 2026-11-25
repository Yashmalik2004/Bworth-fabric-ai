# Fold integrity report (Phase 3B, Stage A): PROPOSED folds, not yet approved

Construction:
* `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
* Groups = `global_fabric_id`; stratified by four-class band. One row per photograph.
* Files: `five_fold_assignments.csv` (strict) and `five_fold_assignments_ALT_family_safe.csv`.
* Per-fold distributions: `fold_distribution_report*.csv`. Raw checks: `integrity_checks.json`.

## Leakage checks: both policies PASS

| Check | Strict | Family-safe |
|---|---|---|
| Physical-fabric IDs shared across folds | **0** | **0** |
| Photographs in more than one fold | 0 | 0 |
| SHA-256 values (any copy of a photograph) in more than one fold | 0 | 0 |
| Fabrics carrying more than one band | 0 | 0 |
| Contradictory or unresolved fabrics included | 0 | 0 |
| Cross-fold photograph pairs with appearance similarity ≥ 0.90 | 0 | 0 |
| All rows assigned a fold | yes | yes |

## Stratification: FAILS for the Low band

| Fold | Strict: fabrics (L / M / H / VH) | Strict photos | Family-safe: fabrics (L / M / H / VH) | Family-safe photos |
|---|---|---|---|---|
| 0 | 0 / 5 / 4 / 2 | 26 | 1 / 6 / 5 / 2 | 44 |
| 1 | 0 / 5 / 5 / 1 | 27 | 0 / 8 / 3 / 2 | 44 |
| 2 | 0 / 5 / 4 / 2 | 26 | 0 / 8 / 6 / 1 | 52 |
| 3 | 1 / 4 / 5 / 1 | 27 | 1 / 8 / 5 / 2 | 51 |
| 4 | 1 / 5 / 2 / 2 | 27 | 1 / 8 / 4 / 2 | 45 |

* **Low (30–40 %)** has only **2 independent fabrics** under the strict policy (4 photographs) and 3 under
  family-safe (13 photographs). Three (strict) or two (family-safe) folds contain **no Low test image**, so
  per-fold balanced accuracy over four bands is undefined there, and pooled Low recall would rest on two or
  three fabrics.
* **Very high** has 8–9 fabrics: one or two per fold, so per-fold scores for that band are very noisy.
* Medium and High (20–38 fabrics) are adequate.

**Conclusion: the corrected data does not support a meaningful four-band five-fold evaluation.** Per the
brief, no five-fold four-band result should be produced from these folds until this is resolved (see the
options in `../reports/STAGE_A_AUDIT_REPORT.md`). The folds are saved as proposals only.
