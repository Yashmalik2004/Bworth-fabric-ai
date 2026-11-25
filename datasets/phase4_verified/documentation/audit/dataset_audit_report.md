# Phase 4 verified fabric dataset: audit report

Generated 2026-11-25T18:58:47 by `python -m data_collection.phase4.validate`. Do not edit by hand: re-run the validator.

## Summary

| | Count |
|---|---|
| Physical fabrics registered | 0 |
| Photographs registered | 0 |
| Verification records | 0 |
| **Fabrics in verified training manifest** | **0** |
| Photographs in verified training manifest | 0 |

Schema / verification policy: schema v2, verification policy v2

Fabric status: no fabrics registered yet

## Progress per band (independent physical fabrics, not photographs)

| Band | Verified fabrics | Target | Pending | Photographs | Sufficient |
|---|---|---|---|---|---|
| Low (30–40%) | 0 | 30 | 0 | 0 | **no** |
| Medium (50–66%) | 0 | 30 | 0 | 0 | **no** |
| High (80–95%) | 0 | 30 | 0 | 0 | **no** |
| Very high (98–100%) | 0 | 30 | 0 | 0 | **no** |

## Issues

| Severity | Check | Count |
|---|---|---|
| WARNING | insufficient_samples_per_band | 4 |

First 40 issues (all in `issues.csv`):

| Severity | Check | Fabric | Image | Record | Message |
|---|---|---|---|---|---|
| WARNING | insufficient_samples_per_band |  |  |  | Low (30–40%): 0 verified independent fabrics, target 30 |
| WARNING | insufficient_samples_per_band |  |  |  | Medium (50–66%): 0 verified independent fabrics, target 30 |
| WARNING | insufficient_samples_per_band |  |  |  | High (80–95%): 0 verified independent fabrics, target 30 |
| WARNING | insufficient_samples_per_band |  |  |  | Very high (98–100%): 0 verified independent fabrics, target 30 |

## Rules applied

* A fabric enters the training manifest only if it is **VERIFIED** (accepted laboratory, manufacturer or product-identified supplier evidence, document on file, reviewer recorded, evidence agreeing within 3.0 points; under verification policy v2 also an independent second review), lies in a task band, and has no ERROR. Garment care labels never verify.
* Fabrics sharing a product identifier must share a `provenance_group` (or be explained); Phase 5 keeps every provenance group in one split.
* Each image must exist under `images/`, match its recorded SHA-256, and not duplicate another photograph.
* Band targets count **independent physical fabrics**. Several photographs of one fabric are one sample.
* Use `fabric_id` as the grouping key for any train/validation/test split.
