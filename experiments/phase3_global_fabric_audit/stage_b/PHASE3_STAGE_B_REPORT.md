# Phase 3, Stage B: corrected three-class experiment and former-conflict ablation

All numbers come from the 20 training runs in this directory. Phase 1, Phase 2 and Stage A files are
unchanged (SHA-256 check of 498 files). Nothing was committed or pushed.

> **Low cotton composition is not evaluable because the dataset does not contain enough independent,
> verified physical-fabric samples.**
> The strict dataset has two low-band fabrics (4 photographs). They are excluded from training and
> evaluation; they were not merged into another band.

> This model estimates a **broad cotton band** from appearance. It does **not** estimate exact cotton
> composition and is not a chemical measurement. Even at band level, its performance on unseen fabrics is
> modest and highly variable (below).

## 1. Dataset and physical-fabric groups

Source: `../datasets/corrected_dataset_manifest.csv` (strict policy) minus the Low band.

| Class | Photographs | Independent physical fabrics |
|---|---|---|
| Medium (50–66 %) | 59 | 24 |
| High (80–95 %) | 56 | 20 |
| Very high (98–99 %) | 14 | 8 |
| **Total** | **129** | **52** |

Every photograph appears once; byte-identical copies are collapsed. The family-safe manifest was not used.

## 2. Folds and integrity verification

* Stage A strict five-fold assignment (`../splits/five_fold_assignments.csv`), unchanged except for removing
  the Low rows.
* In each run: test = fold k, validation = fold k+1 (checkpoint selection and early stopping, as in Phase 2),
  train = the other three folds.

All checks passed before training (`splits/integrity_report.json`):
* every photograph appears once, maps to one fabric ID, and has a fold;
* every fabric ID sits in exactly one fold, so **zero fabric IDs are shared across folds**;
* no SHA-256 (any copy of any photograph) appears in two folds or two photographs;
* every fabric has one band, and only CONFIRMED/DISTINCT, band-consistent fabrics are included;
* every train, validation and test split contains all three classes.

| Fold | Fabrics (M / H / VH) | Test photographs |
|---|---|---|
| 0 | 5 / 4 / 2 | 26 |
| 1 | 5 / 5 / 1 | 27 |
| 2 | 5 / 4 / 2 | 26 |
| 3 | 4 / 5 / 1 | 24 |
| 4 | 5 / 2 / 2 | 26 |

Distribution limitations:
* Very high has only 1–2 fabrics (2–3 photographs) in each test and validation fold, so its per-fold metrics
  rest on one or two fabrics.
* Validation folds are 24–27 photographs, which makes early stopping noisy (see §6).

## 3. Training configuration

The Phase-2 best recipe (`band4_strong`) with three output classes (`configs/<variant>/fold*_seed*.json`):

| Setting | Value |
|---|---|
| Model | ResNet-50, ImageNet `IMAGENET1K_V2`, head Dropout(0.3) + Linear(2048→3) |
| Preprocessing | resize 366 → crop 320 (random-resized crop for train, centre crop for eval); ImageNet normalisation |
| Augmentation (train only) | flips, 90° rotation, random-resized crop (0.4–1.0), blur, **strong colour**: ColorJitter(0.4, 0.4, 0.8, hue 0.5) + RandomGrayscale(0.3) |
| Optimiser | AdamW: backbone lr 1e-4, head lr 1e-3, weight decay 1e-4; cosine schedule over 30 epochs |
| Loss | cross-entropy, label smoothing 0.1, inverse-frequency class weights from each training split |
| Batch / precision | 16 / mixed precision (CUDA) |
| Stopping | max 30 epochs; best validation accuracy (ties → lower val loss); patience 8 |
| Seeds | 42 and 43 for each fold (10 runs per variant) |

Hardware: RTX 3050 4 GB. cuDNN is not deterministic, so re-runs differ slightly.

## 4. Corrected baseline (strict, three classes)

Chance: 33.3 % balanced accuracy.

| Metric (test, unseen fabrics) | Mean ± std over 10 runs | Range | Pooled (258 predictions = 129 photos × 2 seeds) |
|---|---|---|---|
| Balanced accuracy | **51.8 ± 17.4 %** | 33.3 – 82.3 % | 52.7 % |
| Accuracy | 56.0 ± 15.5 % | 30.8 – 76.9 % | 55.8 % |
| Macro F1 | 0.481 ± 0.142 | 0.303 – 0.737 | 0.502 |

Per class (pooled over folds and seeds):

| Class | Precision | Recall | F1 | Support (predictions) |
|---|---|---|---|---|
| Medium | 0.76 | 0.52 | 0.62 | 118 |
| High | 0.62 | 0.63 | 0.63 | 112 |
| Very high | **0.19** | **0.43** | 0.26 | 28 |

Pooled confusion matrix (rows true, columns predicted; M, H, VH):
`[[61, 33, 24], [14, 71, 27], [5, 11, 12]]`.
Figures: `reports/confusion_matrix_baseline.png` and `confusion_matrix_baseline_normalized.png`.

* Medium and High are separated above chance.
* Very high is barely usable: most predictions of it are wrong (precision 0.19), and 51 Medium/High
  photographs are pushed into it. The class weight of ~3.2 for its small training set encourages this.

## 5. Per-fold and per-seed variation (baseline)

| Fold | Balanced accuracy (mean of 2 seeds) | Accuracy | Macro F1 |
|---|---|---|---|
| 0 | 82.3 % | 76.9 % | 0.737 |
| 1 | 37.5 % | 44.4 % | 0.418 |
| 2 | 48.5 % | 36.5 % | 0.370 |
| 3 | 43.9 % | 60.4 % | 0.448 |
| 4 | 46.8 % | 61.5 % | 0.431 |

| Seed | Balanced accuracy (mean over folds) | Pooled balanced accuracy |
|---|---|---|
| 42 | 55.9 % | 56.8 % |
| 43 | 47.7 % | 48.5 % |

* One fold (0) is far easier than the rest. Without it, the mean balanced accuracy is 44 %.
* Changing only the seed moves the mean by 8 points.
* The ±17-point standard deviation is the honest uncertainty: a single number such as "52 %" should not be
  quoted without it.

## 6. Former-conflict ablation

Verified with SHA-256 and fabric IDs:
* **24** strict photographs (12 Medium, 9 High, 3 Very high; 18 fabrics) come from former label-conflict
  groups. 7 fabrics consist only of such photographs.
* **All 6** of the previously "recoverable 14" that passed the strict policy are among these 24, so none was
  counted twice.

Design:
* **B (expanded)** = the corrected baseline above.
* **A** = identical, except every former-conflict photograph is removed from the training split only
  (11–17 photographs per fold).
* Folds, seeds, validation sets and **test sets are identical**, and both variants are scored on the same 129
  test photographs.
* A secondary score uses only the 105 non-conflict test photographs.

| Variant | Balanced acc. (mean ± std) | Accuracy | Macro F1 | Pooled bal. acc. |
|---|---|---|---|---|
| A: without former-conflict photos | 46.8 ± 13.3 % | 56.0 ± 12.2 % | 0.437 ± 0.137 | 46.5 % |
| B: with them (baseline) | 51.8 ± 17.4 % | 56.0 ± 15.5 % | 0.481 ± 0.142 | 52.7 % |

Paired differences, B − A, on the same test photographs:

| | Balanced accuracy | Accuracy | Macro F1 |
|---|---|---|---|
| Mean ± std over 10 paired runs | **+5.0 ± 15.0 pts** | 0.0 ± 9.6 pts | +0.044 ± 0.120 |
| Runs where B is better / worse / equal | 7 / 2 / 1 | | |
| Per-fold mean difference (folds 0–4) | +23.1, +1.4, +9.1, −8.3, −0.1 | | |
| Non-conflict test photographs only | +4.8 ± 16.1 pts (B better in 6 of 10 runs) | 0.0 | +0.025 |

Per class (pooled):

| Class | Recall A → B | Precision A → B |
|---|---|---|
| Medium | 0.45 → 0.52 | 0.76 → 0.76 |
| High | 0.77 → 0.63 | 0.54 → 0.62 |
| Very high | 0.18 → 0.43 | 0.17 → 0.19 |

Confusion matrices: `reports/confusion_matrix_ablationA*.png`.

**Interpretation: inconclusive. There is no reliable evidence of improvement.**
* The mean gain (+5 points balanced accuracy) is a third of the run-to-run spread of the difference (±15).
  A paired t-test gives t ≈ 1.1 (p ≈ 0.3), and a sign test over the 9 non-tied runs gives p ≈ 0.18. Both
  overstate certainty, because the two seeds share folds.
* Plain accuracy is identical. B's balanced-accuracy advantage comes mostly from more Very-high recall, a class
  with only 8 fabrics.
* One fold (fold 0) supplies most of the gain. Fold 3 goes the other way.
* In two A runs (folds 0 and 1, seed 42), validation picked the epoch-1 checkpoint (training accuracy
  25–44 %). Tiny validation folds make model selection noisy, and that noise can create or erase differences
  of this size.
* Conclusion: the former-conflict photographs neither clearly help nor clearly hurt. With 52 fabrics, an
  effect of a few points cannot be resolved.

## 7. Comparison with the original Phase 2 result (caveats)

Phase 2 reported 53.9 ± 5.8 % balanced accuracy and 0.462 macro F1 (four bands, 5-fold CV). **That is not
comparable to the 51.8 % here:**

| | Phase 2 (`band4_strong`) | Stage B (corrected) |
|---|---|---|
| Task | 4 bands (chance 25 %) | 3 bands, Low excluded (chance 33.3 %) |
| Data | 638 images, swatch groups clustered **within** each class | 129 photographs, **dataset-wide** fabric identity, CONFIRMED/DISTINCT only |
| Label integrity | included all 27 fabrics later found under contradictory bands | contradictory and unresolved fabrics excluded |
| Leakage control | within-class grouping; a known cross-fold leak (chevron knit) | zero fabric IDs, photos or hashes shared across folds |
| Seeds | 1 per fold | 2 per fold |

The Phase 2 figure mixed two effects that can't be separated now:
* contradictory labels across folds, which push scores down;
* within-band cross-class leakage, which pushes them up.

Stage B is the first evaluation with a verified fabric-disjoint protocol, at the cost of a much smaller
dataset and a coarser task. The Stage B numbers should be treated as the reference, together with their
uncertainty.

## 8. Limitations

* **Too few independent fabrics:** 52 in total, 8 for Very high, 2 for Low. Fold-to-fold spread (±17 points)
  dominates every comparison.
* **Label validity:** band consistency only shows that the dataset does not contradict itself for a fabric. It
  does not show the composition is correct. 27 fabrics were labelled contradictorily, so systematic labelling
  problems are likely elsewhere too.
* **Identity uncertainty:** fabric identities come from one visual review of thumbnails. Look-alike fabrics
  were excluded conservatively, which shrank the data. Residual misgrouping is possible.
* **Visual similarity:** dyed solid fabrics of different blends look alike, and colour carries little
  composition information.
* **Imbalance:** Very high is a quarter the size of the other classes. The class weighting makes the model
  over-predict it.
* **RGB limits:** cotton fraction is a chemical property. RGB close-ups capture weave, sheen and texture, which
  correlate only loosely with blend. Even a perfect classifier on this data would be estimating appearance.
* **Small validation folds:** checkpoint selection is noisy (two runs selected epoch 1).

## 9. Recommendations

1. **Collect verified samples before further modelling.**
   * Target at least 30 independent physical fabrics per band, including Low. For each fabric record a unique
     ID, the composition from the care label or a lab test, and 5–10 photos under controlled lighting and
     distance.
   * Keep one label per physical fabric.
   * This matters more than any model change.
2. **Keep this protocol fixed:** global fabric IDs, fabric-disjoint folds, at least 2 seeds, mean ± std, paired
   comparisons. Re-evaluate every future model this way.
3. **Do not change the architecture to compensate.** The bottleneck is fabric diversity and label validity,
   not model capacity.
4. **Before any deployment**, show the top-2 bands with probabilities, and treat Very-high predictions as
   unreliable.
5. **Optional:** the family-safe sensitivity analysis (236 photographs) could test whether the conservative
   identity policy cost performance. It was kept separate and not run, as instructed.

## Files

* Splits and integrity: `splits/`.
* Per-run configurations: `configs/`.
* Checkpoints: `models/{baseline,ablationA}/fold*_seed*.pth` (20 × 94 MB, git-ignored).
* Per-run history, curves, predictions and summaries: `evaluation/`.
* Aggregates: `reports/`.
* Logs: `logs/`.
* Code: `scripts/stage_b.py` (`prepare`, `run --variant`, `report`).
