# Phase 2 — four-class cotton-band experiment

All numbers come from runs in this repository (seed 42). The 13-class baseline
(`models/composition/best_model.pth`), its evaluation outputs, the dataset audit and the v1 split manifests
are byte-identical to before Phase 2 (MD5 check of 55 files).

## Setup

* **Bands** (as specified): Low = 30, 40 % · Medium = 50, 53, 58, 60, 63, 65, 66 % · High = 80, 95 % ·
  Very high = 98, 99 %.
* **Data**: the 638 cleaned originals from Phase 1. All 538 label-conflict images stay excluded;
  `Augmented/` is unused; augmentation is on the fly, training split only. Validation and test use the
  inference preprocessing (resize 366 → centre-crop 320).
* **Swatch-group correction**: the per-class galleries of the automatic clusters were reviewed by eye. 20
  merges (10 definite, 10 probable) are recorded in `configs/swatch_group_overrides.json`, taking 135 groups
  down to 114. Examples: the 80 % black/pink striped fabric that leaked into the v1 test split, the 65 % plaid,
  and a pink/white stripe filed under both 98 % and 99 %. Generic grey denims and twills could not be
  separated visually and were left as clustered.
* **Splits** (new files, v1 untouched):
  * `splits_group_v2.csv`: 479 / 75 / 84 images. A merged group goes to train if any part was in v1 train,
    otherwise to the split of its largest part. No v2 test image was a v1 training image, so the baseline
    can be scored on it fairly.
  * `cv/fold{0-4}.csv`: StratifiedGroupKFold by swatch group, stratified by cotton %. Per fold: test = fold
    k, validation = fold k+1, train = the remaining three.
* **Grid**: label scheme {13-class, 4-band} × colour augmentation {standard, strong}. The model is the same
  ResNet-50 with ImageNet weights and the same optimiser, schedule, early stopping and class-weighted loss.
  Strong colour augmentation = ColorJitter(brightness 0.4, contrast 0.4, saturation 0.8, hue 0.5) plus 30 %
  random grayscale. Standard = ColorJitter(0.3, 0.3, 0.3, hue 0.05).
* **Comparing schemes**: 13-class models are scored in the 4 bands by summing their probabilities within
  each band. Balanced accuracy is the headline metric because the Medium band holds 45 % of images: a model
  that favours Medium gets high plain accuracy without separating bands. Chance balanced accuracy is 25 %.

## 1. Did four classes improve unseen-swatch performance?

**Yes, by balanced accuracy and macro F1. No, by plain accuracy.**

5-fold swatch-grouped cross-validation, scored in 4 bands (mean ± std over folds):

| Config | Balanced acc. | Macro F1 | Plain acc. | 13-class acc. |
|---|---|---|---|---|
| 13-class, standard colour (baseline recipe) | 35.7 ± 10.1 % | 0.338 ± 0.124 | 47.4 ± 17.2 % | 17.0 ± 3.6 % |
| 13-class, strong colour | 37.3 ± 4.4 % | 0.368 ± 0.030 | 48.7 ± 12.0 % | 19.4 ± 6.0 % |
| 4-band, standard colour | 45.1 ± 2.7 % | 0.399 ± 0.069 | 43.7 ± 7.0 % | — |
| **4-band, strong colour** | **53.9 ± 5.8 %** | **0.462 ± 0.104** | 48.1 ± 9.4 % | — |

Paired by fold (identical test images), band-4 balanced accuracy:

| Comparison | Mean diff | Folds better |
|---|---|---|
| 4-band std − 13-class std | +9.5 pts | 4 / 5 |
| 4-band strong − 4-band std | +8.8 pts | 4 / 5 |
| 4-band strong − 13-class std | +18.3 pts | 4 / 5 |
| 13-class strong − 13-class std | +1.6 pts | 3 / 5 (mixed) |

Fixed v2 test split (84 unseen-swatch images):

| Model | Acc. | Balanced acc. | Macro F1 | Recall L / M / H / VH |
|---|---|---|---|---|
| Existing 13-class baseline, scored in bands | 42.9 % | 40.7 % | 0.406 | 0.11 / 0.55 / 0.50 / 0.47 |
| 4-band, standard colour | 56.0 % | 57.5 % | 0.566 | 0.39 / 0.52 / 0.75 / 0.65 |
| 4-band, strong colour | 51.2 % | 55.1 % | 0.526 | 0.72 / 0.33 / 0.50 / 0.65 |

The existing baseline reaches 21.4 % 13-class accuracy on v2 test, down from 32.3 % on v1 test. Most of the
drop comes from removing leaked swatches.

Plain accuracy did not improve under CV (47.4 % → 48.1 %), because the 13-class models predict Medium for
most images. Medium recall is 0.77 pooled, but Low recall is only 0.25 and High recall 0.13. The 4-band
models spread their predictions across the bands, which is what balanced accuracy and macro F1 reward.

## 2. Did colour augmentation help the 30 % category?

**At band level, yes for coloured fabrics, no for grey. At the exact-30 % level, no.**

Out-of-fold band predictions for every 30 % swatch group (each predicted by a model that never saw it):

| 30 % swatch | Images | 4-band std | 4-band strong | 13-class std | 13-class strong |
|---|---|---|---|---|---|
| orange (g00) | 11 | 2 | **10** | 0 | 3 |
| red / pink (g01) | 32 | 21 | **24** | 9 | 9 |
| grey gauze (g02) | 6 | **5** | 1 | 5 | 2 |
| teal / green (g03) | 7 | 0 | **5** | 2 | 0 |
| purple (g04) | 3 | 0 | **2** | 0 | 0 |
| grey, spotted (g05) | 2 | 0 | **1** | 0 | 0 |
| all grey / purple / teal (g02–g05) | 18 | 5 | **9** | 7 | 2 |

* Low-band recall over all 30 % and 40 % images: 37.9 % (4-band std) → **61.3 %** (4-band strong).
  13-class models: 25.0 % → 18.5 %.
* Teal and purple improve clearly. The grey gauze gets worse (5/6 → 1/6). With grayscale and hue jitter its
  texture now resembles other bands' grey fabrics.
* Exact 30 % recognition stays at 3/61 for both 13-class variants. Colour augmentation alone does not teach
  the model to tell 30 % from 40 %.
* The strong-colour model's training/validation gap shrinks a lot. At the best epoch: train 64 %, val 60 %
  (4-band fixed split), versus 69 % / 28 % for the Phase-1 baseline.

## 3. Are results consistent across folds?

* **The ranking is mostly consistent:** 4-band strong beats 4-band standard, and 4-band standard beats the
  13-class baseline recipe, in 4 of 5 folds. The exception is fold 3, where 4-band strong (46.6 %) and
  4-band standard (48.2 %) do not beat the 13-class baseline recipe (48.7 %).
* **Absolute numbers are not:** 4-band strong ranges 46.6–61.0 % balanced accuracy across folds. Per-fold
  test sets are 123–136 images, and a few very large swatches (one 30 % swatch has 32 photos, one 66 % swatch
  41) dominate whichever fold they land in.
* **The fixed split disagrees** on standard vs strong (57.5 % vs 55.1 %). That is one 84-image split; CV
  covers 5 × ~127 images with every swatch tested once, and is the more reliable evidence.
* Each config was trained once per fold with one seed. Seed-to-seed variance is not measured.

## 4. Which model is better?

**For the BWorth use case (unseen garments): the 4-band, strong-colour model.** It has the best
cross-validated balanced accuracy (53.9 % vs 35.7 % for the baseline recipe), beats the baseline recipe in
4/5 folds, and has the best macro F1 and the smallest overfitting gap.

**Caveats:**
* It is still weak. On unseen swatches it gets roughly half the bands right. Medium (the largest band) has
  only 33 % recall, and ~29 % of Medium images go to Very high.
* It answers a coarser question. If a 13-level answer is required, the 13-class model is the only option,
  and its unseen-swatch exact accuracy is 17–21 %.
* The model was chosen by CV results. The fixed-split test numbers are therefore not a fully independent
  confirmation.

The app offers both models. **The 13-class baseline remains the default**; the 4-band model is labelled
"experimental".

## 5. Is more data collection necessary?

**Yes. It is the main lever left.**

* The ranking holds across folds, but absolute performance swings by ±6–10 points depending on which swatches
  land in test. That is a sign of too few distinct fabrics: ~114 swatch groups across 13 percentages, some
  with 5–6 groups.
* Errors follow individual fabrics (whole swatches right or wrong). More photos of the same swatches will not
  help; more *distinct* fabrics per band will.
* Priority: Low and High bands (fewest swatches, most confusion with each other: 23–28 % of each goes to the
  other), plus non-saturated fabrics in every band, so colour stops separating bands.
* Cheap next step before collecting: **216 of the 240 excluded conflict clusters (216 unique images) carry
  labels that all fall in one band** (for example 60 % vs 63 %). Under the 4-band scheme they are not
  conflicts. Adding them back, with swatch grouping, would add ~34 % more images. It was not done here
  because the brief required excluding all conflicts.

## Files

* Models: `models/composition/band4_standard.pth`, `models/composition/band4_strong.pth`
  (baseline `best_model.pth` untouched).
* Fixed-split evaluation: `evaluation/composition/{baseline13_on_v2, band4_standard_v2, band4_strong_v2}/`
  (metrics, classification reports, confusion matrices, training curves).
* Cross-validation: `evaluation/composition/cv/` — `cv_summary.json`, `per_fold_metrics.csv`,
  `per_group_results.csv`, per-config pooled confusion matrices, per-fold curves and predictions.
* Splits: `processed/composition/splits_group_v2.csv`, `processed/composition/cv/fold*.csv`;
  overrides in `configs/swatch_group_overrides.json`.
* Code: `training/composition/experiment_splits.py`, `training/composition/cross_validate.py`; label-scheme
  and colour-aug options in `configs/config.py`, `train.py`, `transforms.py`, `evaluate.py`, `predict.py`;
  model selector in `app/streamlit_app.py`.

Reproduce:
```
python -m training.composition.experiment_splits
python -m training.composition.train --label-scheme band4 --color-aug strong --splits-path processed/composition/splits_group_v2.csv --run-name band4_strong_v2 --checkpoint-path models/composition/band4_strong.pth
python -m training.composition.cross_validate run
python -m training.composition.evaluate --splits-path processed/composition/splits_group_v2.csv --output-name baseline13_on_v2
```
