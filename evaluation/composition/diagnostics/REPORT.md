# Diagnostic audit — composition checkpoint `models/composition/best_model.pth`

Read-only audit. No retraining, no weight, label or split changes
(checkpoint MD5 `50290e1d…5ef5` and split manifest MD5 `c5c94324…b7bd` identical before and after).
Reproduce with `python -m training.composition.diagnose`; look up any image with
`python -m training.composition.diagnose lookup <image> ...`.

All metrics: same checkpoint (best epoch 13), same eval preprocessing (resize 366, centre-crop 320,
ImageNet normalisation), fp32.

## 1. Overall

| Split | Images | Accuracy |
|---|---|---|
| Train (eval preprocessing, no augmentation) | 445 | **84.9 %** |
| Validation | 100 | **28.0 %** |
| Test | 93 | **32.3 %** |

Accuracy on every file in `Original/` (1,300), grouped by what the file is:

| Membership | Files | Accuracy vs folder label |
|---|---|---|
| Training image (or byte/near duplicate of one) | 445 + 95 | 84.9 % / 86.3 % |
| Validation image (or duplicate) | 100 + 13 | 28.0 % / 7.7 % |
| Test image (or duplicate) | 93 + 16 | 32.3 % / 18.8 % |
| Excluded: same image exists under ≥2 cotton labels | 538 | 18.6 % |

Uploading files straight from `Original/` mostly measures **which split the file is in**: 41.5 % of the
folder (540 files) are training images or copies of them, and the model gets those right about 85 % of the time.

## 2. Per-category performance (test split)

| Category | Test n | Correct | Wrong | Acc / Recall | Precision | F1 | Times predicted | Most confused with | Train acc | Val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 % | 9 | 0 | 9 | 0.00 | 0.00 | 0.00 | 0 | 40 % / 53 % / 60 % / 80 % (2 each) | 0.74 | 0.00 |
| 40 % | 9 | 1 | 8 | 0.11 | 0.14 | 0.13 | 7 | 53 % (6) | 0.89 | 0.00 |
| 50 % | 8 | 7 | 1 | 0.88 | 0.47 | 0.61 | 15 | 99 % (1) | 1.00 | 0.56 |
| 53 % | 7 | 1 | 6 | 0.14 | 0.10 | 0.12 | 10 | 40 % (3) | 0.76 | 0.14 |
| 58 % | 7 | 5 | 2 | 0.71 | 0.56 | 0.63 | 9 | 99 % (2) | 0.88 | 0.38 |
| 60 % | 2 | 0 | 2 | 0.00 | 0.00 | 0.00 | 3 | 99 % (2) | 1.00 | 0.00 |
| 63 % | 2 | 0 | 2 | 0.00 | 0.00 | 0.00 | 3 | 58 % (2) | 1.00 | 0.00 |
| 65 % | 6 | 0 | 6 | 0.00 | 0.00 | 0.00 | 4 | 80 % (5) | 0.65 | 0.00 |
| 66 % | 9 | 2 | 7 | 0.22 | 0.50 | 0.31 | 4 | 65 % (3) | 0.93 | 0.55 |
| 80 % | 9 | 8 | 1 | 0.89 | 0.42 | 0.57 | **19** | 95 % (1) | 0.79 | 0.67 |
| 95 % | 9 | 3 | 6 | 0.33 | 0.75 | 0.46 | 4 | 50 % (3) | 0.93 | 0.33 |
| 98 % | 9 | 1 | 8 | 0.11 | 0.25 | 0.15 | 4 | 99 % (4) | 0.68 | 0.33 |
| 99 % | 7 | 2 | 5 | 0.29 | 0.18 | 0.22 | 11 | 98 % (2) | 0.94 | 0.14 |

Full tables: `per_class_metrics_{train,val,test}.csv`. Confusion matrices:
`confusion_matrix_{train,val,test}.png` (+ `_normalized`).

## 3. 30 % cotton

* Clean images: **43 train / 9 val / 9 test** (of 100 files: 37 are duplicates, 2 label conflicts).
* Swatch groups: train has only **2 groups, both warm-coloured** (g00 orange, 11 images; g01 red/pink, 32).
  Validation is teal/green + grey; test is **grey gauze (6) + purple (3)**. See `sheets/30cotton.jpg`.
* Colour statistics: mean saturation **0.70 in train vs 0.17 in test**.
* Class weighting: weight 0.80 (30 % has more training images than average). It behaves as designed and is
  not the cause.
* Learned features: 74 % on its own training images, 0 % on val and test, **never predicted once on the
  test set**. Probability given to the true class on test averages 0.015. Fine-tuned-feature similarity of
  test 30 % images to their best training 30 % image is 0.24, the lowest of any class.
* Conclusion: the model learned "warm, saturated colour ⇒ 30 %", which is what the 30 % training swatches
  have in common. No grey or purple 30 % fabric exists in its training data, so it cannot recognise them.

## 4. 80 % and 99 % cotton

**80 % looks reliable for three reasons, only one of them genuine:**
1. *Over-prediction (sink class).* 80 % is predicted 19 times for 9 true test images (precision 0.42), and
   22 times on validation. It catches 5/6 of the 65 % test images and grey or striped fabrics from other
   classes. High recall partly comes from predicting it often.
2. *Residual swatch leakage.* Test group `80cotton_g08` (black with pink/red stripes, 3 images) is visibly
   the same fabric as training group `80cotton_g03`; the clustering split one swatch in two. 3 of the 8
   correct test predictions come from that swatch. Excluding it, 80 % is 5/6 on test.
3. *Coherent appearance (genuine).* 80 % training data is mostly yarn-dyed striped and checked shirting,
   and the other test swatches share that style.

**99 % succeeds or fails by swatch** (`per_group_test.csv`, `sheets/99cotton.jpg`):
* g05 (blue/white houndstooth): 2/3 correct, one predicted 58 %.
* g10 (grey denim twill): 0/2, both predicted 98 %. 98 % and 99 % share 30 identical images in the raw data
  and look the same.
* g07 (pink/white striped knit): 0/2, predicted 80 % and 65 %; it looks like 80 % shirting.
* Of the 100 files in `Original/99cotton`, 52 are also stored under another label (98 %, 50 %, 60/63 %).

**63 % → 58 %:** only 12 training images (83 of 100 files are duplicated in 60 %, and some in 98/99 %).
Both test images are predicted 58 %. With 2 test images this category's metrics are anecdotal.

## 5. Train vs val vs test

Train 84.9 % vs val 28.0 % vs test 32.3 %: a ~55-point generalisation gap. The network fits the swatches it
has seen and does not transfer to new swatches. Across the test set, the predicted class matches the label of
the most similar training image (fine-tuned features) in 32 % of cases, and mean test-to-train similarity (0.53)
is far below train-to-train (0.72). The model behaves like a lookup of the nearest known swatch.

## 6. Inference pipeline parity

| Comparison | Max abs. probability diff | Top class identical |
|---|---|---|
| Independent pure-PyTorch implementation vs `CompositionPredictor` (26 test images) | 0.0 | yes |
| Real Streamlit app (headless, 4 test samples) vs `CompositionPredictor` | 5e-5 (table rounding) | yes |
| `CompositionPredictor` CPU vs GPU (93 test images) | 0.013 | yes (0 flips) |
| `evaluate.py` (batched, fp16 autocast) vs fp32 (93 test images) | 0.005 | yes (0 flips) |

The app uses exactly the training-time evaluation preprocessing. Inconsistent predictions are a model and
data problem, not a pipeline bug. Your machine sets `CUDA_VISIBLE_DEVICES=-1`, so the app runs on CPU; that
changes probabilities by at most ~1 point and never the predicted class.

## 7. Causes, ranked

1. **Too few distinct swatches per class, unevenly split.** Each class has ~6–20 swatches. Where training
   happens to get only one colour family (30 %), the model learns colour, not composition.
2. **Unreliable labels in the source dataset.** 538 of 1,300 files sit under two or more cotton labels;
   60/63, 53/58, 65/66, 80/95 and 98/99 are effectively interchangeable in the raw data.
3. **Adjacent categories are not separable from appearance** (98 vs 99, 65 vs 66, 60 vs 63).
4. **Sink classes** (80 %, 50 %, 99 %) absorb uncertain predictions.
5. **Imperfect swatch grouping** inflates 80 % slightly; the effect elsewhere is small (accuracy rises only
   from 25 % to 37.5 % between the least and most train-similar test images).

## 8. Recommended corrective actions (not yet applied)

1. **Coarser label bands**, e.g. ≤40 %, 50–66 %, 80–95 %, ≥98 %, merging exactly the classes the dataset
   itself confuses. Single most likely improvement.
2. **Repeated grouped cross-validation** (e.g. 5 folds over swatch groups) instead of one 93-image test
   split, so no class is judged on 2 images and swatch draw luck (like 30 %) averages out.
3. **Manually curated swatch IDs** replacing clustering (a few hours of tagging 638 images), fixing cases
   like `80cotton_g08`.
4. **Reduce the colour shortcut**: stronger hue/saturation jitter or grayscale augmentation, then check
   whether val accuracy holds up.
5. **More distinct fabrics** for weak classes (especially 30 %, 60 %, 63 %, 65 %), ideally photographed
   under BWorth's real capture conditions.
6. In the app, show the **top-3** categories rather than one answer. On test the true class is in the top 3
   49.5 % of the time versus 32.3 % for top 1.
