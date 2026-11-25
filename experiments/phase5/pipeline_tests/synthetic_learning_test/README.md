# SYNTHETIC — PIPELINE TEST (learnability check)

> **Not evidence of fabric-composition accuracy.** Every image here is procedurally generated stripes. No real images,
> real dataset paths or real composition labels were used. This is separate from E0.

**Purpose:** show that the Phase 5 training pipeline can learn a signal when one exists. It runs the real path:
synthetic Phase 4 registry → hash-verified snapshot → sealed lockbox → fabric-disjoint grouped CV (3 folds) → guarded
training → fabric-level evaluation.

**Signal:** the stripe frequency doubles from class to class (0.08 / 0.16 / 0.32 / 0.64 rad/px). Colour, angle,
phase, supplier, device, lighting and session are random, so colour carries no information. Each band has 24 fabrics
with 4 photos each; 20 % went to the lockbox.

**Recipe:** ResNet-50 from scratch (no pretrained weights, nothing downloaded), 64 px, 30 epochs, protocol
augmentation (strong colour). Synthetic-only `recipe_overrides`: learning rate 1e-3 for backbone and head, because
the protocol's 1e-4 is a fine-tuning rate for ImageNet weights; early stopping off. `train_split` refuses overrides
for any non-synthetic snapshot.

Reproduce: `python -m training.phase5.synthetic_learning --work <scratch dir> --device cuda`

## Result (pipeline test)
| | fold 0 | fold 1 | fold 2 |
|---|---|---|---|
| Training loss, epoch 1 → min | 1.80 → 1.08 | 1.85 → 0.97 | 2.05 → 0.98 |
| Held-out fabric balanced accuracy | 0.70 | 0.46 | 0.55 |

* Pooled held-out balanced accuracy: **0.58** (95 % bootstrap CI 0.47–0.68). Chance is 0.25.
* The training loss decreased in every fold, and held-out accuracy is above chance.
* Pooled confusion matrix (rows = true low, medium, high, very_high):
  `[[12,5,2,0],[2,11,5,1],[4,1,6,8],[0,1,3,15]]`.
* Files: `SYNTHETIC_PIPELINE_TEST_learning_results.json` and `SYNTHETIC_PIPELINE_TEST_learning_curves.png`.

## Finding: MobileNetV3-Large from scratch collapses in eval mode
With the same data and recipe, MobileNetV3-Large reduced the training loss (1.54 → 0.93) but predicted a single class
for every held-out image (balanced accuracy 0.25). The same checkpoint run in train mode produced varied predictions.
The cause is torchvision's MobileNetV3 BatchNorm momentum (0.01): with only a few hundred optimisation steps, the
running statistics do not match the weights.

This is a property of training from scratch on very small data. It does not affect ImageNet-pretrained fine-tuning,
the protocol's recipe for real data, as much, but it should be watched: check that predictions per class are not
degenerate in every run. Record: `SYNTHETIC_PIPELINE_TEST_mobilenet_v3_eval_collapse_results.json`.
