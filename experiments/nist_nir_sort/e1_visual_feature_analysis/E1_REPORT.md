# NIST Experiment E1: visual feature analysis (exploratory)

- **Date:** 27 November 2026
- **Status:** exploratory feature analysis only. No classifier was trained, no model was fine-tuned and no accuracy is reported.
- **Scripts:**
  - `extract_e1.py`: validation, embeddings and colour features.
  - `analyse_e1.py`: PCA, confound analysis and magnification analysis.
- **Raw results:** `analysis_results.json`. Run details: `experiment_metadata.json`.
- **Data source:** K.E. Goodge, C.J. Vederman, R.V. Patel, C.M. Wentz, Z. Tsinas, A.K. Landauer, A.L. Forster, "Near Infrared Spectra of Origin-defined and Real-world Textiles (NIR-SORT)", National Institute of Standards and Technology, Version 2.0.0, https://doi.org/10.18434/mds2-3325. The embeddings, colour features and plots here are derived from NIST's colour microscopy images; the NIST data itself is not redistributed in this repository.

## Conclusion: STRONG COLOUR/SOURCE CONFOUNDING

Frozen ImageNet ResNet-50 and EfficientNet-B0 features do capture real, repeatable visual structure. Both architectures agree on how swatches relate to each other, and the six views of a swatch are much closer to each other than to other swatches.

That structure, however, follows **source, dye class and fabric construction**, which are all confounded with cotton content in this dataset. Two observations are decisive:

1. **The broad subsets.** In the all-swatch, supplier-documented and Source-A-only subsets, the cotton/no-cotton separation coincides with dye chemistry, as the metadata shows. In Source A, all 31 cotton swatches carry cellulosic dyes and none of the 20 non-cotton swatches is cotton-dyed except the 3 viscose swatches.
2. **The cleanest subset.** In the undyed white Source B subset, colour-only features separate cotton from no-cotton *more* than either network. After the top colour components are regressed out, the networks' cotton separation **disappears** (silhouette about −0.09, permutation p ≈ 1.0).

Nothing here is evidence that the representations encode fibre composition.

---

## 1. Dataset validation

| Check | Result |
|---|---|
| TIFF files found under the NIST colour folders | **564** |
| Filenames parsed (`<SampleID>T_<rr>C.tiff`) | 564 of 564 |
| Swatches with images | **94**, all present in `nist_composition_inventory.csv` |
| Images per swatch | **6 for all 94** (replicates 01 to 06); the inventory's `n_images` agrees for every row |
| Source folder matches the sample ID's source letter | 564 of 564 |
| Image format, asserted per image | RGBA, 2448 × 2048, alpha constant 255 |
| Missing metadata (source, cotton %, other fibres, evidence, band, strict group) | 0 missing in any field |
| Strict provenance groups | 69 |
| Validation problems | **none** |

**Exclusions.** No swatch was excluded from the embedding or colour analyses. Labels were excluded only where a statistic needed them, and the exclusions are recorded per test in `analysis_results.json` under `dropped_groups_by_label`:

- **Dye class:** the 23 C/D groups have dye type "unknown" and were left out of dye-class statistics. The single acrylic group A35 ("other") was dropped as a one-group class.
- **Mixed groups:** 4 strict groups contain both Source A and Source B swatches (the 400M, 365, 266 and 527 catalogue groups). They were left out of the source and dye-class statistics, where they have no single label. They stay in the cotton statistics, because cotton status is consistent within every group.
- **Heavy clipping:** eleven swatches flagged for exposure clipping in the feasibility report were kept and remain flagged. No sensitivity run excluding them was done.

**Unit of analysis.**
- Plots show one point per **swatch**.
- Statistics use one point per **strict provenance group**: the mean of its swatches' standardised features.
- Images are never treated as independent fabrics. The only image-level analysis is the comparison of views in §7.

## 2. Architectures and pretrained weights

| | ResNet-50 | EfficientNet-B0 |
|---|---|---|
| torchvision builder | `resnet50` | `efficientnet_b0` |
| Weights | `ResNet50_Weights.IMAGENET1K_V2` | `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| File | `resnet50-11ad3fa6.pth` (already cached) | `efficientnet_b0_rwightman-7f5810bc.pth` (downloaded for E1 with the project owner's approval, 21.4 MB, hash-checked) |
| File SHA-256, first 12 characters | `11ad3fa62ca7…` | `7f5810bc96de…` |
| ImageNet-1K top-1 accuracy (torchvision metadata) | 80.858 | 77.692 |
| Head removed | `fc` → Identity | `classifier` (Dropout + Linear) → Identity |
| Embedding | 2048-d global-average-pooled | 1280-d global-average-pooled |
| Fine-tuned | No (eval mode, `torch.inference_mode`) | No |

The full SHA-256 values, URLs and preprocessing descriptions are in `experiment_metadata.json`.

**Software:**
- Python 3.12.5
- torch 2.11.0+cu128, torchvision 0.26.0+cu128
- numpy 2.5.5, pandas 3.1.1, Pillow 13.0.0
- scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2

**Hardware:**
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA), batch size 8. Extraction took 131 s for both models together.
- The shell environment sets `CUDA_VISIBLE_DEVICES=-1`. It was overridden to `0` for the extraction process only.

## 3. Feature extraction

1. **Loading:** each TIFF is opened with Pillow. RGBA mode and constant alpha are asserted, then the image is converted to RGB. Dropping a constant-255 alpha loses nothing, and RGB colour is kept.
2. **Preprocessing:** each model uses its own official transform, `weights.transforms()`:
   - ResNet-50 V2: resize the short side to 232 (bilinear), centre crop 224, normalise with ImageNet mean and std.
   - EfficientNet-B0 V1: resize to 256 (bicubic), centre crop 224, same normalisation.
   - Input to both networks: 3 × 224 × 224.
3. **Embeddings:** one embedding per image (564 rows per model). The **swatch embedding** is the arithmetic mean of a swatch's six raw image embeddings (94 rows).
4. **Before PCA and statistics:** features are standardised (z-scored) within each analysed subset. Silhouettes use cosine distance on the standardised group means.

**Resolution limitation [observation from the documented scale].**
- ResNet-50 resizes the 2048 px short side to 232 px, so one input pixel covers about 26 µm at the lowest magnification (replicates 01 and 04) and about 12 µm at the highest (03 and 06). EfficientNet-B0 resizes to 256 px: about 23 µm and 10 µm.
- Cotton fibres are roughly 12 to 22 µm across. At the input resolution used here, individual fibres are at most 1 to 2 px wide.
- So E1's features mostly describe yarn, weave and knit structure and colour, not fibre morphology.

## 4. Colour-only baseline

- **Features:** 95 colour features per image, computed at quarter resolution (612 × 512):
  - mean and standard deviation of R, G and B;
  - 16-bin histograms for each RGB channel;
  - HSV histograms (18 hue, 8 saturation and 8 value bins);
  - brightness, value and saturation statistics;
  - highlight and shadow clipping fractions.
- **Swatch level:** the mean of the six images' features.
- **Output:** `colour_features.csv`, with image-level rows (`level=image`) and swatch-level rows (`level=swatch`).

**Results (observations)** are in the colour row of each table in §8:
- The colour features separate **Source B from Source A** strongly. In the supplier subset, the source silhouette is **0.346** (p = 0.001), the largest value in the experiment.
- They separate dye classes moderately: silhouette 0.10 to 0.11 in the all-swatch and supplier subsets.
- Across the broad subsets they do **not** separate cotton from no-cotton: silhouette 0.002 to 0.004, p > 0.3. Cotton and non-cotton swatches share hues within Source A.
- Inside Source B, which is all white, colour features *do* separate cotton from no-cotton (§6). There, "colour" means brightness, clipping (glare) and slight tint differences such as optical brightener.

Colour separation is not evidence of fibre identification.

## 5. PCA observations

The plots are `pca_resnet50.png`, `pca_efficientnet_b0.png` and `pca_colour_baseline.png`:

- one point per swatch, n = 94;
- colour = cotton status, or dye class in the right-hand panels;
- marker = source;
- hollow marker = care-label evidence;
- star = C02, the lab-confirmed sample;
- dark outline = Source B.

**Observations:**
- **ResNet-50:** PC1 (17%) and PC2 (6%) are dominated by a few outlying, strongly patterned or textured swatches: D04 "shadow box", D07 pinstripe, D08, C09, C10 and A51 wool flannel. Most swatches crowd near the origin. The main axes reflect pattern and texture extremes, not composition.
- **EfficientNet-B0:** PC1 is 13% and PC2 9%.
  - Source B fabrics (white) form a band along the top.
  - The C/D knit garments sit to the right.
  - Source A swatches spread through the middle and bottom.
  - Within Source A, cotton and non-cotton swatches overlap heavily.
  - Recoloured by dye class, the layout follows dye class and source at least as closely as cotton status.
- **Colour baseline:** PC1 (23%) runs from dark and saturated (left) to light and white (right). Source B lines up at the right. PC2 (18%) tracks hue. Cotton status is mixed throughout.
- **C02**, the only lab-confirmed sample: in both CNN embeddings, its nearest neighbours are mostly other Source C garment knits. In the full feature space the closest are C01, C03 and C06, with a few Source A swatches (A07, A15, A53) also nearby. It does not sit with the bulk of the Source A cotton swatches. **[Hypothesis]** That probably reflects its knit construction and garment origin, not its fibre.

## 6. Source B analysis

`source_b_comparison.png` shows PCA fitted within each subset. Labels show the supplier-stated cotton percentage, and dashed lines join members of the same strict group.

| Subset | Swatches | Strict groups | Cotton-containing | No cotton |
|---|---|---|---|---|
| C. Source B | 20 | 17 | 10 swatches / 7 groups | 10 swatches / 10 groups |
| D. Source B, undyed white (`Dye Type` = none and `Visual Inspected Color` = White) | 18 | 16 | 9 / 7 | 9 / 9 |

Subset D excludes B11 (greige, "Light Brown") and B18 (wool, "Light Yellow"). Its members:
- **Cotton-containing:** B01 (92%), B04, B06, B08 (50%), B05 and B07 (35%, one group), B12 and B15 (100%, one group), and B14 (100%).
- **No cotton:** B02, B03, B09, B10, B13, B16, B17, B19 and B20.

**Cotton/no-cotton silhouettes** (group level; permutation p in brackets):

| Subset | ResNet-50 | EfficientNet-B0 | Colour-only | ResNet-50 after removing 5 colour PCs | EfficientNet-B0 after removing 5 colour PCs |
|---|---|---|---|---|---|
| C. Source B (17 groups) | 0.102 (0.004) | 0.073 (0.044) | **0.154** (0.035) | −0.085 (1.0) | −0.097 (1.0) |
| D. Undyed white B (16 groups) | 0.102 (0.007) | 0.072 (0.056) | **0.256** (0.011) | −0.088 (0.996) | −0.099 (0.997) |

**Observations:**
- In the subset with the least confounding, colour and brightness features separate cotton from no-cotton more than either network.
- The networks' separation vanishes once 5 colour principal components are regressed out.
- In the plots:
  - ResNet-50 squeezes most Source B swatches together, with B09 and B16 as outliers.
  - EfficientNet-B0 places the 50% knits B01, B04 and B08 near the polyester/spandex knit B02 (construction).
  - The woven cottons and blends (B05, B06, B07, B12, B14, B15) sit near the viscose B17.
- Construction (knit vs woven) separates comparably to cotton status in this subset. For EfficientNet-B0 it is larger (0.183 vs 0.072); for ResNet-50 (0.074 vs 0.102) and colour (0.218 vs 0.256) it is somewhat smaller. Knit/woven and cotton status overlap in these 16 groups, so they can't be fully separated.

**Hypothesis, not tested here:** in white fabrics, the brightness and glare differences that colour features capture may partly reflect real fibre or yarn properties, for example the sheen of continuous-filament synthetics such as B10 pongee, B19 tricot and B20 taffeta, against spun cotton. They may equally reflect finish (optical brightener) and per-sample exposure settings. With 16 groups, these explanations can't be separated.

**Small numbers.** The groups number 7 against 9. Silhouette values around 0.1 to 0.25 on so few groups are weak structure. The permutation p-values describe this sample only and support no general claim.

## 7. Magnification and side

`magnification_comparison.png` analyses images by view. It is image-level only to compare views; the main analysis averages the six views.

| Statistic | ResNet-50 | EfficientNet-B0 | Colour-only |
|---|---|---|---|
| Share of feature variance between swatches | 36% | 51% | 92% |
| Share of feature variance between views (view means) | 2% | 4% | 0.3% |
| Mean cosine similarity: same swatch, front vs back at the same magnification | 0.57 | 0.74 | 0.92 |
| Mean cosine similarity: same swatch, same side, different magnification | 0.29 | 0.39 | 0.93 |
| Mean cosine similarity: different swatches, same view | 0.04 | 0.04 | −0.01 |
| Low vs high magnification, front: agreement of inter-swatch distances (Spearman) | 0.42 | 0.49 | 0.89 |
| Front vs back, low magnification: same agreement measure | 0.64 | 0.81 | 0.89 |

**Observations:**
- The views of one swatch are far more alike than views of different swatches (0.29 to 0.74 against about 0.04). The six views clearly belong to one fabric.
- **Magnification changes the CNN representation more than side does.** Front and back at the same magnification are close (0.57 / 0.74). Low and high magnification on the same side are much less similar (low vs high about 0.13), and they agree only moderately on which swatches resemble each other (Spearman 0.42 / 0.49).
- The views therefore carry **different, partly complementary texture information**.
- For colour features, all six views are almost interchangeable (cosine ≥ 0.89).
- No view-specific cluster appears: the image-level silhouette by view is about −0.02 to −0.03.
- Averaging the six views is a reasonable swatch summary, but it mixes scales.

**Hypothesis:** a model meant to capture fibre-scale texture should use the high-magnification views (03 and 06), at native resolution. See §3 for the resolution limit.

## 8. Confound analysis

**Silhouette by metadata label** (strict-group level, cosine; p from 999 group-level label permutations):

| Subset (swatches / groups) | Features | Cotton | Source | Dye class | Construction |
|---|---|---|---|---|---|
| **A. All** (94 / 69) | ResNet-50 | 0.035 (0.002) | 0.081 (0.001) | 0.098 (0.001) | 0.111 (0.001) |
| | EfficientNet-B0 | 0.071 (0.001) | 0.102 (0.001) | **0.209** (0.001) | 0.152 (0.001) |
| | Colour | 0.004 (0.32) | 0.046 (0.001) | 0.108 (0.001) | 0.014 (0.12) |
| **B. Supplier-documented** (71 / 46) | ResNet-50 | 0.083 (0.001) | 0.094 (0.001) | 0.069 (0.001) | 0.054 (0.013) |
| | EfficientNet-B0 | 0.137 (0.001) | 0.172 (0.001) | **0.188** (0.001) | 0.083 (0.001) |
| | Colour | 0.002 (0.38) | **0.346** (0.001) | 0.104 (0.001) | 0.069 (0.004) |
| **C. Source B** (20 / 17) | ResNet-50 | 0.102 (0.004) | — | — (all undyed) | 0.055 (0.043) |
| | EfficientNet-B0 | 0.073 (0.044) | — | — | 0.148 (0.006) |
| | Colour | 0.154 (0.035) | — | — | 0.182 (0.018) |
| **D. Undyed white B** (18 / 16) | ResNet-50 | 0.102 (0.007) | — | — | 0.074 (0.026) |
| | EfficientNet-B0 | 0.072 (0.056) | — | — | 0.183 (0.003) |
| | Colour | **0.256** (0.011) | — | — | 0.218 (0.011) |
| **E. Cotton vs no cotton within Source A** (51 / 33) | ResNet-50 | 0.110 (0.001) | — | 0.083 (0.001) | — (32 of 33 woven) |
| | EfficientNet-B0 | 0.204 (0.001) | — | 0.154 (0.001) | — |
| | Colour | 0.073 (0.016) | — | −0.037 (0.09) | — |

Analysis E, cotton against no cotton, is covered by the cotton column of every row. The Source-A-only row is the within-source version.

**Other measures:**

| Subset | Features | Correlation with colour distances (Mantel Spearman) | Cotton silhouette after removing 5 colour PCs |
|---|---|---|---|
| All | ResNet-50 | 0.19 | 0.030 (p 0.004) |
| | EfficientNet-B0 | 0.29 | 0.077 (p 0.001) |
| Supplier | ResNet-50 | 0.30 | 0.070 (p 0.001) |
| | EfficientNet-B0 | 0.41 | 0.137 (p 0.001) |
| Source B | ResNet-50 | 0.42 | −0.085 (p 1.0) |
| | EfficientNet-B0 | 0.47 | −0.097 (p 1.0) |
| Undyed white B | ResNet-50 | 0.35 | −0.088 (p 0.996) |
| | EfficientNet-B0 | 0.39 | −0.099 (p 0.997) |
| Source A | ResNet-50 | 0.29 | 0.049 (p 0.008) |
| | EfficientNet-B0 | 0.52 | 0.115 (p 0.001) |

ResNet-50 and EfficientNet-B0 agree with each other: Mantel 0.63 to 0.68 in every subset.

**Interpretation:**
- **Observation:** across all swatches, both CNNs separate dye class and source more than cotton status (ResNet-50: dye 0.098, source 0.081 vs cotton 0.035; EfficientNet-B0: 0.209, 0.102 vs 0.071). In the supplier subset this holds for EfficientNet-B0 (dye 0.188, source 0.172 vs cotton 0.137). For ResNet-50 there, source is only slightly above cotton (0.094 vs 0.083) and dye class is below it (0.069). In the supplier subset, colour alone separates source (0.35) far more than anything separates cotton.
- **Observation, from the metadata:** within Source A, dye class is almost fully determined by fibre. Cellulosic dyes go on cotton and viscose. Disperse dyes go on polyester, and acid or metal-complex dyes on nylon and wool. Non-cotton Source A fabrics are also different constructions: nylon spun plain weave, wool flannel, viscose challis, filament polyester.
  - So the cotton separation that *survives* colour removal in subsets A, B and E (EfficientNet 0.08 to 0.14) can't be credited to fibre composition.
  - Residualising colour removes hue and brightness, not dye-related surface appearance or weave and yarn structure.
- **Observation:** in the only subset where dye and source are held constant (D), the CNN cotton separation is explained by colour features and disappears after residualising.
- **Hypothesis:** the frozen features mainly encode construction (knit vs woven, weave pattern, yarn type), surface finish and pattern, with colour and exposure as a large secondary factor. Composition-related signal, if present at all, is not separable from these with this data.

## 9. Limitations

- **Group counts.**
  - All: 69 groups (44 cotton, 25 no cotton).
  - Supplier: 46 (26 / 20).
  - Source B: 17 (7 / 10).
  - Undyed white B: 16 (7 / 9).
  - The cleanest subset has only 7 cotton-containing groups, so silhouettes and permutation p-values there are fragile.
- **Labels.** Labels are supplier specifications (A/B) or care labels (C/D). Only C02 is lab-confirmed.
- **Binary label.** The cotton/no-cotton label lumps 30% to 100% cotton together. E1 says nothing about cotton *percentage*.
- **Colour residualisation.** It uses 5 colour PCs fitted on the same few groups, which may over- or under-remove colour. It shows dependence, not causation.
- **Resolution.** Input resolution (§3) limits fibre-scale detail. Centre cropping also discards the edges of each field of view.
- **Exposure.** Per-sample exposure and white balance are documented in `ColorParameters.csv`, so colour isn't comparable across swatches.
- **Statistics.** No multiple-comparison correction was applied across the many silhouettes. Treat p-values as descriptive.

## 10. Any promising composition-related patterns?

**No credible ones.**

- **Observation:** the frozen features are informative about fabric appearance. They agree across two architectures, show strong swatch identity across views, and capture construction and dye-class structure.
- **Observation:** every cotton-related separation seen is matched or exceeded by separation along source, dye or construction. In the one controlled subset, it is explained by colour and brightness features.
- **Hypotheses worth testing,** none of them supported yet:
  1. Native-resolution, high-magnification tiles (replicates 03 and 06) may expose fibre-scale texture that the 224-px resize removes.
  2. Colour-invariant input (greyscale or per-image normalised) may reduce the dye and exposure confounds.
  3. Filament-vs-spun sheen may be a genuine visual cue for synthetics. It would still not measure cotton percentage.

## Recommended next experiment (E2, still exploratory)

Run a **texture-focused, colour-controlled re-extraction**, restricted to the controlled subsets:

1. **Input:** native-resolution 224 × 224 tiles taken from the high-magnification views (03 and 06). That is about 1.3 µm per pixel, with many tiles per swatch averaged to swatch level. Use greyscale or per-image colour-normalised input.
2. **Subsets:**
   - undyed white Source B, 16 groups;
   - within-dye-class comparisons in Source A, for example cotton vs viscose inside cellulosic dyes. That is 3 viscose swatches, a single group, so it is descriptive only.
3. **Controls:** keep the colour-only baseline and the colour-residualisation check, and use the same strict-group statistics.
4. **What would count as progress:** a cotton-status signal in undyed white B that survives colour residualisation under *both* architectures. Even then, it would be a hypothesis for BWorth's own lab-verified collection, not evidence of composition prediction.

E1 doesn't change the feasibility conclusion: NIST NIR-SORT is for exploratory analysis only and doesn't reduce BWorth's 120-fabric collection target.

---

### Output files (this directory)

| File | Contents |
|---|---|
| `E1_REPORT.md` | This report |
| `resnet50_image_embeddings.csv` | 564 rows: image_id, swatch_id, replicate, side, magnification, scale, metadata, f0000 to f2047. **Git-ignored** (large, regenerable). |
| `resnet50_swatch_embeddings.csv` | 94 rows: swatch_id, metadata, n_images, f0000 to f2047 |
| `efficientnet_b0_image_embeddings.csv` | 564 rows, 1280-d. **Git-ignored**. |
| `efficientnet_b0_swatch_embeddings.csv` | 94 rows, 1280-d |
| `colour_features.csv` | 564 image rows and 94 swatch rows (`level` column), 95 colour features |
| `pca_resnet50.png`, `pca_efficientnet_b0.png`, `pca_colour_baseline.png` | Swatch-level PCA |
| `source_b_comparison.png` | Source B and undyed white B PCA, three feature sets |
| `magnification_comparison.png` | View analysis |
| `experiment_metadata.json` | Versions, weight files and hashes, preprocessing, device, validation, analysis settings |
| `analysis_results.json` | All statistics, class and group counts, and dropped labels |
| `extract_e1.py`, `analyse_e1.py` | Reproduction scripts (read the NIST data read-only; write only here) |
