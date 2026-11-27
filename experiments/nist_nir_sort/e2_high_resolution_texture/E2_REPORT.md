# NIST Experiment E2: high-resolution texture analysis (exploratory)

- **Date:** 27 November 2026
- **Status:** exploratory feature analysis only. No classifier was trained, no network was fine-tuned and no accuracy is reported.
- **Data source:** K.E. Goodge, C.J. Vederman, R.V. Patel, C.M. Wentz, Z. Tsinas, A.K. Landauer, A.L. Forster, "Near Infrared Spectra of Origin-defined and Real-world Textiles (NIR-SORT)", NIST, Version 2.0.0, https://doi.org/10.18434/mds2-3325. The NIST data is not redistributed here.
- **Scripts:**
  - `extract_e2.py`: selection, tiling, tile QC and features.
  - `analyse_e2.py`: group-level analysis, controls and plots.
- **Previous experiment:** `../e1_visual_feature_analysis/E1_REPORT.md`.

## Verdict: WEAK OR INCONCLUSIVE SIGNAL

At native resolution, frozen CNN features of the undyed white Source B fabrics separate the cotton-containing groups from the non-cotton groups modestly. The separation is consistent across representations and views:

- silhouette 0.11 to 0.15, exact permutation p 0.005 to 0.03;
- it survives colour-normalised and greyscale input;
- it survives regressing out brightness;
- it survives permuting labels only within knit and within woven fabrics.

It does **not** hold up as evidence of an independent cotton signal:

- **The colour-only baseline separates the groups as much or more, in silhouette terms.** Overall it gives 0.16. Its exact p (0.07) is weaker than the CNNs' (0.005 to 0.03). Within woven fabrics it gives 0.53, against 0.16 to 0.25 for the CNNs.
- **Colour removal can't confirm that the CNN separation is independent of colour.** At high magnification, regressing out three colour components leaves it non-significant (0.03 to 0.07). Those values are below the mean for regressing out three random covariates (0.07 to 0.10), but within the random-covariate 5–95% ranges. With 16 groups, this test can't tell a colour effect from a loss of degrees of freedom. At medium magnification, colour removal turns every CNN silhouette negative, below the random-covariate range.
- **It is driven mainly by four woven cotton groups.** These are woven print cloths and broadcloths of 100 to 107 gsm from one supplier, probably plain weave (the weave isn't documented for B06 or B14). The texture features strongly encode fabric construction identity.
- **The sample is small.** Only 16 independent groups, a knit comparison of 3 vs 3, and many correlated tests.

Some texture-only results are suggestive but inconsistent across magnifications and are single results among many tests:
- GLCM on normalised greyscale;
- GLCM on raw greyscale after colour removal (0.275, p 0.011). **This is not chemical fibre identification, and it is not evidence that texture encodes cotton content.**

---

## 1. Scope and validation

**Selection rule:**
- `source == B` in `nist_composition_inventory.csv`;
- `Dye Type == none` and `Visual Inspected Color == White` in NIST `SampleInformation.csv`.

The rule excludes B11 (greige, "Light Brown") and B18 (wool, "Light Yellow").

| | Count |
|---|---|
| Swatches | **18** |
| Independent strict provenance groups | **16** (7 cotton-containing, 9 no cotton) |
| Images | 108 (6 views × 18), each mapped to its swatch; every swatch has replicates 01 to 06 (asserted) |
| Main analysis views | high magnification, front 03 and back 06 (36 images) |
| Also analysed | medium (02, 05) and low (01, 04), for the magnification comparison |

**Groups:**

| Group | Members | Cotton | Construction | NIST notes / finish |
|---|---|---|---|---|
| SINGLE-B01 | B01 | 92% | knit | leotard fabric (cotton/spandex) |
| SINGLE-B04 | B04 | 50% | knit | interlock tubular; bleached with optical brightener |
| SINGLE-B08 | B08 | 50% | knit | jersey knit, tubular |
| TF-CAT-7409 | B05, B07 | 35% | woven | broadcloth; B05 with and B07 without optical brightener |
| SINGLE-B06 | B06 | 50% | woven | |
| TF-CAT-400M | B12, B15 | 100% | woven | print cloth; B12 bleached and desized, B15 also mercerised |
| SINGLE-B14 | B14 | 100% | woven | combed fiber; bleached |
| SINGLE-B02 | B02 | 0 (polyester/spandex) | knit | |
| SINGLE-B03 | B03 | 0 (nylon/spandex) | knit | |
| SINGLE-B19 | B19 | 0 (nylon 6) | knit | **filament fiber, tricot** |
| SINGLE-B09 | B09 | 0 (polyester) | woven | with optical brightener |
| SINGLE-B10 | B10 | 0 (polyester) | woven | pongee weave; without optical brightener |
| SINGLE-B13 | B13 | 0 (polyester) | woven | |
| TF-CAT-365 | B16 | 0 (nylon 6,6) | woven | spun fiber, plain weave; texturised |
| TF-CAT-266 | B17 | 0 (viscose) | woven | spun fiber, challis weave |
| SINGLE-B20 | B20 | 0 (nylon 6,6) | woven | **filament fiber, taffeta** |

Composition labels are supplier specifications (Testfabrics), not laboratory results.

## 2. Tiling and tile QC

- **Tiling:** non-overlapping **224 × 224 px tiles cut from the original 2448 × 2048 images**, a centred 10 × 9 grid giving 90 tiles per image. Whole images were never resized. Coordinates and parent image IDs are in `tile_inventory.csv`, one row per tile.
- **Physical size at high magnification** (1.31 µm per px): a tile covers about 0.29 × 0.29 mm, and a 12 to 22 µm cotton fibre is about 9 to 17 px wide. In E1 it was 1 to 2 px.

**Exclusion rules,** recorded per tile:

| Rule | Threshold |
|---|---|
| Highlight clipping | > 20% of pixels with max(R,G,B) ≥ 250 |
| Shadow clipping | > 20% of pixels with min(R,G,B) ≤ 5 |
| Insufficient texture or fabric | luma standard deviation < 3 |
| Poor focus | Laplacian variance < 0.25 × the image's median tile |

**Result:** 9,695 of 9,720 tiles were included (99.7%).
- All **25 exclusions were highlight clipping**: B19 (nylon filament tricot) lost 7, 8, 3 and 6 tiles in views 02, 03, 05 and 06, and B16 lost 1 tile in view 03.
- No tile failed the focus or texture rules.
- At high magnification, 3,225 of 3,240 tiles were used.

## 3. Representations and features

| Representation | Definition |
|---|---|
| A. RGB | original tile, /255, ImageNet mean/std |
| B. Greyscale | luma 0.299R + 0.587G + 0.114B, replicated to 3 channels, ImageNet mean/std |
| C. Colour-normalised | per parent image, per channel z-score using whole-image mean and standard deviation; fed directly. Removes colour cast, brightness and global contrast. |

| Extractor | Details |
|---|---|
| ResNet-50 | torchvision `IMAGENET1K_V2` (`resnet50-11ad3fa6.pth`); `fc` removed; 2048-d pooled features; native 224 tiles, no resize or crop; frozen |
| EfficientNet-B0 | torchvision `IMAGENET1K_V1` (`efficientnet_b0_rwightman-7f5810bc.pth`); classifier removed; 1280-d; frozen |
| Gabor | 4 wavelengths (4, 8, 16, 32 px, about 5 to 42 µm at high magnification) × 4 orientations; zero-mean, L2-normalised even/odd pairs; magnitude computed on the full-resolution image; tile mean and standard deviation (32 features). Computed on greyscale, and on per-image-normalised greyscale. |
| GLCM | 16 grey levels; distances 1, 2, 4, 8 and 16 px; 4 angles averaged; contrast, dissimilarity, homogeneity, energy, correlation (25 features). Implemented in NumPy, since scikit-image isn't installed. Computed on greyscale (fixed quantisation) and on per-image-normalised greyscale. |
| Colour-only baseline | tile mean and standard deviation of RGB, brightness mean and standard deviation, saturation, blue–yellow chroma, clipping fractions (12 features) |

**Aggregation and statistics:**
- tiles → image mean → swatch mean over the selected views;
- features z-scored across the 18 swatches → mean per **strict provenance group** (16);
- statistics on group means, using cosine distance.

**Statistics, all descriptive:**
- Cotton/no-cotton silhouette, with an **exact permutation p** over all 11,440 ways of assigning 7 cotton labels to 16 groups.
- Exact p stratified by construction: 4,200 assignments that keep 4 of 10 woven and 3 of 6 knit groups as cotton.
- Leave-one-group-out jackknife range, and per-group silhouette values.
- Silhouette after linear regression on the top-3 colour-baseline PCs, compared with regression on 3 random covariates (200 draws).
- Silhouette after regressing out brightness only.
- Within-woven (210 assignments) and within-knit (20 assignments) silhouettes.
- Silhouette excluding the two documented filament groups (3,432 assignments).
- Knit/woven silhouette and Mantel correlation with colour distances.

**Run details:**
- GPU: NVIDIA RTX 3050 Laptop, 4 GB. 3,163 MiB free after loading the models; peak allocated 276 MiB at batch size 16.
- `CUDA_VISIBLE_DEVICES=0` was set for this process only.
- Extraction took 547 s for 108 images; the analysis takes about 20 s.
- Software versions are in `experiment_metadata.json`.

## 4. Main results (high magnification, 16 groups)

All values are from `model_feature_summary.csv`, view `high`.

| Features | Cotton silhouette (exact p) | p stratified by construction | Jackknife range | After 3 colour PCs removed (p) | Same, 3 random covariates (mean) | After brightness removed (p) |
|---|---|---|---|---|---|---|
| ResNet-50 · RGB | 0.149 (0.006) | 0.001 | 0.130–0.181 | 0.063 (0.093) | 0.102 | 0.126 (0.012) |
| ResNet-50 · grey | 0.118 (0.015) | 0.003 | 0.099–0.152 | 0.041 (0.170) | 0.082 | 0.097 (0.031) |
| ResNet-50 · colour-norm | 0.131 (0.016) | 0.004 | 0.107–0.156 | 0.069 (0.095) | 0.095 | 0.153 (0.010) |
| EffNet-B0 · RGB | 0.131 (0.005) | 0.001 | 0.117–0.158 | 0.062 (0.075) | 0.087 | 0.120 (0.009) |
| EffNet-B0 · grey | 0.115 (0.029) | 0.006 | 0.092–0.148 | 0.063 (0.077) | 0.087 | 0.102 (0.040) |
| EffNet-B0 · colour-norm | 0.108 (0.030) | 0.021 | 0.083–0.147 | 0.030 (0.247) | 0.066 | 0.159 (0.017) |
| Gabor · grey | 0.064 (0.176) | 0.162 | 0.024–0.131 | 0.022 (0.311) | 0.061 | 0.018 (0.336) |
| Gabor · grey-norm | 0.045 (0.240) | 0.140 | 0.007–0.093 | 0.100 (0.095) | 0.031 | 0.102 (0.090) |
| GLCM · grey | 0.015 (0.323) | 0.289 | −0.024–0.049 | 0.275 (0.011) | 0.033 | 0.097 (0.150) |
| GLCM · grey-norm | 0.134 (0.081) | 0.061 | 0.098–0.215 | 0.184 (0.045) | 0.108 | 0.186 (0.042) |
| **Colour-only baseline** | **0.162** (0.070) | 0.027 | 0.118–0.242 | — | — | — |

**Construction controls** (high magnification):

| Features | Knit/woven silhouette (p) | Woven only: 4 vs 6 groups (p) | Knit only: 3 vs 3 groups (p; minimum possible 0.05) | Excluding filament B19, B20: 7 vs 7 (p) | Mantel r with colour distances |
|---|---|---|---|---|---|
| ResNet-50 · RGB | 0.118 (0.018) | 0.232 (0.010) | 0.341 (0.10) | 0.124 (0.030) | 0.61 |
| ResNet-50 · grey | 0.132 (0.009) | 0.212 (0.010) | 0.304 (0.10) | 0.092 (0.063) | 0.54 |
| ResNet-50 · colour-norm | 0.133 (0.014) | 0.246 (0.019) | 0.202 (0.20) | 0.108 (0.052) | 0.53 |
| EffNet-B0 · RGB | 0.152 (0.002) | 0.168 (0.019) | 0.395 (0.10) | 0.109 (0.029) | 0.55 |
| EffNet-B0 · grey | 0.189 (0.005) | 0.218 (0.014) | 0.388 (0.10) | 0.082 (0.096) | 0.62 |
| EffNet-B0 · colour-norm | 0.155 (0.010) | 0.161 (0.014) | 0.080 (0.30) | 0.097 (0.067) | 0.47 |
| Gabor · grey | 0.086 (0.135) | −0.055 (0.61) | 0.420 (0.20) | 0.029 (0.30) | 0.40 |
| Gabor · grey-norm | 0.340 (0.001) | 0.119 (0.15) | −0.093 (0.60) | 0.083 (0.17) | 0.32 |
| GLCM · grey | 0.265 (0.032) | 0.141 (0.18) | 0.373 (0.20) | 0.012 (0.34) | 0.76 |
| GLCM · grey-norm | 0.146 (0.063) | 0.265 (0.048) | −0.022 (0.50) | 0.195 (0.046) | 0.36 |
| Colour-only baseline | 0.314 (0.008) | **0.529** (0.010) | 0.271 (0.40) | 0.113 (0.14) | — |

## 5. Observations

1. **Native-resolution CNN features separate cotton status modestly and consistently.**
   - Silhouettes are 0.11 to 0.15 for all six CNN × representation combinations, with exact p 0.005 to 0.03.
   - Dropping any single group still leaves a positive silhouette (jackknife minimum 0.08).
   - The separation remains when labels are permuted only within knit and within woven fabrics (p 0.001 to 0.02). Knit vs woven alone does not explain it.
2. **Greyscale and colour-normalised tiles keep most of the separation.** RGB 0.13 to 0.15; greyscale 0.12; colour-normalised 0.11 to 0.13. Removing colour cast and global brightness at the input does not remove it.
3. **Brightness alone does not explain it.** After regressing out group brightness, CNN silhouettes are 0.10 to 0.16 (p 0.009 to 0.04). Recorded acquisition settings are **near-identical across the subset**. Every swatch has light preset "General" and gain NA in `ColorParameters.csv`. The balance ratio is 2.18 for B01 and 2.19 for all the others.
4. **The colour-only baseline matches or exceeds the CNNs.**
   - Overall: 0.16, p 0.07.
   - Within woven fabrics: **0.53** against 0.16 to 0.25.
   - CNN distances correlate with colour distances (Mantel r 0.47 to 0.62).
   - In these all-white fabrics, "colour" means brightness, glare and slight tint (optical brightener, bleaching).
5. **Colour removal can't confirm the CNN separation as colour-independent, and its effect depends on the view.**
   - At high magnification, regressing out the top-3 colour PCs leaves CNN silhouettes of 0.03 to 0.07, all non-significant (p 0.08 to 0.25).
   - Regressing out 3 *random* covariates (200 draws) gives means of 0.07 to 0.10. The colour-removed values are lower for all six CNN sets, but each lies inside the corresponding random-covariate 5–95% range (roughly −0.03 to 0.20).
   - So at high magnification, this test can't separate a colour effect from the loss of degrees of freedom when 3 covariates are regressed out of 16 groups.
   - At medium magnification, colour removal turns every CNN silhouette negative (about −0.06 to −0.08, p ≥ 0.95). All six values fall below the random-covariate range: there, the colour components remove more than chance would.
   - At high-front, 3 of 6 fall below the range. At low and high-back, none do.
6. **The separation is carried mainly by four woven cotton groups.** See `group_level_results.csv` and `group_level_silhouettes.png`.
   - B05/B07, B06, B12/B15 and B14 are all positive (0.16 to 0.33) for every CNN.
   - These are Testfabrics woven print cloths and broadcloths, 100 to 107 gsm (documented). Plain weave is inferred from "print cloth" and "broadcloth"; B06 has no fabric description and B14 only "combed fiber". They form a tight cluster in the CNN PCAs (`pca_cnn_high_mag.png`).
   - The knit cottons (B01, B04, B08) are weaker and inconsistent, and sit next to the polyester/spandex knit B02.
   - B02's group silhouette is negative for every CNN: it looks more like the cotton class.
   - Viscose B17, a cellulosic, is near zero or negative.
7. **The texture features strongly encode construction identity.**
   - Two pairs share a base cloth but differ in finish: B05/B07 (optical brightener) and B12/B15 (mercerised).
   - CNN, Gabor and GLCM features place these pairs among the **closest pairs** in the subset. CNNs put them at the 0th to 18th percentile of the 153 pair distances; Gabor and GLCM at the 0th to 27th (B05/B07 at ≤ 8.5, B12/B15 at 5.9 to 26.8).
   - The colour baseline places them at the 39th to 52nd percentile.
   - For these two pairs, the texture features were less affected by finish differences than the colour features. Two pairs are too few to generalise.
8. **The knit comparison is uninformative.** With 3 vs 3 groups, the smallest attainable p is 0.05; observed p values are 0.10 to 0.60.
9. **Removing the documented filament fabrics weakens the signal.** Excluding B19 and B20 lowers CNN silhouettes to 0.08 to 0.12 (p 0.03 to 0.10).
10. **Handcrafted texture results are mixed.**
    - Before colour removal, Gabor and GLCM on raw greyscale show little at high magnification (0.02 to 0.06).
    - After colour removal, two handcrafted sets *increase*, both above their random-covariate 5–95% ranges:
      - **GLCM on raw greyscale rises from 0.015 to 0.275 (p 0.011)**; the random range is −0.07 to 0.14.
      - Gabor on normalised greyscale rises from 0.045 to 0.100 (p 0.095); the random range is −0.05 to 0.10.
    - Caveats:
      - Raw-greyscale GLCM uses a fixed 0–255 quantisation, so it is exposure-sensitive. The colour components (dominated by brightness and clipping) may simply be acting as an exposure normalisation.
      - The rise isn't consistent across views. For GLCM on raw greyscale, the colour-removed value is 0.200 at low, −0.006 at medium, −0.045 at high-front and 0.260 at high-back.
      - Brightness-only regression gives only 0.097 (p 0.150).
      - It is one result among many tests, so it isn't treated as evidence of a cotton-specific signal.
    - GLCM on normalised greyscale is the only texture-only feature that holds up at high magnification:
      - 0.13 overall (p 0.08);
      - 0.18 after colour removal (p 0.045);
      - 0.19 after brightness removal (p 0.042);
      - 0.27 within woven fabrics (p 0.048).
    - With roughly 11 feature sets × 5 views × several tests, a few p values near 0.05 are expected by chance.

## 6. Magnification comparison

`magnification_comparison.png` shows cotton silhouettes (exact p) for each view:

| Features | Low (01+04) | Medium (02+05) | High (03+06) | High, front only | High, back only |
|---|---|---|---|---|---|
| ResNet-50 · RGB | 0.160 (0.004) | 0.150 (0.006) | 0.149 (0.006) | 0.150 (0.003) | 0.122 (0.012) |
| ResNet-50 · colour-norm | 0.051 (0.134) | 0.060 (0.116) | 0.131 (0.016) | 0.137 (0.012) | 0.113 (0.018) |
| EffNet-B0 · colour-norm | 0.115 (0.032) | 0.108 (0.034) | 0.108 (0.030) | 0.133 (0.014) | 0.063 (0.084) |
| Gabor · grey | **0.281** (0.010) | 0.208 (0.027) | 0.064 (0.176) | 0.023 (0.307) | 0.094 (0.112) |
| GLCM · grey-norm | **0.277** (0.005) | 0.209 (0.024) | 0.134 (0.081) | 0.119 (0.096) | 0.168 (0.050) |
| Colour-only | 0.184 (0.054) | 0.205 (0.038) | 0.162 (0.070) | 0.149 (0.080) | 0.153 (0.085) |

**Observations:**
- **Handcrafted texture separates best at *low* magnification** (2.92 µm per px). There, the Gabor wavelengths of 4 to 32 px correspond to about 12 to 93 µm, and the GLCM distances of 1 to 16 px to about 3 to 47 µm: mostly yarn and weave scale, not single fibres. **[Hypothesis]** Their separation reflects yarn and weave structure more than fibre morphology.
- **The CNNs vary little by view,** except ResNet-50 colour-norm, which is strongest at high magnification.
- **After colour removal, CNN results flip with the view:** small positive values at low, high and high-back, and negative values at medium and high-front (`model_feature_summary.csv`). Handcrafted results also vary by view (§5, item 10). That instability is itself evidence against a robust signal.

## 7. Controls

| Possible explanation | Evidence | Assessment |
|---|---|---|
| Knit vs woven | Construction silhouettes are as large as cotton's (0.12 to 0.34). CNN cotton separation survives construction-stratified permutation and holds within woven fabrics; the knit comparison can't be tested (3 vs 3). | Doesn't fully explain the CNN result. Within woven fabrics, colour separates more. |
| Weave and fabric family | The 4 woven cotton groups are similar woven print cloths and broadcloths from one supplier, 100 to 107 gsm (plain weave inferred; not documented for B06, B14); the woven non-cotton groups are diverse (pongee, taffeta, challis, spun-nylon plain weave). Texture features place same-base-cloth pairs among the closest pairs. | **Likely major contributor.** "Woven cotton" here is close to one fabric family. |
| Yarn arrangement (filament vs spun) | Filament yarn is documented only for B19 and B20; excluding them lowers CNN separation to 0.08 to 0.12. Cotton is always a spun staple yarn; most synthetics' yarn type isn't documented. | Partial contributor; can't be resolved from the metadata. |
| Surface finish (optical brightener, bleaching, mercerising) | The two finish pairs are far apart in colour space but close in texture space. Optical brightener is present in both classes (B04, B05 cotton; B09 no cotton). | For these two pairs, finish affected colour features much more than texture. Probably not the main CNN driver, but two pairs are too few to be sure. |
| Magnification | Handcrafted features peak at low magnification; colour-removal results change sign by view. | Results depend on view, so they aren't robust. |
| Image exposure | Recorded settings near-identical across the subset (light preset "General", gain NA, balance ratio 2.18 to 2.19). Brightness regression leaves CNN separation intact. Per-sample brightness adjustment is described in the README but not recorded. Glare is noted for Source B undyed fabrics; 25 highlight-clipped tiles were removed. | Brightness alone doesn't explain it; glare and sheen may still contribute through "colour" features. |
| Source-specific acquisition | Single source (B), single supplier, near-identical recorded settings. | Controlled by design within E2; it limits generalisation. |

## 8. Limitations

- **Sample size:** 16 independent groups (7 vs 9); knit 3 vs 3; woven 4 vs 6. Effect sizes are small (silhouette 0.1 to 0.3).
- **Multiple comparisons:** many correlated tests (11 feature sets, 5 views, several controls), with no correction applied. P values are descriptive and specific to this sample. **No claim of statistical reliability is made.**
- **Labels:** supplier specifications. The cotton/no-cotton split lumps 35% to 100% cotton together. Nothing here addresses cotton *percentage*.
- **Colour regression:** 3 PCs fitted on 16 groups is a strong intervention. Regressing out 3 random covariates already moves CNN silhouettes over a wide range (5–95% roughly −0.03 to 0.20). So at high magnification, colour removal can't be told apart from a loss of degrees of freedom. Over- or under-correction is possible.
- **Visible-light imaging only.** Microscopy texture is a proxy for fibre and yarn structure and gives no chemical identification.
- **Frozen ImageNet features** were not designed for microscopy, and no tile-level or fine-tuned analysis was done, by design.

## 9. Why WEAK OR INCONCLUSIVE, not the other verdicts

- **Not PROMISING TEXTURE SIGNAL.**
  - The best-supported separation, from the CNNs, is matched or exceeded in silhouette terms by colour-only features.
  - Colour removal can't establish it as colour-independent. At high magnification the test is inconclusive (values within the random-covariate range). At medium magnification colour removal turns it negative.
  - It is driven by one family of structurally similar woven fabrics.
  - The texture-only results that survive colour removal are view-dependent and borderline among many tests:
    - GLCM on normalised greyscale, p ≈ 0.04 to 0.05;
    - GLCM on raw greyscale, 0.275 (p 0.011) at high magnification but negative at medium and high-front.
- **Not NO EVIDENCE.** Some separation survives colour-normalised input, brightness regression and construction-stratified permutation. GLCM on normalised greyscale stays positive after colour removal at high and low magnification. A small texture-related difference can't be ruled out. It can't be shown to be cotton-specific either.

## 10. Recommendation

The NIST data can't settle this further. With 16 groups, a single supplier, and cotton status tied to fabric family, more analysis of the same images would mostly re-measure construction. If high-resolution texture is pursued, the decisive test needs **matched pairs**: the same construction and yarn type (for example, spun plain-weave cotton vs spun plain-weave polyester at similar gsm), imaged under fixed, recorded exposure, with lab-verified composition. That belongs in BWorth's own collection protocol, not in more NIST analysis.

---

## Reproducing E2

**Prerequisites:** nothing is downloaded automatically.

- **NIST data.** NIST NIR-SORT v2.0.0 colour microscopy and metadata, extracted to `datasets/external/NIST_NIR_SORT/raw_download/NIST_SORT_DATA/` (git-ignored). The scripts locate `SampleInformation.csv` and `ColorParameters.csv` under `*/mds2-3325/`, and the TIFFs anywhere below that folder.
- **Inventory.** `experiments/nist_nir_sort/nist_composition_inventory.csv` (committed). It provides source, cotton percentage, evidence category and strict provenance groups.
- **Pretrained weights,** already in the local torch hub cache (`<torch hub dir>/checkpoints/`):
  - `resnet50-11ad3fa6.pth` (torchvision `ResNet50_Weights.IMAGENET1K_V2`);
  - `efficientnet_b0_rwightman-7f5810bc.pth` (torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1`).

  `extract_e2.py` loads them with `torch.load` from the cache and **never downloads**. If a file is missing it stops with an error. SHA-256 hashes are in `experiment_metadata.json`.
- **Python packages:** torch, torchvision, numpy, pandas, Pillow, opencv-python (`cv2`), scikit-learn, scipy and matplotlib. Versions used are recorded in `experiment_metadata.json`; scikit-image is not needed. `opencv-python` and `scipy` are not yet listed in the project's `requirements.txt`.

**Commands,** run from the repository root:

```
# 1. Tiling, tile QC and feature extraction (GPU; about 9 minutes on an RTX 3050 Laptop GPU, 547 s for 108 images)
CUDA_VISIBLE_DEVICES=0 python experiments/nist_nir_sort/e2_high_resolution_texture/extract_e2.py --batch 16

# 2. Group-level analysis, controls and figures (CPU, about 20 s)
python experiments/nist_nir_sort/e2_high_resolution_texture/analyse_e2.py
```

**Notes:**

- **GPU visibility.** `CUDA_VISIBLE_DEVICES=0` is needed on this machine because the shell sets `CUDA_VISIBLE_DEVICES=-1`, which hides the GPU. Without it, extraction falls back to CPU, which is much slower.
- **Caches.** Image-level feature arrays (`cache/img_*.npy`) are git-ignored, so after a fresh clone **step 1 must run before step 2**.
- **The three small index files in `cache/`,** written by step 1 and read by step 2:
  - `image_index.csv`: the row order of every `img_*.npy` array (108 rows: image ID, swatch, replicate, side, magnification, scale, relative NIST path);
  - `selection.csv`: the 18 selected swatches with the metadata used by the analysis (group, cotton, construction, finish, NIST notes);
  - `feature_names.json`: column names of the Gabor, GLCM and colour feature sets.

  They are regenerated with the arrays and don't need to be committed: their contents are duplicated in `tile_inventory.csv` and `swatch_texture_features.csv`.
- **Randomness.**
  - The analysis uses one NumPy generator seeded with `SEED = 0`, for the Mantel permutations and the random-covariate control (200 draws). Both are consumed in a fixed order, so every number in the CSV and JSON outputs reproduces exactly.
  - Mantel permutations: **999** for the high-magnification view and **199** for the other views.
  - Exact permutation tests involve no randomness.
  - Extraction sets no seed. Frozen-network inference is deterministic apart from possible tiny cuDNN floating-point differences.
  - **Known non-determinism:** the swatch PCA scatter plots of CNN features use scikit-learn's default solver. For 18 × 2048 data that solver is `randomized`, with no fixed `random_state`. `pca_cnn_high_mag.png` therefore differs slightly between runs (point coordinates by about 1e-4; a few hundred anti-aliased pixels). The handcrafted PCA plot and all CSV and JSON values are unaffected.

---

### Output files (this directory)

| File | Contents |
|---|---|
| `E2_REPORT.md` | This report |
| `tile_inventory.csv` | 9,720 tiles: tile_id, parent image_id, swatch, view, x0/y0/x1/y1, QC metrics, included flag, exclusion reason, group, cotton |
| `swatch_texture_features.csv` | 18 rows (one per physical swatch): metadata, controls (construction, finish, optical brightener, documented yarn, ColorParameters), high-magnification tile counts, and all 11 feature sets at high magnification (column prefix `<set>__`) |
| `model_feature_summary.csv` | 55 rows (11 feature sets × 5 views): every statistic and control above |
| `group_level_results.csv` | Per-group cotton silhouettes, high magnification |
| `pca_cnn_high_mag.png`, `pca_handcrafted_high_mag.png` | Swatch-level PCA, high magnification |
| `magnification_comparison.png` | Cotton, colour-removed and knit/woven silhouettes by view |
| `group_level_silhouettes.png` | Per-group silhouettes heatmap |
| `analysis_results.json` | Tile QC summary, per-swatch controls, finish-pair distances, group list, random seed |
| `experiment_metadata.json` | Selection, tiling, QC thresholds, representations, weights and hashes, device, extraction and analysis software versions, every analysis control, seed, permutation counts |
| `extract_e2.py`, `analyse_e2.py` | Reproduction scripts (read NIST data read-only; write only here) |
| `cache/img_*.npy` | Image-level feature arrays, 11 files, about 4.3 MB; git-ignored by the existing `experiments/**/*.npy` rule; regenerate with `extract_e2.py` |
| `cache/image_index.csv`, `cache/selection.csv`, `cache/feature_names.json` | Small index files for the arrays (see Reproducing E2); not needed in Git |
