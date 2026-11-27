"""NIST E2, step 2: group-level texture analysis of undyed white Source B (exploratory; no classifier).

Unit of analysis: strict provenance group (16). Swatch features = mean of image features for the chosen views
(image feature = mean of included tiles); features are z-scored across the 18 swatches, then averaged per group.
Statistics (descriptive, small-n; no correction for multiple comparisons):
  * cotton silhouette (cosine) with an EXACT permutation p over all C(16,7)=11,440 label assignments;
  * construction-stratified exact p (cotton labels permuted only within knit / woven strata: 4,200 assignments);
  * leave-one-group-out jackknife range and per-group silhouette values;
  * colour control: cotton silhouette after regressing out the top-3 colour-baseline PCs (exact p);
  * random-covariate control: the same regression on 3 random Gaussian covariates, 200 draws (mean and 5-95% range),
    to show how much regression alone shrinks a silhouette with only 16 groups;
  * exposure control: cotton silhouette after regressing out group mean brightness only (exact p);
  * within-construction: cotton silhouette inside woven (4 vs 6 groups, 210 assignments) and knit (3 vs 3, 20 assignments;
    smallest attainable p = 0.05);
  * cotton silhouette excluding the two documented filament groups (B19, B20; exact p over 3,432);
  * knit/woven silhouette (exact p over 8,008), Mantel correlation with colour distances (999 permutations for the high
    view, 199 for the others), PC1-vs-brightness correlation;
  * finish-pair control (high view): distance percentile of B05-B07 (optical brightener) and B12-B15 (mercerised) among
    all 153 swatch pairs.
Random numbers (Mantel permutations, random covariates) come from one generator seeded with SEED = 0, consumed in a
fixed order, so reruns reproduce every value exactly.

    python experiments/nist_nir_sort/e2_high_resolution_texture/analyse_e2.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CACHE = HERE / "cache"
NIST = PROJECT / "datasets" / "external" / "NIST_NIR_SORT" / "raw_download" / "NIST_SORT_DATA"
COLORPARAMS = next(NIST.glob("*/mds2-3325/Microscopy/Optical/Stereo/Color/ColorParameters.csv"))
SEED = 0
RNG = np.random.default_rng(SEED)
MANTEL_PERMS = {"high": 999, "other_views": 199}
RANDOM_COVARIATE_DRAWS = 200
SETS = ["resnet50_rgb", "resnet50_grey", "resnet50_cnorm", "efficientnet_b0_rgb", "efficientnet_b0_grey",
        "efficientnet_b0_cnorm", "gabor_grey", "gabor_greynorm", "glcm_grey", "glcm_greynorm", "colour_rgb"]
LABEL = {"resnet50_rgb": "ResNet-50 · RGB", "resnet50_grey": "ResNet-50 · grey", "resnet50_cnorm": "ResNet-50 · colour-norm",
         "efficientnet_b0_rgb": "EffNet-B0 · RGB", "efficientnet_b0_grey": "EffNet-B0 · grey",
         "efficientnet_b0_cnorm": "EffNet-B0 · colour-norm", "gabor_grey": "Gabor · grey", "gabor_greynorm": "Gabor · grey-norm",
         "glcm_grey": "GLCM · grey", "glcm_greynorm": "GLCM · grey-norm", "colour_rgb": "Colour-only baseline"}
VIEWS = {"high": ["03", "06"], "medium": ["02", "05"], "low": ["01", "04"], "high_front": ["03"], "high_back": ["06"]}
FILAMENT_DOCUMENTED = {"B19", "B20"}     # NIST notes: "Filament fiber"
COT, INK, MUTED, BG = {"cotton": "#2a78d6", "no cotton": "#eb6834"}, "#252421", "#52514e", "#fcfcfb"


# ------------------------------------------------------------------ data
def load():
    sel = pd.read_csv(CACHE / "selection.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    idx = pd.read_csv(CACHE / "image_index.csv", dtype={"replicate": str})
    feats = {k: np.load(CACHE / f"img_{k}.npy").astype(np.float64) for k in SETS}
    names = json.loads((CACHE / "feature_names.json").read_text(encoding="utf-8"))
    tiles = pd.read_csv(HERE / "tile_inventory.csv", dtype={"replicate": str})
    cp = pd.read_csv(COLORPARAMS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    cp.columns = [c.strip() for c in cp.columns]
    cp["sid"] = cp["Sample ID"].str.strip().str[:3]
    cp = cp.set_index("sid")
    sel["short"] = sel.index.str[:3]
    sel["yarn_documented"] = ["filament" if "filament" in n.lower() else ("spun" if "spun" in n.lower() else "not stated")
                              for n in sel.nist_notes]
    sel["optical_brightener"] = ["with" if ("with optical" in f.lower() and "without" not in f.lower()) else
                                 ("without" if "without" in f.lower() else "not stated") for f in sel.finish]
    for c_src, c_dst in (("Light Source", "cp_light_source"), ("Gain", "cp_gain"), ("Balance Ratio", "cp_balance_ratio")):
        sel[c_dst] = [cp.loc[s, c_src].strip() if s in cp.index else "" for s in sel.short]
    return sel, idx, feats, names, tiles


def swatch_matrix(X_img, idx, swatches, reps):
    m = idx.replicate.isin(reps).to_numpy()
    df = pd.DataFrame(X_img[m]).groupby(idx.swatch_id.to_numpy()[m]).mean()
    return df.loc[swatches].to_numpy()


def to_groups(Xs, groups):
    Z = StandardScaler().fit_transform(Xs)
    keys = list(dict.fromkeys(groups))
    return np.vstack([Z[np.array(groups) == g].mean(0) for g in keys]), keys


def cosdist(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return np.clip(1 - Xn @ Xn.T, 0, 2)


# ------------------------------------------------------------------ silhouettes (two classes, from a distance matrix)
def sil_batch(D, L):
    """L: P x n binary label matrix (1 = class 1). Returns P mean silhouettes and the per-point values of row 0."""
    n1 = L.sum(1, keepdims=True); n0 = L.shape[1] - n1
    S1 = L @ D; S0 = (1 - L) @ D                       # sum of distances from each point to class1 / class0
    a = np.where(L == 1, S1 / np.maximum(n1 - 1, 1), S0 / np.maximum(n0 - 1, 1))
    b = np.where(L == 1, S0 / n0, S1 / n1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return s.mean(1), s[0]


def all_assignments(n, k):
    combos = list(itertools.combinations(range(n), k))
    L = np.zeros((len(combos), n), np.int8)
    for r, c in enumerate(combos):
        L[r, list(c)] = 1
    return L


def stratified_assignments(lab, strata):
    blocks = []
    for s in np.unique(strata):
        ii = np.where(strata == s)[0]
        blocks.append([(ii, c) for c in itertools.combinations(ii, int(lab[ii].sum()))])
    rows = []
    for combo in itertools.product(*blocks):
        r = np.zeros(len(lab), np.int8)
        for ii, c in combo:
            r[list(c)] = 1
        rows.append(r)
    return np.array(rows)


def exact_test(D, lab, perms):
    obs, per_point = sil_batch(D, lab[None, :].astype(np.int8))
    null, _ = sil_batch(D, perms)
    return float(obs[0]), float((null >= obs[0] - 1e-12).mean()), per_point, len(perms)


def mantel(D1, D2, n_perm=1999):
    iu = np.triu_indices_from(D1, 1)
    r = spearmanr(D1[iu], D2[iu]).statistic
    null = [spearmanr(D1[iu], D2[p][:, p][iu]).statistic for p in (RNG.permutation(len(D1)) for _ in range(n_perm))]
    return float(r), float((1 + np.sum(np.array(null) >= r)) / (n_perm + 1))


# ------------------------------------------------------------------ analysis
def analyse(sel, idx, feats, names, tiles):
    swatches = list(sel.index)
    groups = sel.group_strict.tolist()
    keys = list(dict.fromkeys(groups))
    gmeta = pd.DataFrame({g: sel[sel.group_strict == g].iloc[0] for g in keys}).T
    gmeta["members"] = [",".join(sel[sel.group_strict == g].short) for g in keys]
    lab = (gmeta.cotton == "cotton").to_numpy().astype(np.int8)
    knit = (gmeta.construction == "knit").to_numpy().astype(np.int8)
    perms_cot = all_assignments(len(keys), int(lab.sum()))
    perms_strat = stratified_assignments(lab, knit)
    perms_knit = all_assignments(len(keys), int(knit.sum()))
    keep_nf = np.array([not set(m.split(",")) & FILAMENT_DOCUMENTED for m in gmeta.members])
    perms_nf = all_assignments(int(keep_nf.sum()), int(lab[keep_nf].sum()))
    colour_names = names["colour_rgb"]
    rows, group_rows, swatch_feats, pca_store = [], [], {}, {}
    for view, reps in VIEWS.items():
        Gcol, _ = to_groups(swatch_matrix(feats["colour_rgb"], idx, swatches, reps), groups)
        Zc = PCA(n_components=3).fit_transform(Gcol)
        bright = swatch_matrix(feats["colour_rgb"], idx, swatches, reps)[:, colour_names.index("brightness_mean")]
        gbright = np.array([bright[np.array(groups) == g].mean() for g in keys])
        Dcol = cosdist(Gcol)
        for k in SETS:
            Xs = swatch_matrix(feats[k], idx, swatches, reps)
            G, _ = to_groups(Xs, groups)
            D = cosdist(G)
            s, p, per_point, nperm = exact_test(D, lab, perms_cot)
            s_st, p_st, _, n_st = exact_test(D, lab, perms_strat)
            jack = []
            for drop in range(len(keys)):
                kk = np.arange(len(keys)) != drop
                jack.append(float(sil_batch(D[np.ix_(kk, kk)], lab[kk][None, :])[0][0]))
            sk, pk, _, _ = exact_test(D, knit, perms_knit)
            Dnf = D[np.ix_(keep_nf, keep_nf)]
            snf, pnf, _, _ = exact_test(Dnf, lab[keep_nf], perms_nf)
            row = {"feature_set": k, "label": LABEL[k], "view": view, "replicates": "+".join(reps),
                   "n_swatches": len(swatches), "n_groups": len(keys), "n_cotton_groups": int(lab.sum()),
                   "n_no_cotton_groups": int(len(keys) - lab.sum()), "feature_dim": int(Xs.shape[1]),
                   "cotton_silhouette": round(s, 4), "cotton_exact_p": round(p, 4), "n_label_assignments": nperm,
                   "cotton_p_stratified_by_construction": round(p_st, 4), "n_stratified_assignments": n_st,
                   "jackknife_min": round(min(jack), 4), "jackknife_max": round(max(jack), 4),
                   "knit_woven_silhouette": round(sk, 4), "knit_woven_exact_p": round(pk, 4),
                   "cotton_silhouette_excl_filament_B19_B20": round(snf, 4), "cotton_exact_p_excl_filament": round(pnf, 4)}
            for cname, cval in (("woven", 0), ("knit", 1)):      # within-construction: woven 4 vs 6 groups, knit 3 vs 3
                kk = knit == cval
                sw_, pw_, _, nw_ = exact_test(D[np.ix_(kk, kk)], lab[kk], all_assignments(int(kk.sum()), int(lab[kk].sum())))
                row[f"cotton_silhouette_{cname}_only"] = round(sw_, 4)
                row[f"cotton_exact_p_{cname}_only"] = round(pw_, 4)
                row[f"n_assignments_{cname}_only"] = nw_
            if k != "colour_rgb":
                R = G - LinearRegression().fit(Zc, G).predict(Zc)
                sr, pr, _, _ = exact_test(cosdist(R), lab, perms_cot)
                r_m, p_m = mantel(D, Dcol, MANTEL_PERMS["high"] if view == "high" else MANTEL_PERMS["other_views"])
                # control 1: how much does regressing out 3 RANDOM covariates shrink the silhouette with only 16 groups?
                rand = []
                for _ in range(RANDOM_COVARIATE_DRAWS):
                    Zr = RNG.normal(size=(len(keys), 3))
                    rand.append(sil_batch(cosdist(G - LinearRegression().fit(Zr, G).predict(Zr)), lab[None, :])[0][0])
                # control 2: exposure only - regress out group mean brightness (1 covariate)
                Zb = gbright[:, None]
                sb, pb, _, _ = exact_test(cosdist(G - LinearRegression().fit(Zb, G).predict(Zb)), lab, perms_cot)
                row.update(cotton_silhouette_after_colour_removal=round(sr, 4), cotton_exact_p_after_colour_removal=round(pr, 4),
                           cotton_silhouette_after_random3_removal_mean=round(float(np.mean(rand)), 4),
                           cotton_silhouette_after_random3_removal_p5_p95=f"{np.percentile(rand, 5):.3f}..{np.percentile(rand, 95):.3f}",
                           cotton_silhouette_after_brightness_removal=round(sb, 4), cotton_exact_p_after_brightness_removal=round(pb, 4),
                           mantel_r_vs_colour=round(r_m, 4), mantel_p_vs_colour=round(p_m, 4))
            pc1 = PCA(n_components=2).fit(G)
            P = pc1.transform(G)
            row["group_pc1_vs_brightness_spearman"] = round(float(spearmanr(P[:, 0], gbright).statistic), 4)
            row["group_pc1_vs_cotton_spearman"] = round(float(spearmanr(P[:, 0], lab).statistic), 4)
            rows.append(row)
            if view == "high":
                for g, v in zip(keys, per_point):
                    group_rows.append({"feature_set": k, "label": LABEL[k], "group": g, "members": gmeta.loc[g, "members"],
                                       "cotton": gmeta.loc[g, "cotton"], "construction": gmeta.loc[g, "construction"],
                                       "group_silhouette_cotton": round(float(v), 4)})
                swatch_feats[k] = Xs
                Zs = StandardScaler().fit_transform(Xs)
                pz = PCA(n_components=2).fit(Zs)
                pca_store[k] = (pz.transform(Zs), pz.explained_variance_ratio_)
    return pd.DataFrame(rows), pd.DataFrame(group_rows), swatch_feats, pca_store, gmeta, keys


# ------------------------------------------------------------------ plots
def style(ax):
    ax.set_facecolor(BG); ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=7); ax.grid(color="#e6e5df", lw=0.6); ax.set_axisbelow(True)


def pca_grid(sets, pca_store, sel, summary, out, title):
    n = len(sets); cols = 3; rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(17, 5.3 * rws), facecolor=BG)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), sets):
        style(ax); P, ev = pca_store[k]
        for g in sel.group_strict.unique():
            ii = np.where((sel.group_strict == g).to_numpy())[0]
            if len(ii) > 1:
                ax.plot(P[ii, 0], P[ii, 1], color="#898781", ls="--", lw=1, zorder=1)
        for i, (sid, r) in enumerate(sel.iterrows()):
            ax.scatter(*P[i], s=70, c=COT[r.cotton], marker="o" if r.construction == "woven" else "^", edgecolors=INK, lw=0.8, zorder=3)
            ax.annotate(f"{r.short} ({float(r.cotton_percentage):g}%)", P[i], xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK)
        srow = summary[(summary.feature_set == k) & (summary.view == "high")].iloc[0]
        ax.set_title(f"{LABEL[k]}\ncotton silhouette {srow.cotton_silhouette:+.3f} (exact p {srow.cotton_exact_p:.3f}); "
                     f"knit/woven {srow.knit_woven_silhouette:+.3f}", loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=7.5, color=MUTED); ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=7.5, color=MUTED)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=v, markersize=8, label=k) for k, v in COT.items()]
    h += [Line2D([], [], marker="o", ls="", markerfacecolor="#c3c2b7", markeredgecolor=INK, markersize=8, label="woven"),
          Line2D([], [], marker="^", ls="", markerfacecolor="#c3c2b7", markeredgecolor=INK, markersize=8, label="knit"),
          Line2D([], [], color="#898781", ls="--", label="same strict provenance group")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(out, dpi=130, facecolor=BG); plt.close(fig)


def magnification_plot(summary, out):
    views = list(VIEWS)
    metrics = [("cotton_silhouette", "cotton_exact_p", "Cotton silhouette (exact p)"),
               ("cotton_silhouette_after_colour_removal", "cotton_exact_p_after_colour_removal", "After removing 3 colour PCs"),
               ("knit_woven_silhouette", "knit_woven_exact_p", "Knit vs woven silhouette")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5), facecolor=BG)
    for ax, (m, pcol, ttl) in zip(axes, metrics):
        M = np.full((len(SETS), len(views)), np.nan); Pm = M.copy()
        for i, k in enumerate(SETS):
            for j, v in enumerate(views):
                r = summary[(summary.feature_set == k) & (summary.view == v)].iloc[0]
                if pd.notna(r.get(m, np.nan)):
                    M[i, j] = r[m]; Pm[i, j] = r[pcol]
        lim = 0.4
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(SETS)):
            for j in range(len(views)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}\np={Pm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(M[i, j]) > 0.25 else INK)
        ax.set_xticks(range(len(views))); ax.set_xticklabels(views, fontsize=8, rotation=20)
        ax.set_yticks(range(len(SETS))); ax.set_yticklabels([LABEL[k] for k in SETS], fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Undyed white Source B (18 swatches, 16 strict groups: 7 cotton vs 9 no cotton). Silhouettes by view; "
                 "exact permutation p over all label assignments. Descriptive only.", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=130, facecolor=BG); plt.close(fig)


def group_plot(groups_df, gmeta, keys, out):
    M = groups_df.pivot(index="group", columns="feature_set", values="group_silhouette_cotton").loc[keys, SETS]
    fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=BG)
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]; ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.3 else INK)
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels([LABEL[k] for k in SETS], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{gmeta.loc[g, 'members']} · {gmeta.loc[g, 'cotton']} · {gmeta.loc[g, 'construction']}" for g in keys], fontsize=8)
    ax.set_title("Per-group cotton silhouette, high magnification (positive = closer to its own cotton class). "
                 "Shows which groups drive any separation.", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=BG); plt.close(fig)


def main() -> int:
    sel, idx, feats, names, tiles = load()
    summary, groups_df, swatch_feats, pca_store, gmeta, keys = analyse(sel, idx, feats, names, tiles)
    summary.to_csv(HERE / "model_feature_summary.csv", index=False)
    groups_df.to_csv(HERE / "group_level_results.csv", index=False)
    # swatch-level features (high magnification), one row per physical swatch
    inc = tiles[tiles.replicate.isin(VIEWS["high"])].groupby("swatch_id").agg(
        high_mag_tiles_total=("included", "size"), high_mag_tiles_included=("included", "sum"))
    meta_cols = ["short", "cotton_percentage", "other_fibre_composition", "cotton", "evidence_category", "group_strict",
                 "construction", "finish", "optical_brightener", "yarn_documented", "nist_notes", "cp_light_source", "cp_gain",
                 "cp_balance_ratio"]
    parts = [sel[meta_cols].join(inc)]
    for k in SETS:
        cols = names[k] if k in names else [f"f{j:04d}" for j in range(swatch_feats[k].shape[1])]
        parts.append(pd.DataFrame(swatch_feats[k], index=sel.index, columns=[f"{k}__{c}" for c in cols]))
    out = pd.concat(parts, axis=1)
    out.index.name = "swatch_id"
    out.to_csv(HERE / "swatch_texture_features.csv", float_format="%.6g")
    pca_grid(SETS[:6], pca_store, sel, summary, HERE / "pca_cnn_high_mag.png",
             "CNN tile features, high magnification (views 03+06), one point per swatch (n=18). PCA per panel. EXPLORATORY.")
    pca_grid(SETS[6:], pca_store, sel, summary, HERE / "pca_handcrafted_high_mag.png",
             "Handcrafted texture and colour features, high magnification (views 03+06), one point per swatch (n=18). EXPLORATORY.")
    magnification_plot(summary, HERE / "magnification_comparison.png")
    group_plot(groups_df, gmeta, keys, HERE / "group_level_silhouettes.png")
    # tile + control summaries
    tq = {"tiles_total": int(len(tiles)), "tiles_included": int(tiles.included.sum()),
          "excluded_by_reason": tiles.loc[~tiles.included, "exclusion_reason"].str.split(";").explode().value_counts().to_dict(),
          "excluded_by_swatch_view": tiles[~tiles.included].groupby(["swatch_id", "replicate"]).size().rename("n").reset_index()
                                       .assign(swatch=lambda d: d.swatch_id.str[:3]).drop(columns="swatch_id").to_dict("records"),
          "included_per_magnification": tiles[tiles.included].groupby("magnification").size().to_dict()}
    # surface-finish control: same base cloth, different finish. Percentile of the pair's distance among all swatch pairs
    # (0 = the closest pair in the subset). B05/B07: with vs without optical brightener; B12/B15: bleached-desized vs mercerised.
    finish_pairs = {}
    shorts = list(sel.short)
    iu = np.triu_indices(len(shorts), 1)
    for k in SETS:
        Ds = cosdist(StandardScaler().fit_transform(swatch_feats[k]))
        allp = Ds[iu]
        finish_pairs[k] = {f"{a}-{b}": round(float((allp < Ds[shorts.index(a), shorts.index(b)]).mean() * 100), 1)
                           for a, b in (("B05", "B07"), ("B12", "B15"))}
    controls = sel[["short", "cotton", "cotton_percentage", "construction", "finish", "optical_brightener", "yarn_documented",
                    "cp_light_source", "cp_gain", "cp_balance_ratio", "group_strict"]].to_dict("records")
    (HERE / "analysis_results.json").write_text(json.dumps({"tile_qc": tq, "controls_by_swatch": controls,
                                                             "finish_pair_distance_percentile_high_mag": finish_pairs,
                                                             "groups": gmeta[["members", "cotton", "construction"]].reset_index().rename(columns={"index": "group"}).to_dict("records"),
                                                             "permutation": "exact enumeration; p = share of assignments with silhouette >= observed",
                                                             "random_seed": SEED,
                                                             "random_seed_used_for": ["Mantel permutations", "random-covariate control draws"]},
                                                            indent=1, default=str), encoding="utf-8")
    import matplotlib as _mpl, scipy as _scipy, sklearn as _sk
    md = json.loads((HERE / "experiment_metadata.json").read_text(encoding="utf-8"))
    md["analysis"] = {
        "script": "analyse_e2.py", "unit": "strict provenance group (16: 7 cotton, 9 no cotton)", "views": VIEWS,
        "main_view": "high (replicates 03 + 06)",
        "aggregation": "tiles -> image mean -> swatch mean over the selected views -> z-score across 18 swatches -> mean per strict group",
        "silhouette": "cosine distance on group-level features",
        "exact_permutations": {"cotton": len(all_assignments(16, 7)), "stratified_by_construction": "4200 (woven 4/10, knit 3/6)",
                               "knit_woven": len(all_assignments(16, 6)), "excl_filament_B19_B20": len(all_assignments(14, 7)),
                               "woven_only": len(all_assignments(10, 4)), "knit_only": len(all_assignments(6, 3)),
                               "p_definition": "share of all assignments with silhouette >= observed (observed included)"},
        "controls": {
            "colour_removal": "linear regression of group features on the top-3 PCs of the colour baseline (same views); exact p",
            "random_covariate_control": f"same regression on 3 random N(0,1) covariates, {RANDOM_COVARIATE_DRAWS} draws; mean and 5-95% range "
                                        "reported, to separate colour effects from loss of degrees of freedom",
            "brightness_regression": "linear regression of group features on group mean brightness (1 covariate); exact p",
            "within_construction": "cotton silhouette within woven groups (4 vs 6) and within knit groups (3 vs 3; min attainable p 0.05)",
            "excl_filament": "groups containing documented filament-yarn swatches B19, B20 removed (14 groups)",
            "jackknife": "leave-one-group-out silhouette range",
            "finish_pairs": "high view: percentile of the B05-B07 (optical brightener) and B12-B15 (mercerised) distances among all 153 "
                            "swatch pairs, per feature set",
            "mantel_vs_colour": f"Spearman correlation of group distance matrices; {MANTEL_PERMS['high']} permutations (high view), "
                                f"{MANTEL_PERMS['other_views']} (other views)",
            "exposure_correlation": "Spearman correlation of group PC1 with group mean brightness"},
        "random_seed": SEED, "random_seed_used_for": ["Mantel permutations", "random-covariate control draws"],
        "multiple_comparisons": "not corrected; p values are descriptive",
        "software": {"scikit-learn": _sk.__version__, "scipy": _scipy.__version__, "matplotlib": _mpl.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "outputs": ["model_feature_summary.csv", "group_level_results.csv", "swatch_texture_features.csv", "analysis_results.json",
                    "pca_cnn_high_mag.png", "pca_handcrafted_high_mag.png", "magnification_comparison.png",
                    "group_level_silhouettes.png"]}
    (HERE / "experiment_metadata.json").write_text(json.dumps(md, indent=2), encoding="utf-8")
    print(summary[summary.view == "high"][["label", "cotton_silhouette", "cotton_exact_p", "cotton_p_stratified_by_construction",
                                           "jackknife_min", "jackknife_max", "cotton_silhouette_after_colour_removal",
                                           "cotton_exact_p_after_colour_removal", "knit_woven_silhouette",
                                           "cotton_silhouette_excl_filament_B19_B20", "cotton_exact_p_excl_filament"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
