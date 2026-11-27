"""NIST E1, step 2: exploratory PCA, confound and magnification analysis of the frozen embeddings.

No classifier is trained and no accuracy is computed. Statistics are descriptive:
  * silhouette (cosine, standardised features) of a metadata label, computed on STRICT-PROVENANCE-GROUP means,
    with a permutation null that shuffles labels across groups;
  * Mantel-style Spearman correlation between two feature spaces' group-level distance matrices;
  * the cotton-status silhouette after linearly regressing out the top colour principal components.

    python experiments/nist_nir_sort/e1_visual_feature_analysis/analyse_e1.py
"""

from __future__ import annotations

import json
import re
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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SAMPLEINFO = next((PROJECT / "datasets/external/NIST_NIR_SORT/raw_download/NIST_SORT_DATA").glob("*/mds2-3325/SampleInformation.csv"))
RNG = np.random.default_rng(0)
N_PERM = 999
FEATURE_SETS = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "colour": "Colour-only baseline"}
VIEW_ORDER = ["01", "02", "03", "04", "05", "06"]
VIEW_LABEL = {"01": "front low", "02": "front med", "03": "front high", "04": "back low", "05": "back med", "06": "back high"}

# palette (validated categorical slots; cotton status uses two well-separated hues)
COT = {"cotton": "#2a78d6", "no cotton": "#eb6834"}
DYE_COL = {"cellulosic dyes": "#2a78d6", "disperse": "#e87ba4", "acid / metal-complex": "#eda100", "undyed": "#1baf7a",
           "other": "#898781", "unknown (C/D)": "#c3c2b7"}
MARK = {"A": "o", "B": "s", "C": "^", "D": "D"}
VIEW_COL = dict(zip(VIEW_ORDER, ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4", "#8a5cd6"]))
INK, MUTED, BG = "#252421", "#52514e", "#fcfcfb"


def dye_class(t: str) -> str:
    t = t.strip().lower()
    if t == "none": return "undyed"
    if t == "unknown": return "unknown (C/D)"
    if "disperse" in t: return "disperse"
    if "acid" in t or "chromium" in t: return "acid / metal-complex"
    if any(k in t for k in ("reactive", "direct", "vat", "sulphur", "azoic", "priocion")): return "cellulosic dyes"
    return "other"


def load() -> tuple[dict, pd.DataFrame, dict]:
    si = pd.read_csv(SAMPLEINFO, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    si["Sample ID"] = si["Sample ID"].str.strip()
    si = si.set_index("Sample ID")
    feats, meta = {}, None
    for k in ("resnet50", "efficientnet_b0"):
        df = pd.read_csv(HERE / f"{k}_swatch_embeddings.csv", dtype={"swatch_id": str}, keep_default_na=False)
        cols = [c for c in df.columns if re.fullmatch(r"f\d{4}", c)]
        feats[k] = df.set_index("swatch_id")[cols].astype(np.float64)
        if meta is None:
            meta = df.set_index("swatch_id").drop(columns=cols)
    c = pd.read_csv(HERE / "colour_features.csv", keep_default_na=False)
    csw = c[c.level == "swatch"].set_index("swatch_id")
    ccols = [x for x in c.columns if x not in ("level", "image_id", "swatch_id", "replicate", "side", "magnification",
                                                 "scale_px_per_0p1mm", "source", "cotton_percentage", "other_fibre_composition",
                                                 "evidence_category", "bworth_band", "group_strict", "n_images")]
    feats["colour"] = csw.loc[meta.index, ccols].astype(np.float64)
    meta = meta.copy()
    meta["cotton_percentage"] = meta.cotton_percentage.astype(float)
    meta["cotton"] = np.where(meta.cotton_percentage > 0, "cotton", "no cotton")
    meta["dye_class"] = [dye_class(si.loc[s, "Dye Type"]) for s in meta.index]
    meta["dye_type_raw"] = [si.loc[s, "Dye Type"].strip() for s in meta.index]
    meta["visual_colour"] = [si.loc[s, "Visual Inspected Color"].strip() for s in meta.index]
    meta["construction"] = [si.loc[s, "Form Factor"].strip().lower() for s in meta.index]
    meta["short"] = [s[:3] for s in meta.index]
    imgs = {}
    for k in ("resnet50", "efficientnet_b0"):
        df = pd.read_csv(HERE / f"{k}_image_embeddings.csv", dtype={"replicate": str}, keep_default_na=False)
        cols = [x for x in df.columns if re.fullmatch(r"f\d{4}", x)]
        imgs[k] = (df[["image_id", "swatch_id", "replicate"]], df[cols].to_numpy(np.float64))
    ci = c[c.level == "image"].copy()
    ci["replicate"] = ci.replicate.astype(str).str.zfill(2)
    imgs["colour"] = (ci[["image_id", "swatch_id", "replicate"]].reset_index(drop=True), ci[ccols].to_numpy(np.float64))
    return feats, meta, imgs


def std(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def group_means(X: np.ndarray, groups: pd.Series) -> tuple[np.ndarray, list]:
    keys = list(dict.fromkeys(groups))
    return np.vstack([X[(groups == g).to_numpy()].mean(0) for g in keys]), keys


def group_label(meta: pd.DataFrame, keys: list, col: str) -> np.ndarray:
    out = []
    for g in keys:
        v = meta.loc[meta.group_strict == g, col].unique()
        out.append(v[0] if len(v) == 1 else "__mixed__")
    return np.array(out)


def silhouette_perm(X: np.ndarray, lab: np.ndarray) -> dict:
    """Drops groups with mixed or unknown labels and classes represented by <2 groups; records what was dropped."""
    vals, counts = np.unique(lab, return_counts=True)
    small = {v for v, c in zip(vals, counts) if c < 2}
    drop = {"__mixed__", "unknown (C/D)"} | small
    dropped = {v: int(c) for v, c in zip(vals.tolist(), counts.tolist()) if v in drop}
    keep = ~np.isin(lab, list(drop))
    X, lab = X[keep], lab[keep]
    vals, counts = np.unique(lab, return_counts=True)
    if len(vals) < 2:
        return {"silhouette": None, "p_perm": None, "n_groups": int(len(lab)), "class_groups": dict(zip(vals.tolist(), counts.tolist())),
                "dropped_groups_by_label": dropped, "note": "fewer than 2 classes with >=2 groups"}
    s = silhouette_score(X, lab, metric="cosine")
    null = np.array([silhouette_score(X, RNG.permutation(lab), metric="cosine") for _ in range(N_PERM)])
    return {"silhouette": round(float(s), 4), "p_perm": round(float((1 + (null >= s).sum()) / (N_PERM + 1)), 4),
            "null_mean": round(float(null.mean()), 4), "n_groups": int(len(lab)),
            "class_groups": dict(zip(vals.tolist(), counts.tolist())), "dropped_groups_by_label": dropped}


def cosdist(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return 1 - Xn @ Xn.T


def mantel(D1, D2) -> dict:
    iu = np.triu_indices_from(D1, 1)
    r = spearmanr(D1[iu], D2[iu]).statistic
    n = D1.shape[0]; null = []
    for _ in range(N_PERM):
        p = RNG.permutation(n)
        null.append(spearmanr(D1[iu], D2[p][:, p][iu]).statistic)
    null = np.array(null)
    return {"spearman_r": round(float(r), 4), "p_perm": round(float((1 + (null >= r).sum()) / (N_PERM + 1)), 4), "n": int(n)}


def residualise(Xcnn: np.ndarray, Xcol: np.ndarray, k: int) -> np.ndarray:
    Z = PCA(n_components=min(k, Xcol.shape[0] - 1, Xcol.shape[1])).fit_transform(Xcol)
    return Xcnn - LinearRegression().fit(Z, Xcnn).predict(Z)


# ---------------------------------------------------------------- subsets
def subsets(meta: pd.DataFrame) -> dict:
    sB = meta.source == "B"
    white_undyed = sB & (meta.dye_class == "undyed") & (meta.visual_colour.str.lower() == "white")
    return {
        "A_all_94": meta.index[np.ones(len(meta), bool)],
        "B_supplier_documented": meta.index[meta.evidence_category == "supplier_documented"],
        "C_source_B": meta.index[sB],
        "D_source_B_undyed_white": meta.index[white_undyed],
        "E_source_A_only (dye confound check)": meta.index[meta.source == "A"],
    }


def analyse_subset(name, idx, feats, meta) -> dict:
    m = meta.loc[idx]
    res = {"n_swatches": int(len(m)), "n_strict_groups": int(m.group_strict.nunique()),
           "swatches_by_cotton": m.cotton.value_counts().to_dict(),
           "groups_by_cotton": m.groupby("cotton").group_strict.nunique().to_dict(),
           "swatch_ids": sorted(m.short.tolist()), "features": {}}
    G = {}
    for fs in FEATURE_SETS:
        X = std(feats[fs].loc[idx].to_numpy())
        Xg, keys = group_means(X, m.group_strict)
        G[fs] = (Xg, keys)
        r = {}
        for lab in ("cotton", "source", "dye_class", "construction"):
            r[f"silhouette_{lab}"] = silhouette_perm(Xg, group_label(m, keys, lab))
        p = PCA(n_components=min(5, len(m) - 1)).fit(X)
        r["pca_explained_variance_ratio_top5"] = [round(float(v), 4) for v in p.explained_variance_ratio_]
        res["features"][fs] = r
    keys = G["colour"][1]
    Dcol = cosdist(G["colour"][0])
    for fs in ("resnet50", "efficientnet_b0"):
        res["features"][fs]["mantel_vs_colour"] = mantel(cosdist(G[fs][0]), Dcol)
        k = 5
        if len(keys) > k + 3:
            Xr = residualise(G[fs][0], G["colour"][0], k)
            res["features"][fs][f"silhouette_cotton_after_removing_top{k}_colour_PCs"] = silhouette_perm(Xr, group_label(m, keys, "cotton"))
    res["mantel_resnet_vs_efficientnet"] = mantel(cosdist(G["resnet50"][0]), cosdist(G["efficientnet_b0"][0]))
    return res


# ---------------------------------------------------------------- plots
def style(ax):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#e6e5df", linewidth=0.6); ax.set_axisbelow(True)


def scatter_swatches(ax, P, m, colour_by: str, label=True):
    for i, (sid, r) in enumerate(m.iterrows()):
        col = COT[r.cotton] if colour_by == "cotton" else DYE_COL[r.dye_class]
        hollow = r.evidence_category == "uncorroborated_care_label"
        ec = INK if r.source == "B" else ("white" if not hollow else col)
        lw = 1.6 if r.source == "B" else 0.6
        if r.evidence_category == "lab_confirmed":
            ax.scatter(*P[i], marker="*", s=230, c=col, edgecolors=INK, linewidths=1.0, zorder=4)
        else:
            ax.scatter(*P[i], marker=MARK[r.source], s=52, facecolors="none" if hollow else col,
                       edgecolors=ec if not hollow else col, linewidths=1.4 if hollow else lw, zorder=3)
        if label:
            ax.annotate(r.short, P[i], xytext=(3, 3), textcoords="offset points", fontsize=6, color=MUTED)


def legend_handles(colour_by):
    from matplotlib.lines import Line2D
    h = []
    pal = COT if colour_by == "cotton" else DYE_COL
    for k, v in pal.items():
        h.append(Line2D([], [], marker="o", ls="", color=v, markersize=7, label=k))
    for s, mk in MARK.items():
        h.append(Line2D([], [], marker=mk, ls="", markerfacecolor="#c3c2b7", markeredgecolor=INK if s == "B" else "white",
                        markeredgewidth=1.6 if s == "B" else 0.6, markersize=7, label=f"Source {s}" + (" (dark outline)" if s == "B" else "")))
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED, markersize=7, label="hollow = care-label (uncorroborated)"))
    h.append(Line2D([], [], marker="*", ls="", color=MUTED, markersize=11, label="C02 lab-confirmed"))
    return h


def pca_plot(fs, feats, meta, out):
    X = std(feats[fs].to_numpy())
    p = PCA(n_components=2).fit(X); P = p.transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=BG)
    for ax, cb, t in ((axes[0], "cotton", "Coloured by cotton content (documented label)"),
                      (axes[1], "dye", "Same projection, coloured by dye class (confound)")):
        style(ax); scatter_swatches(ax, P, meta, "cotton" if cb == "cotton" else "dye")
        ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%} of variance)", color=MUTED, fontsize=9)
        ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%})", color=MUTED, fontsize=9)
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.legend(handles=legend_handles("cotton" if cb == "cotton" else "dye"), fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"{FEATURE_SETS[fs]}: PCA of swatch-level features (one point per physical swatch, n=94). "
                 "EXPLORATORY: clustering is not evidence of fibre identification.", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor=BG); plt.close(fig)


def source_b_plot(feats, meta, subs, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), facecolor=BG)
    for row, key, ttl in ((0, "C_source_B", "All Source B"), (1, "D_source_B_undyed_white", "Source B, undyed white only")):
        idx = subs[key]; m = meta.loc[idx]
        for col, fs in enumerate(FEATURE_SETS):
            ax = axes[row, col]; style(ax)
            X = std(feats[fs].loc[idx].to_numpy()); p = PCA(n_components=2).fit(X); P = p.transform(X)
            for g in m.group_strict.unique():  # connect members of the same strict provenance group
                ii = np.where((m.group_strict == g).to_numpy())[0]
                if len(ii) > 1:
                    ax.plot(P[ii, 0], P[ii, 1], color="#898781", lw=1, ls="--", zorder=1)
            for i, (sid, r) in enumerate(m.iterrows()):
                ax.scatter(*P[i], s=70, c=COT[r.cotton], edgecolors=INK, linewidths=0.8, zorder=3)
                ax.annotate(f"{r.short} ({r.cotton_percentage:g}%)", P[i], xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
            ax.set_title(f"{FEATURE_SETS[fs]}: {ttl} (n={len(m)} swatches, {m.group_strict.nunique()} groups)", loc="left", fontsize=9.5, color=INK)
            ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%})", color=MUTED, fontsize=8)
            ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%})", color=MUTED, fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=v, markersize=8, label=k) for k, v in COT.items()]
    h.append(Line2D([], [], color="#898781", ls="--", label="same strict provenance group"))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Source B comparison: PCA fitted within each subset; labels show supplier-stated cotton %. "
                 "Very small n: descriptive only.", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(out, dpi=140, facecolor=BG); plt.close(fig)


# ---------------------------------------------------------------- magnification
def magnification(imgs, meta) -> tuple[dict, dict]:
    out, plotdata = {}, {}
    for fs, (ids, X) in imgs.items():
        Z = std(X)
        sw, rep = ids.swatch_id.to_numpy(), ids.replicate.to_numpy()
        mu = Z.mean(0)
        sst = ((Z - mu) ** 2).sum()
        ss_sw = sum(((Z[sw == s].mean(0) - mu) ** 2).sum() * (sw == s).sum() for s in np.unique(sw))
        ss_v = sum(((Z[rep == v].mean(0) - mu) ** 2).sum() * (rep == v).sum() for v in VIEW_ORDER)
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        S = Zn @ Zn.T
        same_sw = sw[:, None] == sw[None, :]
        same_v = rep[:, None] == rep[None, :]
        off = ~np.eye(len(sw), dtype=bool)
        cross = np.zeros((6, 6))
        for a, va in enumerate(VIEW_ORDER):
            for b, vb in enumerate(VIEW_ORDER):
                msk = same_sw & (rep[:, None] == va) & (rep[None, :] == vb) & off
                cross[a, b] = S[msk].mean() if msk.any() else np.nan
        # per-view swatch-level distance structure agreement (Mantel between views)
        swatches = list(meta.index)
        per_view = {}
        for v in VIEW_ORDER:
            sel = rep == v
            order = pd.Series(np.arange(sel.sum()), index=sw[sel]).loc[swatches].to_numpy()
            per_view[v] = Z[sel][order]
        mv = np.zeros((6, 6))
        for a, va in enumerate(VIEW_ORDER):
            for b, vb in enumerate(VIEW_ORDER):
                iu = np.triu_indices(len(swatches), 1)
                mv[a, b] = spearmanr(cosdist(per_view[va])[iu], cosdist(per_view[vb])[iu]).statistic
        img_lab_sw = silhouette_score(Z, sw, metric="cosine")
        img_lab_v = silhouette_score(Z, rep, metric="cosine")
        out[fs] = {
            "variance_fraction_between_swatches": round(float(ss_sw / sst), 4),
            "variance_fraction_between_views": round(float(ss_v / sst), 4),
            "mean_cos_sim_same_swatch_diff_view": round(float(S[same_sw & ~same_v].mean()), 4),
            "mean_cos_sim_diff_swatch_same_view": round(float(S[~same_sw & same_v].mean()), 4),
            "mean_cos_sim_diff_swatch_diff_view": round(float(S[~same_sw & ~same_v].mean()), 4),
            "mean_cos_sim_same_swatch_front_vs_back_same_mag": round(float(np.nanmean([cross[0, 3], cross[1, 4], cross[2, 5]])), 4),
            "mean_cos_sim_same_swatch_same_side_diff_mag": round(float(np.nanmean([cross[0, 1], cross[0, 2], cross[1, 2], cross[3, 4], cross[3, 5], cross[4, 5]])), 4),
            "within_swatch_cross_view_similarity": {VIEW_LABEL[va]: {VIEW_LABEL[vb]: (None if np.isnan(cross[a, b]) else round(float(cross[a, b]), 4)) for b, vb in enumerate(VIEW_ORDER)} for a, va in enumerate(VIEW_ORDER)},
            "between_view_mantel_spearman": {VIEW_LABEL[va]: {VIEW_LABEL[vb]: round(float(mv[a, b]), 4) for b, vb in enumerate(VIEW_ORDER)} for a, va in enumerate(VIEW_ORDER)},
            "image_level_silhouette_by_swatch": round(float(img_lab_sw), 4),
            "image_level_silhouette_by_view": round(float(img_lab_v), 4),
        }
        p = PCA(n_components=2).fit(Z)
        plotdata[fs] = (p.transform(Z), rep, p.explained_variance_ratio_, cross, mv, out[fs])
    return out, plotdata


def magnification_plot(plotdata, out):
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), facecolor=BG)
    for r, fs in enumerate(FEATURE_SETS):
        P, rep, ev, cross, mv, st = plotdata[fs]
        ax = axes[r, 0]; style(ax)
        for v in VIEW_ORDER:
            s = rep == v
            ax.scatter(P[s, 0], P[s, 1], s=10, c=VIEW_COL[v], label=VIEW_LABEL[v], alpha=0.8, edgecolors="none")
        ax.set_title(f"{FEATURE_SETS[fs]}: all 564 images by view", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=8, color=MUTED); ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=8, color=MUTED)
        ax.legend(fontsize=7, frameon=False, markerscale=1.6)
        for c, (M, ttl) in enumerate(((cross, "Mean cosine similarity, same swatch, view pairs"),
                                      (mv, "Agreement of inter-swatch distances between views (Spearman)")), start=1):
            ax = axes[r, c]
            lo = float(np.nanmin(M[~np.eye(6, dtype=bool)])) if c == 1 else 0.0
            hi = float(np.nanmax(M[~np.eye(6, dtype=bool)])) if c == 1 else 1.0
            im = ax.imshow(M, cmap="Blues", vmin=lo, vmax=hi)
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            ax.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=7)
            for i in range(6):
                for j in range(6):
                    if not np.isnan(M[i, j]) and not (c == 1 and i == j):
                        frac = (M[i, j] - lo) / max(hi - lo, 1e-9)
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if frac > 0.6 else INK)
            ax.set_title(f"{FEATURE_SETS[fs]}: {ttl}", loc="left", fontsize=8.5, color=INK)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[r, 0].text(0.01, -0.16, f"variance between swatches {st['variance_fraction_between_swatches']:.0%}; "
                        f"between views {st['variance_fraction_between_views']:.0%}", transform=axes[r, 0].transAxes, fontsize=8, color=MUTED)
    fig.suptitle("Magnification and side: image-level analysis (here each point is an image, used only to compare views; "
                 "the main analysis aggregates the 6 views per swatch).", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out, dpi=130, facecolor=BG); plt.close(fig)


def main() -> int:
    feats, meta, imgs = load()
    subs = subsets(meta)
    results = {"subsets": {}, "label_definitions": {
        "cotton": "cotton_percentage > 0 (documented label; supplier spec for A/B, care label for C/D)",
        "dye_class": "from SampleInformation 'Dye Type': cellulosic dyes (reactive/direct/vat/sulphur/azoic), disperse, "
                     "acid/metal-complex, undyed (none), other (acrylic trichromatic), unknown (C/D)",
        "unit_of_statistics": "strict provenance group mean (groups with mixed labels excluded for that label)",
        "permutations": N_PERM}}
    for k, idx in subs.items():
        results["subsets"][k] = analyse_subset(k, idx, feats, meta)
        print(k, results["subsets"][k]["n_swatches"], results["subsets"][k]["n_strict_groups"])
    results["dye_vs_fibre_crosstab_source_A"] = pd.crosstab(meta.loc[subs["E_source_A_only (dye confound check)"], "dye_class"],
                                                            meta.loc[subs["E_source_A_only (dye confound check)"], "cotton"]).to_dict()
    results["magnification"], pdata = magnification(imgs, meta)
    for fs in FEATURE_SETS:
        pca_plot(fs, feats, meta, HERE / ("pca_colour_baseline.png" if fs == "colour" else f"pca_{fs}.png"))
    source_b_plot(feats, meta, subs, HERE / "source_b_comparison.png")
    magnification_plot(pdata, HERE / "magnification_comparison.png")
    (HERE / "analysis_results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    md = json.loads((HERE / "experiment_metadata.json").read_text(encoding="utf-8"))
    md["analysis"] = {"script": "analyse_e1.py", "results_file": "analysis_results.json", "pca": "StandardScaler then PCA, fit per subset",
                      "silhouette": "cosine, on standardised features averaged per strict provenance group", "permutations": N_PERM,
                      "random_seed": 0, "colour_residualisation": "linear regression of group-level CNN features on top-5 colour PCs",
                      "scipy_used_for": "Spearman correlation"}
    (HERE / "experiment_metadata.json").write_text(json.dumps(md, indent=2), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
