"""E1 baselines (no neural network, no downloads): majority, colour histogram, metadata-only.

They are floors for S2 and confound detectors for R6: if band can be predicted from colour or capture metadata,
the collection is biased. Every baseline predicts per image and is scored per fabric (mean of probabilities).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

from training.phase5.protocol import CLASSES

COLOUR_FAMILIES = {
    "red": ["red", "maroon", "burgundy", "wine", "crimson", "coral", "rust", "terracotta"],
    "pink": ["pink", "rose", "magenta", "fuchsia", "salmon"],
    "orange": ["orange", "peach", "apricot"],
    "yellow": ["yellow", "mustard", "gold", "ochre"],
    "green": ["green", "olive", "sage", "mint", "teal", "emerald", "khaki"],
    "blue": ["blue", "navy", "denim", "indigo", "cyan", "turquoise", "azure"],
    "purple": ["purple", "violet", "lilac", "lavender", "plum", "mauve"],
    "brown": ["brown", "tan", "beige", "camel", "chocolate", "taupe"],
    "neutral_light": ["white", "cream", "ivory", "off-white", "ecru", "natural"],
    "neutral_dark": ["black", "charcoal", "grey", "gray", "silver"],
    "multicolour": ["multi", "stripe", "check", "plaid", "print", "pattern"],
}


def colour_family(text: str) -> str:
    t = str(text).casefold()
    for fam, words in COLOUR_FAMILIES.items():
        if any(w in t for w in words):
            return fam
    return "unknown"


def colour_histogram(path: Path, bins=(8, 4, 4)) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("HSV").resize((64, 64)), dtype=np.float32) / 255
    h, _ = np.histogramdd(a.reshape(-1, 3), bins=bins, range=((0, 1), (0, 1), (0, 1)))
    h = h.flatten()
    return h / max(h.sum(), 1)


def metadata_frame(df: pd.DataFrame, fabrics: pd.DataFrame) -> pd.DataFrame:
    meta = fabrics.drop_duplicates("fabric_id").set_index("fabric_id")
    out = pd.DataFrame({
        "colour_family": df.fabric_id.map(meta.colour).map(colour_family),
        "weave_or_knit": df.fabric_id.map(meta.weave_or_knit).fillna(""),
        "fabric_type": df.fabric_id.map(meta.fabric_type).fillna("").str.casefold(),
        "capture_device": df.get("capture_device", pd.Series("", index=df.index)).fillna(""),
        "lighting": df.get("lighting", pd.Series("", index=df.index)).fillna(""),
        "capture_session": df.get("capture_session", pd.Series("", index=df.index)).fillna(""),
    }, index=df.index)
    return out.astype(str)


def _fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    labels = sorted(set(ytr))
    if len(labels) < 2:
        p = np.zeros((Xte.shape[0], len(CLASSES)))
        p[:, labels[0]] = 1
        return p
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = np.zeros((Xte.shape[0], len(CLASSES)))
    p[:, clf.classes_] = clf.predict_proba(Xte)
    return p


def predict_baseline(kind: str, train: pd.DataFrame, test: pd.DataFrame, fabrics: pd.DataFrame,
                     image_root: Path) -> np.ndarray:
    """Image-level class probabilities for ``test`` from a baseline fitted on ``train``."""
    ytr = train.band.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    if kind == "majority":
        p = np.zeros((len(test), len(CLASSES)))
        p[:, np.bincount(ytr, minlength=len(CLASSES)).argmax()] = 1
        return p
    if kind == "colour_histogram":
        Xtr = np.stack([colour_histogram(Path(image_root) / x) for x in train.image_path])
        Xte = np.stack([colour_histogram(Path(image_root) / x) for x in test.image_path])
        return _fit_predict(Xtr, ytr, Xte)
    if kind == "metadata":
        enc = OneHotEncoder(handle_unknown="ignore")
        Xtr = enc.fit_transform(metadata_frame(train, fabrics))
        Xte = enc.transform(metadata_frame(test, fabrics))
        return _fit_predict(Xtr, ytr, Xte)
    raise ValueError(f"unknown baseline {kind!r}")


def grouped_cv_metadata_balanced_accuracy(manifest: pd.DataFrame, fabrics: pd.DataFrame, groups: pd.Series,
                                          folds: int = 5, seed: int = 0) -> float | None:
    """Fabric-level balanced accuracy of the metadata-only baseline under grouped CV (used by readiness R6)."""
    from training.phase5.evaluate import aggregate_fabrics, fabric_metrics

    man = manifest.reset_index(drop=True).copy()
    man["provenance_id"] = man.fabric_id.map(groups)
    fab = man.drop_duplicates("fabric_id")
    if fab.groupby("band").provenance_id.nunique().min() < folds:
        return None
    preds = []
    for tr_f, te_f in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(fab, fab.band, fab.provenance_id):
        trf, tef = set(fab.iloc[tr_f].fabric_id), set(fab.iloc[te_f].fabric_id)
        tr, te = man[man.fabric_id.isin(trf)], man[man.fabric_id.isin(tef)]
        p = predict_baseline("metadata", tr, te, fabrics, Path("."))
        d = te[["fabric_id", "band"]].copy()
        for j, c in enumerate(CLASSES):
            d[f"p_{c}"] = p[:, j]
        preds.append(d)
    return fabric_metrics(aggregate_fabrics(pd.concat(preds)))["balanced_accuracy"]
