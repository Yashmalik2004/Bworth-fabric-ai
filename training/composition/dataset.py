"""Dataset indexing, cleaning, leakage-safe splitting and the PyTorch Dataset.

Pipeline (see README "Data strategy" for the reasoning):

1. Index every image in ``Original/<class>/`` (``Augmented/`` is never used:
   its files cannot be mapped back to their source originals, and some are
   byte-identical copies of originals).
2. Clean: images whose content is identical or near-identical (perceptual
   hash) are collapsed into one *duplicate cluster*. Clusters whose copies
   carry more than one cotton label are label conflicts and are dropped.
   Every other cluster keeps exactly one representative image.
3. Group: each class consists of repeated photos of a small number of physical
   fabric swatches. With ``split_strategy="group"`` images are clustered into
   approximate *swatch groups* (ImageNet ResNet-50 embeddings, agglomerative
   clustering) and whole groups are assigned to train/val/test, so the test set
   measures generalisation to swatches the model has not seen. With
   ``split_strategy="random"`` every unique image is its own group.
4. Split: groups are assigned per class (stratified) to train / val / test.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    BAND_DEFINITIONS,
    FINE_TO_BAND,
    IMAGE_EXTENSIONS,
    CompositionConfig,
    class_sort_key,
)
from utils.image_utils import load_rgb_image  # noqa: E402

SPLITS = ("train", "val", "test")


def to_relative(path: str | Path) -> str:
    """Store paths relative to the project root so split CSVs are portable."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# 1. Indexing
# --------------------------------------------------------------------------- #
def discover_classes(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    classes = sorted((p.name for p in root.iterdir() if p.is_dir()), key=class_sort_key)
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 class folders in {root}, found {classes}")
    return classes


def index_images(data_dir: str | Path) -> pd.DataFrame:
    classes = discover_classes(data_dir)
    rows = []
    for idx, cls in enumerate(classes):
        for p in sorted((Path(data_dir) / cls).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": to_relative(p), "label": cls, "label_idx": idx})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Cleaning (duplicates and label conflicts)
# --------------------------------------------------------------------------- #
def dhash_bits(path: str | Path, size: int = 16) -> np.ndarray:
    with Image.open(resolve_path(path)) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def _union_find_components(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(adjacency, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


@dataclass
class CleaningResult:
    clean: pd.DataFrame
    removed_duplicates: pd.DataFrame
    removed_conflicts: pd.DataFrame


def clean_index(df: pd.DataFrame, hamming_threshold: int = 8) -> CleaningResult:
    """Collapse (near-)duplicate images and drop clusters with conflicting labels."""
    hashes = np.stack([dhash_bits(p) for p in df["path"]])
    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    comp = _union_find_components(dist <= hamming_threshold)
    df = df.copy()
    df["dup_cluster"] = comp

    keep, dups, conflicts = [], [], []
    for _, g in df.groupby("dup_cluster", sort=False):
        if g["label"].nunique() > 1:
            conflicts.append(g)
            continue
        g = g.sort_values("path")
        keep.append(g.iloc[[0]].assign(n_copies=len(g)))
        if len(g) > 1:
            dups.append(g.iloc[1:])

    def _cat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()

    return CleaningResult(_cat(keep).sort_values("path", ignore_index=True), _cat(dups), _cat(conflicts))


# --------------------------------------------------------------------------- #
# 3. Swatch grouping
# --------------------------------------------------------------------------- #
def extract_embeddings(paths: list[str], batch_size: int = 32) -> np.ndarray:
    """L2-normalised ImageNet ResNet-50 embeddings (no fine-tuning)."""
    import torch
    from torchvision import models

    from utils.device import get_device

    device = get_device()
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    tf = weights.transforms()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            x = torch.stack([tf(load_rgb_image(resolve_path(p))) for p in paths[i:i + batch_size]]).to(device)
            feats.append(torch.nn.functional.normalize(net(x).float(), dim=1).cpu().numpy())
    return np.concatenate(feats)


def cluster_class(emb: np.ndarray, threshold: float, min_groups: int) -> np.ndarray:
    """Agglomerative (average-linkage, cosine) clustering of one class.

    The distance threshold is lowered step by step if it produces fewer than
    ``min_groups`` clusters, so every class can contribute to all three splits.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(emb)
    if n <= min_groups:
        return np.arange(n)
    t = threshold
    while True:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=t
        ).fit_predict(emb)
        if labels.max() + 1 >= min_groups or t <= 0.02:
            return labels
        t -= 0.02


def assign_groups(df: pd.DataFrame, cfg: CompositionConfig) -> pd.DataFrame:
    df = df.copy()
    if cfg.split_strategy == "random":
        df["group"] = [f"img{i}" for i in range(len(df))]
        return df
    if cfg.split_strategy != "group":
        raise ValueError(f"Unknown split_strategy: {cfg.split_strategy!r}")
    emb = extract_embeddings(df["path"].tolist())
    groups = np.empty(len(df), dtype=object)
    for cls, idx in df.groupby("label").indices.items():
        lab = cluster_class(emb[idx], cfg.group_distance_threshold, cfg.group_min_groups_per_class)
        groups[idx] = [f"{cls}_g{int(k):02d}" for k in lab]
    df["group"] = groups
    return df


# --------------------------------------------------------------------------- #
# 4. Splitting
# --------------------------------------------------------------------------- #
def split_groups(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Stratified group split: per class, whole groups go to one split only."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0,1) and sum to < 1")
    rng = np.random.default_rng(seed)
    fractions = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    assignment: dict[str, str] = {}
    for _, cdf in df.groupby("label"):
        sizes = cdf.groupby("group").size()
        names = list(sizes.index)
        rng.shuffle(names)
        names.sort(key=lambda g: -sizes[g])  # stable sort keeps random tie order
        n = len(cdf)
        target = {s: fractions[s] * n for s in SPLITS}
        current = {s: 0 for s in SPLITS}
        members: dict[str, list[str]] = {s: [] for s in SPLITS}
        for g in names:
            s = max(SPLITS, key=lambda k: target[k] - current[k])
            members[s].append(g)
            current[s] += sizes[g]
        # every split gets at least one group when possible
        for s in ("test", "val"):
            if not members[s] and len(members["train"]) > 1:
                g = min(members["train"], key=lambda k: sizes[k])
                members["train"].remove(g)
                members[s].append(g)
        for s, gs in members.items():
            for g in gs:
                assignment[g] = s
    out = df.copy()
    out["split"] = out["group"].map(assignment)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if any group or duplicate cluster spans more than one split."""
    for col in ("group", "dup_cluster"):
        if col in df:
            spans = df.groupby(col)["split"].nunique()
            bad = spans[spans > 1]
            if len(bad):
                raise AssertionError(f"Leakage: {len(bad)} '{col}' values appear in multiple splits")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_splits(cfg: CompositionConfig, rebuild: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Build (or load) the split CSV for ``cfg``; returns a DataFrame with a ``split`` column."""
    path = Path(cfg.splits_path)
    if path.is_file() and not rebuild:
        df = pd.read_csv(path)
        if verbose:
            print(f"Loaded existing splits from {path}")
        return df

    raw = index_images(cfg.data_dir)
    cleaned = clean_index(raw)
    df = assign_groups(cleaned.clean, cfg)
    df = split_groups(df, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    check_no_leakage(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    summary = {
        "split_strategy": cfg.split_strategy,
        "seed": cfg.seed,
        "raw_images": len(raw),
        "removed_duplicate_copies": len(cleaned.removed_duplicates),
        "removed_label_conflict_images": len(cleaned.removed_conflicts),
        "label_conflict_clusters": int(cleaned.removed_conflicts["dup_cluster"].nunique()) if len(cleaned.removed_conflicts) else 0,
        "clean_images": len(df),
        "groups_per_class": df.groupby("label")["group"].nunique().to_dict(),
        "counts": {s: df[df.split == s].groupby("label").size().to_dict() for s in SPLITS},
        "groups_per_split": {s: int(df[df.split == s]["group"].nunique()) for s in SPLITS},
    }
    summary["counts_total"] = {s: int(sum(v.values())) for s, v in summary["counts"].items()}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=int))
    cleaned.removed_conflicts.to_csv(path.parent / "removed_label_conflicts.csv", index=False)
    if verbose:
        print(f"Built splits ({cfg.split_strategy}) -> {path}")
        print(json.dumps({k: summary[k] for k in ("raw_images", "removed_duplicate_copies",
                                                  "removed_label_conflict_images", "clean_images",
                                                  "counts_total", "groups_per_split")}, default=int))
    return df


# --------------------------------------------------------------------------- #
# Label schemes
# --------------------------------------------------------------------------- #
def scheme_classes(scheme: str, fine_classes: list[str]) -> list[str]:
    """Ordered class names for a label scheme."""
    if scheme == "fine13":
        return list(fine_classes)
    if scheme == "band4":
        missing = set(fine_classes) - set(FINE_TO_BAND)
        if missing:
            raise ValueError(f"Fine classes without a band: {sorted(missing)}")
        return list(BAND_DEFINITIONS)
    raise ValueError(f"Unknown label_scheme {scheme!r}")


def apply_label_scheme(df: pd.DataFrame, scheme: str, classes: list[str]) -> pd.DataFrame:
    """Add a ``target`` column (class index under ``scheme``); fine labels are kept untouched."""
    idx = {c: i for i, c in enumerate(classes)}
    out = df.copy()
    names = out["label"] if scheme == "fine13" else out["label"].map(FINE_TO_BAND)
    out["target"] = names.map(idx).astype(int)
    return out


class FabricDataset(Dataset):
    """Images + integer labels for one split (``target`` column if present, else ``label_idx``)."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.paths = [resolve_path(p) for p in frame["path"]]
        col = "target" if "target" in frame else "label_idx"
        self.labels = frame[col].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_rgb_image(self.paths[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]
