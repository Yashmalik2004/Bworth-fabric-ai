"""Build the Phase-2 split manifests from the v1 manifest + manual swatch corrections.

    python -m training.composition.experiment_splits

Never modifies the v1 manifest. Writes:
    processed/composition/splits_group_v2.csv   fixed train/val/test split with corrected groups
    processed/composition/splits_group_v2.json  summary (moved images, counts)
    processed/composition/cv/folds.csv          5-fold assignment (stratified by cotton %, grouped by swatch)
    processed/composition/cv/fold{k}.csv        fold k = test, fold k+1 = val, the other 3 = train

v2 split rule for a merged group: train if any part was in v1 train, otherwise the
split of its largest part (ties -> test). Hence no v2 test image was ever a v1
training image, and the v1 baseline checkpoint can be evaluated on v2 test fairly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FINE_TO_BAND, PROCESSED_DIR  # noqa: E402
from training.composition.dataset import SPLITS, check_no_leakage  # noqa: E402

V1 = PROCESSED_DIR / "splits_group.csv"
V2 = PROCESSED_DIR / "splits_group_v2.csv"
CV_DIR = PROCESSED_DIR / "cv"
OVERRIDES = PROJECT_ROOT / "configs" / "swatch_group_overrides.json"
N_FOLDS = 5
SEED = 42


def merged_group_ids(groups: list[str], merges: list[list[str]]) -> dict[str, str]:
    parent = {g: g for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in merges:
        unknown = [g for g in m if g not in parent]
        if unknown:
            raise KeyError(f"Override refers to unknown groups: {unknown}")
        for g in m[1:]:
            parent[find(g)] = find(m[0])
    members: dict[str, list[str]] = {}
    for g in groups:
        members.setdefault(find(g), []).append(g)
    return {g: "+".join(sorted(members[find(g)])) for g in groups}


def build_v2(v1: pd.DataFrame, merges: list[list[str]]) -> tuple[pd.DataFrame, dict]:
    df = v1.rename(columns={"group": "group_v1", "split": "split_v1"}).copy()
    df["group"] = df["group_v1"].map(merged_group_ids(sorted(df["group_v1"].unique()), merges))
    assign = {}
    for g, gdf in df.groupby("group"):
        if (gdf.split_v1 == "train").any():
            assign[g] = "train"
        else:
            sizes = gdf.groupby("split_v1").size()
            best = sizes.max()
            assign[g] = "test" if sizes.get("test", 0) == best else sizes.idxmax()
    df["split"] = df["group"].map(assign)
    check_no_leakage(df)
    assert not ((df.split == "test") & (df.split_v1 == "train")).any()
    moved = df[df.split != df.split_v1]
    summary = {
        "n_images": len(df),
        "groups_v1": int(df.group_v1.nunique()),
        "groups_v2": int(df.group.nunique()),
        "moved_images": {f"{a}->{b}": int(n) for (a, b), n in moved.groupby(["split_v1", "split"]).size().items()},
        "counts": {s: df[df.split == s].groupby("label").size().to_dict() for s in SPLITS},
        "counts_band4": {s: df[df.split == s]["label"].map(FINE_TO_BAND).value_counts().to_dict() for s in SPLITS},
        "counts_total": {s: int((df.split == s).sum()) for s in SPLITS},
    }
    return df, summary


def build_folds(df: pd.DataFrame) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for k, (_, test_idx) in enumerate(sgkf.split(df, y=df["label"], groups=df["group"])):
        fold.iloc[test_idx] = k
    out = df[["path", "label", "label_idx", "group", "dup_cluster"]].copy()
    out["fold"] = fold.values
    assert (out.fold >= 0).all()
    assert out.groupby("group")["fold"].nunique().max() == 1
    return out


def fold_manifest(folds: pd.DataFrame, k: int) -> pd.DataFrame:
    m = folds.copy()
    val_fold = (k + 1) % N_FOLDS
    m["split"] = "train"
    m.loc[m.fold == k, "split"] = "test"
    m.loc[m.fold == val_fold, "split"] = "val"
    check_no_leakage(m)
    return m


def main() -> int:
    v1 = pd.read_csv(V1)
    merges = [m["groups"] for m in json.loads(OVERRIDES.read_text())["merges"]]
    v2, summary = build_v2(v1, merges)
    v2.to_csv(V2, index=False)
    V2.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=int))
    print(f"v2 split -> {V2}: {summary['counts_total']} | groups {summary['groups_v1']} -> {summary['groups_v2']} "
          f"| moved {summary['moved_images']}")

    CV_DIR.mkdir(parents=True, exist_ok=True)
    folds = build_folds(v2)
    folds.to_csv(CV_DIR / "folds.csv", index=False)
    for k in range(N_FOLDS):
        fold_manifest(folds, k).to_csv(CV_DIR / f"fold{k}.csv", index=False)
    tab = pd.crosstab(folds.label, folds.fold)
    print("Images per class per fold:\n" + tab.to_string())
    print("Groups per fold:", folds.groupby("fold").group.nunique().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
