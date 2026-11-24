"""Static report figures (training curves, confusion matrices) saved as PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e0"
SERIES = {"train": "#2a78d6", "val": "#eb6834"}  # categorical slots 1 and 2, fixed order
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_training_curves(history: list[dict], best_epoch: int, out_path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, metric, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        _style(ax)
        for split, name in (("train", "Training"), ("val", "Validation")):
            ys = [h[f"{split}_{metric}"] for h in history]
            ax.plot(epochs, ys, color=SERIES[split], linewidth=2, label=name)
            ax.annotate(name, (epochs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK_2, fontsize=9, va="center")
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1)
        ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", color=INK_2, fontsize=8, va="top")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "acc":
            ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: str | Path,
                          title: str, normalize: bool = False) -> None:
    data = cm.astype(float)
    if normalize:
        row = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row, out=np.zeros_like(data), where=row > 0)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 + 0.62 * n), max(4.2, 0.6 + 0.55 * n)), facecolor=SURFACE)
    im = ax.imshow(data, cmap=SEQ_CMAP, vmin=0, vmax=1 if normalize else max(1, data.max()))
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(n), labels, fontsize=8, color=INK_2)
    ax.set_xlabel("Predicted", color=INK_2, fontsize=9)
    ax.set_ylabel("True", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    # thin surface-coloured gaps between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    thresh = (1 if normalize else data.max()) * 0.55
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if v == 0:
                continue
            txt = f"{v:.2f}" if normalize else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > thresh else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
