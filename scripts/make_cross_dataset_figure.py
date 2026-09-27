#!/usr/bin/env python3
"""Create publication-ready cross-dataset model comparison figure."""
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from run_columbia_blocks_baseline import REPO

COLORS = {"TruFor": "#0072B2", "MVSS-Net": "#D55E00"}


def style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD",
        "grid.linestyle": "--", "grid.linewidth": .6, "axes.axisbelow": True})


def main():
    col_tru = json.loads((REPO / "results/trufor_columbia_detection/summary.json").read_text())
    col_mvss = json.loads((REPO / "results/mvss_columbia_evaluation/summary.json").read_text())
    casia = json.loads((REPO / "results/casia_v1_evaluation/summary.json").read_text())["models"]
    values = {"TruFor": [col_tru["metrics"]["roc_auc"], casia["trufor"]["detection"]["roc_auc"]],
              "MVSS-Net": [col_mvss["detection"]["roc_auc"], casia["mvss"]["detection"]["roc_auc"]]}
    loc = {"TruFor": [casia["trufor"]["localization_by_type"][x]["f1"] for x in ("CM", "Sp")],
           "MVSS-Net": [casia["mvss"]["localization_by_type"][x]["f1"] for x in ("CM", "Sp")]}
    style(); fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1)); width = .32; x = np.arange(2)
    for i, (model, vals) in enumerate(values.items()):
        axes[0].bar(x + (i-.5)*width, vals, width, color=COLORS[model], label=model, edgecolor="black", linewidth=.4)
    axes[0].set_xticks(x, ["Columbia", "CASIA v1"]); axes[0].set_ylim(0, 1); axes[0].set_ylabel("Image-level ROC-AUC")
    axes[0].legend(frameon=False, loc="lower left")
    for i, (model, vals) in enumerate(loc.items()):
        axes[1].bar(x + (i-.5)*width, vals, width, color=COLORS[model], label=model, edgecolor="black", linewidth=.4)
    axes[1].set_xticks(x, ["Copy-move", "Splicing"]); axes[1].set_ylim(0, 1); axes[1].set_ylabel("CASIA v1 localization F1")
    for ax, label in zip(axes, ["A", "B"]): ax.text(-.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout(w_pad=2.1); out = REPO / "reports/figures"; out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"): fig.savefig(out / f"cross_dataset_comparison.{suffix}", bbox_inches="tight", facecolor="white")
    print(out / "cross_dataset_comparison.pdf")


if __name__ == "__main__": main()
