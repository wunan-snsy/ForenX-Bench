#!/usr/bin/env python3
"""Create the three manuscript figures from archived ForenX-Bench results.

Figure 1 is a schematic (not experimental evidence). Figures 2 and 3 are
quantitative and are read directly from the retained JSON summaries.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
QA_OUT = ROOT / "tmp" / "figure_qa"
QA_OUT.mkdir(parents=True, exist_ok=True)

TRU = "#2878A5"
MVSS = "#C66B43"
INK = "#263746"
MUTED = "#617383"
GRID = "#D8E0E6"
STAGE = ["#EAF3FC", "#ECF6EE", "#FCEEF0", "#F7F4EA", "#EAF3FC", "#F1EEFA"]
STAGE_EDGE = ["#C6DDF4", "#CDE5D1", "#F2D3D7", "#E9DFC2", "#C6DDF4", "#DAD3EF"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.2,
    "axes.labelsize": 7.2,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.facecolor": "white",
})


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def require_matplotlib_panel_alignment(
    fig: plt.Figure,
    *,
    json_out: Path,
    overlay_svg: Path,
    tolerance_pt: float = 1.5,
    gutter_tolerance_pt: float = 1.5,
    strict: bool = True,
    allow_unequal_width: bool = False,
    width_exemption: str | None = None,
) -> dict:
    """Portable alignment gate for this script's two side-by-side panels."""
    fig.canvas.draw()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if len(axes) != 2:
        raise RuntimeError(f"Expected two visible quantitative panels, found {len(axes)}")
    fig_w_pt, fig_h_pt = fig.get_size_inches() * 72
    panels = []
    for label, ax in zip(("A", "B"), axes):
        box = ax.get_position()
        panels.append({
            "id": label,
            "bbox_pt": [box.x0*fig_w_pt, box.y0*fig_h_pt, box.x1*fig_w_pt, box.y1*fig_h_pt],
        })
    a, b = panels[0]["bbox_pt"], panels[1]["bbox_pt"]
    differences = {
        "bottom_pt": abs(a[1]-b[1]),
        "top_pt": abs(a[3]-b[3]),
        "height_pt": abs((a[3]-a[1])-(b[3]-b[1])),
    }
    if not allow_unequal_width:
        differences["width_pt"] = abs((a[2]-a[0])-(b[2]-b[0]))
    failures = {name: value for name, value in differences.items() if value > tolerance_pt}
    report = {
        "backend": "matplotlib-python",
        "figure": {"width_pt": fig_w_pt, "height_pt": fig_h_pt},
        "panels": panels,
        "row_groups": [{"id": "main-row", "panels": ["A", "B"]}],
        "tolerance_pt": tolerance_pt,
        "gutter_tolerance_pt": gutter_tolerance_pt,
        "differences_pt": differences,
        "width_exemption": width_exemption if allow_unequal_width else None,
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if overlay_svg:
        width_px, height_px = fig_w_pt*96/72, fig_h_pt*96/72
        rects = []
        for panel in panels:
            x0, y0, x1, y1 = panel["bbox_pt"]
            rects.append(
                f'<rect x="{x0*96/72:.2f}" y="{height_px-y1*96/72:.2f}" '
                f'width="{(x1-x0)*96/72:.2f}" height="{(y1-y0)*96/72:.2f}" '
                f'fill="none" stroke="#d33" stroke-width="2"/><text x="{x0*96/72+4:.2f}" '
                f'y="{height_px-y1*96/72+14:.2f}" font-size="12">{panel["id"]}</text>'
            )
        overlay_svg.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px:.2f}" height="{height_px:.2f}" '
            f'viewBox="0 0 {width_px:.2f} {height_px:.2f}">{"".join(rects)}</svg>', encoding="utf-8")
    if failures and strict:
        raise RuntimeError(f"Panel alignment failed: {failures}")
    return report


def export(fig: plt.Figure, stem: str, *, alignment: bool = False) -> None:
    if alignment:
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig,
            json_out=OUT / f"{stem}.alignment.json",
            overlay_svg=OUT / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def rounded(ax, x, y, w, h, face, edge, radius=0.035, lw=0.7, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, size=6.5, weight="normal", ha="center", va="center", color=INK, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, **kw)


def arrow(ax, x1, y1, x2, y2, color=TRU, lw=1.4, mutation=10):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=mutation, linewidth=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=5))


def draw_document(ax, x, y, w, h, label, edge=TRU, face="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=0.8, zorder=3))
    for i in range(3):
        ax.plot([x+w*.18, x+w*.80], [y+h*(.70-i*.22)]*2, color=edge, lw=.6, zorder=4)
    text(ax, x+w/2, y+h*.15, label, size=5.8, weight="bold")


def draw_network(ax, x, y, w, h, color):
    xs = [x+w*.15, x+w*.43, x+w*.72, x+w*.95]
    counts = [3, 4, 3, 2]
    ys = [[y+h*(.2+i*.3) for i in range(n)] for n in counts]
    for li in range(len(counts)-1):
        for yy in ys[li]:
            for yn in ys[li+1]:
                ax.plot([xs[li], xs[li+1]], [yy, yn], color=color, alpha=.42, lw=.45, zorder=2)
    for xx, layer in zip(xs, ys):
        for yy in layer:
            ax.add_patch(Circle((xx, yy), h*.075, facecolor="white", edgecolor=color, linewidth=.7, zorder=3))


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.78))
    fig.subplots_adjust(left=.012, right=.988, top=.97, bottom=.08)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    left, gap, w, y0, h = .008, .014, .151, .20, .75
    stages = [
        ("Inputs", "Three datasets;\nsource integrity check"),
        ("Frozen models", "Pretrained checkpoints;\nno tuning"),
        ("Evaluation", "Clean inference +\ncontrolled transforms"),
        ("Predictions", "Detection score +\nlocalization map"),
        ("Evidence", "Per-sample records\nand output manifest"),
        ("Replay audit", "Reload and compare\nin a separate process"),
    ]
    xs = [left+i*(w+gap) for i in range(6)]
    for i, (x, (title, sub)) in enumerate(zip(xs, stages)):
        rounded(ax, x, y0, w, h, STAGE[i], STAGE_EDGE[i], radius=.018, lw=.85)
        ax.add_patch(Circle((x+.025, .922), .021, facecolor=["#4685C2", "#5B9163", "#B94F62", "#897D56", "#4685C2", "#7166A6"][i], edgecolor="none", lw=0))
        text(ax, x+.025, .922, str(i+1), size=8, weight="bold", color="white")
        text(ax, x+.052, .923, title, size=6.6, weight="bold", ha="left")
        text(ax, x+w/2, .855, sub, size=6.1)
        if i < 5:
            arrow(ax, x+w+.001, .565, x+w+gap-.002, .565, color="#5689B8", lw=1.25, mutation=9)

    # Stage 1: source sets represented by document marks, not fabricated samples.
    for j, name in enumerate(("Columbia", "CASIA v1", "COVERAGE")):
        yy = .68-j*.135
        draw_document(ax, xs[0]+.018, yy-.035, .040, .075, "", edge="#5086B6")
        text(ax, xs[0]+.085, yy, name, size=5.8, ha="left")
    rounded(ax, xs[0]+.014, .235, w-.028, .105, "white", STAGE_EDGE[0], radius=.012)
    text(ax, xs[0]+w/2, .306, "SHA-256", size=6.2, weight="bold")
    text(ax, xs[0]+w/2, .270, "source + mask hashes", size=5.8)

    # Stage 2: frozen detector blocks.
    for j, (name, color) in enumerate((("TruFor", TRU), ("MVSS-Net", MVSS))):
        yy = .64-j*.235
        rounded(ax, xs[1]+.013, yy-.095, w-.026, .18, "white", STAGE_EDGE[1], radius=.012)
        text(ax, xs[1]+w/2, yy+.062, name, size=6.0, weight="bold")
        draw_network(ax, xs[1]+.033, yy-.050, .085, .052, color)
        text(ax, xs[1]+w/2, yy-.080, "checkpoint frozen", size=5.8, color=MUTED)

    # Stage 3: clean path plus explicit stress conditions.
    for j, name in enumerate(("Clean", "JPEG q80 / q40", "Resize 0.5 / 0.25")):
        yy = .67-j*.15
        rounded(ax, xs[2]+.017, yy-.044, w-.034, .085, "white", STAGE_EDGE[2], radius=.012)
        text(ax, xs[2]+w/2, yy, name, size=5.8, weight="bold" if j == 0 else "normal")
        if j:
            text(ax, xs[2]+w/2, yy-.036, "deterministic transform", size=5.8, color=MUTED)

    # Stage 4: compact abstract score and map glyphs, explicitly illustrative.
    rounded(ax, xs[3]+.013, .585, w-.026, .15, "white", STAGE_EDGE[3], radius=.012)
    text(ax, xs[3]+w/2, .704, "Image-level score", size=5.8, weight="bold")
    ax.add_patch(Rectangle((xs[3]+.03, .625), .09, .028, facecolor="#E5E9EC", edgecolor="none"))
    ax.add_patch(Rectangle((xs[3]+.03, .625), .061, .028, facecolor="#5A99C9", edgecolor="none"))
    text(ax, xs[3]+w/2, .605, "fixed-threshold decision", size=5.8, color=MUTED)
    rounded(ax, xs[3]+.013, .365, w-.026, .18, "white", STAGE_EDGE[3], radius=.012)
    text(ax, xs[3]+w/2, .515, "Pixel map vs. valid mask", size=5.8, weight="bold")
    gridx, gridy = xs[3]+.036, .402
    for rr in range(4):
        for cc in range(6):
            hit = (rr in (1, 2) and cc in (2, 3, 4))
            ax.add_patch(Rectangle((gridx+cc*.013, gridy+rr*.022), .010, .018,
                                   facecolor="#D25A5A" if hit else "#D7E3EB",
                                   edgecolor="white", lw=.2))
    text(ax, xs[3]+w/2, .383, "schematic only", size=5.8, color=MUTED)

    # Stage 5: audit-trail records.
    for j, (title, desc) in enumerate((("Sample ID", "dataset / file"), ("SHA-256", "input integrity"), ("Model + config", "checkpoint / settings"), ("Output record", "score / map / time"))):
        yy = .70-j*.118
        rounded(ax, xs[4]+.014, yy-.038, w-.028, .075, "white", STAGE_EDGE[4], radius=.010)
        text(ax, xs[4]+.025, yy+.020, title, size=5.8, weight="bold", ha="left")
        text(ax, xs[4]+.025, yy-.020, desc, size=5.8, color=MUTED, ha="left")

    # Stage 6: separate execution and tolerance check.
    rounded(ax, xs[5]+.014, .625, w-.028, .13, "white", STAGE_EDGE[5], radius=.010)
    text(ax, xs[5]+w/2, .710, "Reload source + model", size=5.8, weight="bold")
    text(ax, xs[5]+w/2, .675, "separate process", size=5.8)
    text(ax, xs[5]+w/2, .635, "hash-check inputs", size=5.8, color=MUTED)
    rounded(ax, xs[5]+.014, .435, w-.028, .13, "white", STAGE_EDGE[5], radius=.010)
    text(ax, xs[5]+w/2, .520, "Compare score + map", size=5.8, weight="bold")
    text(ax, xs[5]+w/2, .480, "absolute tolerance 1e-5", size=5.8)
    arrow(ax, xs[5]+w/2, .615, xs[5]+w/2, .575, color="#7166A6", lw=1.0, mutation=8)
    rounded(ax, xs[5]+.014, .235, w-.028, .13, "white", STAGE_EDGE[5], radius=.010)
    text(ax, xs[5]+w/2, .323, "Audit report", size=5.8, weight="bold")
    text(ax, xs[5]+w/2, .278, "agreement / failures", size=5.8, color=MUTED)
    arrow(ax, xs[5]+w/2, .425, xs[5]+w/2, .375, color="#7166A6", lw=1.0, mutation=8)

    # Validation strip, echoing the provided reference's bottom checklist.
    rounded(ax, .008, .025, .984, .115, "#F4F7FA", "#D6E0E7", radius=.018, lw=.7)
    text(ax, .045, .082, "Audit scope", size=6.3, weight="bold", ha="left")
    items = ["Fixed checkpoints + threshold 0.5", "Valid masks only for localization", "No dataset / weight redistribution"]
    for x, item in zip((.29, .59, .86), items):
        ax.add_patch(Circle((x-.11, .082), .009, facecolor="#2B9872", edgecolor="none"))
        text(ax, x-.09, .082, item, size=5.8, ha="left")

    export(fig, "evaluation_workflow")


def get_detection_records() -> list[dict]:
    col_t = load("results/trufor_columbia_detection/summary.json")
    col_m = load("results/mvss_columbia_evaluation/summary.json")
    cas = load("results/casia_v1_evaluation/summary.json")["models"]
    cov = load("results/coverage_evaluation/summary.json")["models"]
    out = []
    for dataset, model, summary, metric_path, n_key in [
        ("Columbia", "TruFor", col_t, ("metrics",), "n_expected"),
        ("Columbia", "MVSS-Net", col_m, ("detection",), "n_expected"),
        ("CASIA v1", "TruFor", cas["trufor"], ("detection",), "n_detection"),
        ("CASIA v1", "MVSS-Net", cas["mvss"], ("detection",), "n_detection"),
        ("COVERAGE", "TruFor", cov["trufor"], ("detection",), "n_detection"),
        ("COVERAGE", "MVSS-Net", cov["mvss"], ("detection",), "n_detection"),
    ]:
        metric = summary
        for key in metric_path:
            metric = metric[key]
        ci = metric.get("roc_auc_group_bootstrap_95ci", metric.get("roc_auc_pair_bootstrap_95ci"))
        if ci is None:
            raise KeyError(f"Missing AUC interval for {dataset}/{model}")
        out.append({"dataset": dataset, "model": model, "n": summary[n_key],
                    "roc_auc": metric["roc_auc"], "ci_low": ci[0], "ci_high": ci[1]})
    return out


def figure2() -> None:
    rec = get_detection_records()
    cas = load("results/casia_v1_evaluation/summary.json")["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), gridspec_kw={"width_ratios": [1.35, 1]})
    colors = {"TruFor": TRU, "MVSS-Net": MVSS}
    datasets = ["Columbia", "CASIA v1", "COVERAGE"]
    x = np.arange(len(datasets)); offsets = {"TruFor": -.10, "MVSS-Net": .10}
    for model in ("TruFor", "MVSS-Net"):
        rs = [next(r for r in rec if r["dataset"] == d and r["model"] == model) for d in datasets]
        xx = x + offsets[model]
        yy = np.array([r["roc_auc"] for r in rs])
        err = np.array([[r["roc_auc"]-r["ci_low"] for r in rs], [r["ci_high"]-r["roc_auc"] for r in rs]])
        axes[0].errorbar(xx, yy, yerr=err, fmt="o", markersize=4.5, capsize=2.4,
                         color=colors[model], ecolor=colors[model], elinewidth=.9, label=model, zorder=3)
    axes[0].set_xticks(x, ["Columbia", "CASIA v1", "COVERAGE"])
    axes[0].set_ylim(.65, 1.04); axes[0].set_ylabel("Image-level ROC-AUC (95% CI)")
    axes[0].grid(axis="y", color=GRID, linestyle="--", linewidth=.5); axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=2, loc="lower left", handletextpad=.25, columnspacing=.8)
    axes[0].text(-.14, 1.06, "A", transform=axes[0].transAxes, fontsize=9, weight="bold")

    strata = [("Copy-move", "CM"), ("Splicing", "Sp")]
    xx = np.arange(2); width = .34
    for mi, model_key in enumerate(("trufor", "mvss")):
        model_name = "TruFor" if model_key == "trufor" else "MVSS-Net"
        vals = [cas[model_key]["localization_by_type"][typ]["f1"] for _, typ in strata]
        bars = axes[1].bar(xx+(mi-.5)*width, vals, width, color=colors[model_name], edgecolor="white", linewidth=.5, label=model_name)
        for bar, val in zip(bars, vals):
            axes[1].text(bar.get_x()+bar.get_width()/2, val+.018, f"{val:.3f}", ha="center", va="bottom", fontsize=5.7, color=colors[model_name])
    axes[1].set_xticks(xx, ["Copy-move", "Splicing"]); axes[1].set_ylim(0, .91)
    axes[1].set_ylabel("CASIA v1 localization macro F1")
    axes[1].grid(axis="y", color=GRID, linestyle="--", linewidth=.5); axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, ncol=2, loc="upper left", handletextpad=.25, columnspacing=.8)
    axes[1].text(-.17, 1.06, "B", transform=axes[1].transAxes, fontsize=9, weight="bold")
    fig.subplots_adjust(left=.085, right=.99, top=.89, bottom=.18, wspace=.38)
    require_matplotlib_panel_alignment(
        fig, json_out=QA_OUT / "cross_dataset_comparison.alignment.json",
        overlay_svg=QA_OUT / "cross_dataset_comparison.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        allow_unequal_width=True,
        width_exemption="Panel A widened to accommodate three dataset AUC estimates and confidence intervals.",
    )
    export(fig, "cross_dataset_comparison")
    with (OUT / "cross_dataset_comparison_source.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "n", "roc_auc", "ci_low", "ci_high"])
        writer.writeheader(); writer.writerows(rec)
    with (OUT / "casia_localization_by_type_source.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["model", "manipulation_type", "n", "macro_f1"])
        for model_key, model in (("trufor", "TruFor"), ("mvss", "MVSS-Net")):
            for label, typ in strata:
                item = cas[model_key]["localization_by_type"][typ]
                writer.writerow([model, label, item["n"], item["f1"]])


def figure3() -> None:
    clean = {
        (r["dataset"], r["model"]): r["roc_auc"]
        for r in get_detection_records()
        if r["dataset"] in ("Columbia", "CASIA v1")
    }
    stress = {
        ("Columbia", "TruFor"): load("results/trufor_full_stress_audit/summary.json"),
        ("Columbia", "MVSS-Net"): load("results/mvss_full_stress_audit/summary.json"),
        ("CASIA v1", "TruFor"): load("results/casia_v1_full_stress_audit/summary.json")["models"]["trufor"],
        ("CASIA v1", "MVSS-Net"): load("results/casia_v1_full_stress_audit/summary.json")["models"]["mvss"],
    }
    conditions = [("JPEG q80", "jpeg_q80"), ("JPEG q40", "jpeg_q40"), ("Resize 0.5", "resize_0.5"), ("Resize 0.25", "resize_0.25")]
    rows = [(d, m) for d in ("Columbia", "CASIA v1") for m in ("TruFor", "MVSS-Net")]
    auc_delta, flips, src = [], [], []
    for d, m in rows:
        summary = stress[(d, m)]
        aucs, frs = [], []
        for label, key in conditions:
            entry = summary["by_condition"][key]
            detection = entry["detection"]
            aucs.append(detection["roc_auc"]-clean[(d, m)])
            frs.append(entry["image_prediction_flip_rate"])
            src.append({"dataset": d, "model": m, "condition": label,
                        "clean_roc_auc": clean[(d,m)], "stress_roc_auc": detection["roc_auc"],
                        "delta_roc_auc": detection["roc_auc"]-clean[(d,m)],
                        "prediction_flip_rate": entry["image_prediction_flip_rate"],
                        "n": detection["n"]})
        auc_delta.append(aucs); flips.append(frs)
    auc_delta = np.asarray(auc_delta); flips = np.asarray(flips)*100
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    cmap = mpl.colors.LinearSegmentedColormap.from_list("shift", ["#B44E4B", "#F7F7F4", "#3D8C83"])
    rowlabels = [f"{d} · {m}" for d,m in rows]
    for ax, data, title, fmt, vmin, vmax in [
        (axes[0], auc_delta, "Change in ROC-AUC", ".3f", -.38, .02),
        (axes[1], flips, "Image decision flips (%)", ".1f", 0, 40),
    ]:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, aspect="auto", cmap=cmap if ax is axes[0] else "Blues", norm=norm)
        ax.set_xticks(np.arange(4), [x[0] for x in conditions], rotation=25, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(4), rowlabels)
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i,j]
                label = (f"{val:{fmt}}" if ax is axes[0] else f"{val:{fmt}}")
                textcolor = "white" if ((ax is axes[1] and val > 23) or (ax is axes[0] and val < -.24)) else INK
                ax.text(j, i, label, ha="center", va="center", fontsize=6.1, color=textcolor)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0, pad=4); ax.tick_params(axis="x", length=0, pad=4)
        for spine in ax.spines.values(): spine.set_visible(False)
    axes[0].text(-.18, 1.13, "A", transform=axes[0].transAxes, fontsize=9, weight="bold")
    axes[1].text(-.18, 1.13, "B", transform=axes[1].transAxes, fontsize=9, weight="bold")
    axes[0].set_ylabel("Dataset · frozen model")
    fig.subplots_adjust(left=.16, right=.99, top=.80, bottom=.30, wspace=.47)
    require_matplotlib_panel_alignment(
        fig, json_out=QA_OUT / "stress_sensitivity.alignment.json",
        overlay_svg=QA_OUT / "stress_sensitivity.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    export(fig, "stress_sensitivity")
    with (OUT / "stress_sensitivity_source.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(src[0].keys()))
        writer.writeheader(); writer.writerows(src)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
