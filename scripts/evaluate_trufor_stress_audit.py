#!/usr/bin/env python3
"""Evaluate TruFor localization and prediction stability under controlled distortions."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from run_columbia_blocks_baseline import REPO

THRESHOLD = 0.5


def decode_boundary(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    prototypes = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.float32)
    return ((rgb[..., None, :] - prototypes) ** 2).sum(-1).argmin(-1) == 2


def localization(score, ref):
    pred = score >= THRESHOLD
    tp = int((pred & ref).sum()); tn = int((~pred & ~ref).sum())
    fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "mcc": mcc, "prediction_prevalence": float(pred.mean())}, pred


def main():
    manifest_path = REPO / "results/trufor_stress_inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = REPO / "data/external/columbia_uncompressed/4cam_splc"
    baseline_root = REPO / "results/trufor_columbia_spliced"
    output_root = REPO / "results/trufor_stress_outputs"
    rows, failures = [], []
    for item in manifest:
        try:
            derivative = REPO / item["path"]
            current = np.load(output_root / (derivative.name + ".npz"))
            baseline = np.load(baseline_root / (item["sample_id"] + ".tif.npz"))
            score = np.asarray(current["map"], dtype=np.float32)
            base_map = np.asarray(baseline["map"], dtype=np.float32)
            ref = decode_boundary(source / "edgemask" / (item["sample_id"] + "_edgemask_3.jpg"))
            if score.shape != ref.shape or base_map.shape != ref.shape:
                raise ValueError(f"shape mismatch: {score.shape}, {base_map.shape}, {ref.shape}")
            metrics, pred = localization(score, ref)
            base_pred = base_map >= THRESHOLD
            rows.append({**item, "image_score": float(current["score"]),
                         "baseline_image_score": float(baseline["score"]),
                         "score_delta": float(current["score"] - baseline["score"]),
                         "map_mae_from_baseline": float(np.abs(score - base_map).mean()),
                         "binary_flip_rate": float((pred != base_pred).mean()), **metrics})
        except Exception as exc:
            failures.append({**item, "error": str(exc)})

    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    fields = ["image_score", "baseline_image_score", "score_delta", "map_mae_from_baseline",
              "binary_flip_rate", "precision", "recall", "f1", "iou", "mcc", "prediction_prevalence"]
    by_condition = {
        condition: {"n": len(values), **{key: float(np.mean([v[key] for v in values])) for key in fields}}
        for condition, values in sorted(groups.items())
    }
    summary = {"status": "complete" if not failures and len(rows) == 24 else "incomplete",
               "n_expected": 24, "n_evaluated": len(rows), "n_source_images": 6,
               "selection": "lexicographically first spliced image from each of six camera pairs",
               "map_threshold": THRESHOLD, "by_condition": by_condition, "failures": failures,
               "weights_md5": "7bee48f3476c75616c3c5721ab256ff8"}
    out = REPO / "results/trufor_stress_audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
