#!/usr/bin/env python3
"""Evaluate TruFor and MVSS-Net on corrected CASIA v1 with Pham masks."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci


def source_group(sample_id, label):
    parts = sample_id.lower().split("_")
    if label == "tampered" and len(parts) >= 4:
        return "tp:" + "|".join(sorted(parts[-3:-1]))
    return "au:" + sample_id.lower().removeprefix("au_")


def localize(amap, ref, threshold=.5):
    pred = amap >= threshold; tp = int((pred & ref).sum()); tn = int((~pred & ~ref).sum())
    fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    precision = tp / (tp + fp) if tp + fp else 0.; recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "mcc": (tp * tn - fp * fn) / den if den else 0.,
            "reference_prevalence": float(ref.mean()), "prediction_prevalence": float(pred.mean())}


def prediction_path(model, row):
    image = Path(row["relative_path"])
    if model == "trufor":
        folder = "trufor_casia_v1_authentic" if row["class"] == "authentic" else "trufor_casia_v1_tampered"
        relative = Path(image.name) if row["class"] == "authentic" else Path(*image.parts[1:])
        return REPO / "results" / folder / (str(relative) + ".npz")
    folder = "mvss_casia_v1_authentic" if row["class"] == "authentic" else "mvss_casia_v1_tampered"
    return REPO / "results" / folder / (image.name + ".npz")


def evaluate(model, manifest, data_root):
    rows, failures = [], []
    for item in manifest:
        try:
            z = np.load(prediction_path(model, item)); score = float(z["score"])
            row = {"sample_id": item["sample_id"], "class": item["class"],
                   "manipulation": item["manipulation"], "score": score,
                   "source_group": source_group(item["sample_id"], item["class"])}
            if item["mask_path"]:
                mask = np.asarray(Image.open(data_root / item["mask_path"]).convert("L")) > 127
                amap = np.asarray(z["map"], dtype=np.float32)
                if amap.shape != mask.shape: raise ValueError(f"shape mismatch {amap.shape}/{mask.shape}")
                row.update(localize(amap, mask))
            rows.append(row)
        except Exception as exc: failures.append({"sample_id": item["sample_id"], "error": str(exc)})
    labels = np.array([int(r["class"] == "tampered") for r in rows]); scores = np.array([r["score"] for r in rows])
    detection = score_metrics(labels, scores, .5)
    cirows = [{"source_group": r["source_group"], "_y": int(r["class"] == "tampered")} for r in rows]
    detection["roc_auc_group_bootstrap_95ci"] = auc_ci(cirows, scores, 2000, 20261003)
    localized = [r for r in rows if "f1" in r]
    fields = ["precision", "recall", "f1", "iou", "mcc", "reference_prevalence", "prediction_prevalence"]
    by_type = {}
    for kind in ("CM", "Sp"):
        subset = [r for r in localized if r["manipulation"].lower() == kind.lower()]
        by_type[kind] = {"n": len(subset), **{k: float(np.mean([r[k] for r in subset])) for k in fields}}
    return {"status": "complete" if len(rows) == 1721 and not failures else "incomplete",
            "n_detection": len(rows), "n_localization": len(localized), "threshold": .5,
            "detection": detection,
            "score_by_class": {"authentic_mean": float(scores[labels == 0].mean()),
                               "tampered_mean": float(scores[labels == 1].mean())},
            "localization_macro": {k: float(np.mean([r[k] for r in localized])) for k in fields},
            "localization_by_type": by_type, "failures": failures}


def main():
    data_root = REPO / "data/external/casia_v1_corrected"
    with (REPO / "manifests/casia_v1_manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    result = {"dataset": "CASIA v1 corrected + Pham et al. ground truth",
              "mask_foreground": "grayscale > 127", "models": {}}
    for model in ("trufor", "mvss"):
        result["models"][model] = evaluate(model, manifest, data_root)
    out = REPO / "results/casia_v1_evaluation"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
