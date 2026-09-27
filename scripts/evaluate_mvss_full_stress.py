#!/usr/bin/env python3
"""Evaluate frozen MVSS-Net over the shared full Columbia stress manifest."""
import json
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci


def boundary(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    p = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.float32)
    return ((rgb[..., None, :] - p) ** 2).sum(-1).argmin(-1) == 2


def spatial(score, ref, threshold=0.5):
    pred = score >= threshold; tp = int((pred & ref).sum()); tn = int((~pred & ~ref).sum())
    fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    precision = tp / (tp + fp) if tp + fp else 0.; recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "mcc": (tp * tn - fp * fn) / den if den else 0.,
            "prediction_prevalence": float(pred.mean())}, pred


def main():
    manifest = json.loads((REPO / "results/trufor_full_stress_inputs/manifest.json").read_text())
    root = REPO / "results/mvss_full_stress_outputs"; data = REPO / "data/external/columbia_uncompressed"
    records, failures = [], []
    for item in manifest:
        try:
            source = REPO / item["path"]
            current = np.load(root / item["condition"] / item["class"] / (source.name + ".npz"))
            base_folder = "mvss_columbia_authentic" if item["class"] == "authentic" else "mvss_columbia_spliced"
            baseline = np.load(REPO / "results" / base_folder / (item["sample_id"] + ".tif.npz"))
            score, base_score = float(current["score"]), float(baseline["score"])
            row = {**item, "image_score": score, "score_delta": score - base_score,
                   "prediction_flipped": int((score >= .5) != (base_score >= .5))}
            if item["class"] == "spliced":
                amap = np.asarray(current["map"]); bmap = np.asarray(baseline["map"])
                ref = boundary(data / "4cam_splc/edgemask" / (item["sample_id"] + "_edgemask_3.jpg"))
                loc, pred = spatial(amap, ref); row.update(loc)
                row["map_mae_from_baseline"] = float(np.abs(amap - bmap).mean())
                row["map_binary_flip_rate"] = float((pred != (bmap >= .5)).mean())
            records.append(row)
        except Exception as exc: failures.append({**item, "error": str(exc)})
    result = {"status": "complete" if len(records) == 1452 and not failures else "incomplete",
              "n_expected": 1452, "n_evaluated": len(records), "image_threshold": .5, "map_threshold": .5,
              "by_condition": {}, "failures": failures,
              "weights_sha256": "080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50"}
    for condition in ("jpeg_q80", "jpeg_q40", "resize_0.5", "resize_0.25"):
        subset = [r for r in records if r["condition"] == condition]
        y = np.array([int(r["class"] == "spliced") for r in subset]); scores = np.array([r["image_score"] for r in subset])
        metrics = score_metrics(y, scores, .5)
        cirows = [{"source_group": r["source_group"], "_y": int(r["class"] == "spliced")} for r in subset]
        metrics["roc_auc_group_bootstrap_95ci"] = auc_ci(cirows, scores, 2000, 20261002)
        spliced = [r for r in subset if r["class"] == "spliced"]
        fields = ["precision", "recall", "f1", "iou", "mcc", "prediction_prevalence",
                  "map_mae_from_baseline", "map_binary_flip_rate"]
        result["by_condition"][condition] = {"detection": metrics,
            "mean_score_delta": float(np.mean([r["score_delta"] for r in subset])),
            "image_prediction_flip_rate": float(np.mean([r["prediction_flipped"] for r in subset])),
            "spatial_spliced_macro": {k: float(np.mean([r[k] for r in spliced])) for k in fields}}
    out = REPO / "results/mvss_full_stress_audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
