#!/usr/bin/env python3
"""Evaluate frozen MVSS-Net CASIAv2 checkpoint on Columbia Uncompressed."""
import json
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci
from run_columbia_full_detection import rows


def boundary(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    p = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.float32)
    return ((rgb[..., None, :] - p) ** 2).sum(-1).argmin(-1) == 2


def localization(score, ref, threshold=0.5):
    pred = score >= threshold
    tp = int((pred & ref).sum()); tn = int((~pred & ~ref).sum())
    fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "mcc": (tp * tn - fp * fn) / den if den else 0.0,
            "prediction_prevalence": float(pred.mean())}


def main():
    dataset = REPO / "data/external/columbia_uncompressed"; items = rows(dataset)
    used, scores, loc, failures = [], [], [], []
    for item in items:
        folder = "mvss_columbia_authentic" if item["class"] == "authentic" else "mvss_columbia_spliced"
        path = REPO / "results" / folder / (item["path"].name + ".npz")
        try:
            result = np.load(path); used.append(item); scores.append(float(result["score"]))
            if item["class"] == "spliced":
                ref = boundary(dataset / "4cam_splc/edgemask" / (item["sample_id"] + "_edgemask_3.jpg"))
                amap = np.asarray(result["map"], dtype=np.float32)
                if amap.shape != ref.shape: raise ValueError(f"shape mismatch {amap.shape}/{ref.shape}")
                loc.append(localization(amap, ref))
        except Exception as exc: failures.append({"sample_id": item["sample_id"], "error": str(exc)})
    s = np.asarray(scores); y = np.asarray([item["_y"] for item in used])
    detection = score_metrics(y, s, 0.5)
    detection["roc_auc_group_bootstrap_95ci"] = auc_ci(used, s, 2000, 20261001)
    result = {"status": "complete" if len(used) == 363 and not failures else "incomplete",
              "n_expected": 363, "n_evaluated": len(used), "threshold": 0.5,
              "image_score_definition": "maximum pixel probability on official 512x512 inference map",
              "detection": detection,
              "score_by_class": {"authentic_mean": float(s[y == 0].mean()),
                                  "spliced_mean": float(s[y == 1].mean())},
              "spatial_spliced_macro": {k: float(np.mean([v[k] for v in loc])) for k in loc[0]},
              "weights_sha256": "080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50",
              "failures": failures}
    out = REPO / "results/mvss_columbia_evaluation"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
