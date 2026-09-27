#!/usr/bin/env python3
"""Evaluate full Columbia TruFor stress runs at frozen thresholds."""
import csv
import json
from collections import defaultdict
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci


def decode_boundary(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    p = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.float32)
    return ((rgb[..., None, :] - p) ** 2).sum(-1).argmin(-1) == 2


def spatial(score, reference, threshold):
    pred = score >= threshold
    tp = int((pred & reference).sum()); tn = int((~pred & ~reference).sum())
    fp = int((pred & ~reference).sum()); fn = int((~pred & reference).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "mcc": (tp * tn - fp * fn) / den if den else 0.0,
            "prediction_prevalence": float(pred.mean())}, pred


def main():
    cfg = json.loads((REPO / "configs/trufor_full_stress.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPO / "results/trufor_full_stress_inputs/manifest.json").read_text(encoding="utf-8"))
    output = REPO / "results/trufor_full_stress_outputs"
    dataset = REPO / "data/external/columbia_uncompressed"
    records, failures = [], []
    for item in manifest:
        try:
            source_path = REPO / item["path"]
            current_path = output / item["condition"] / item["class"] / (source_path.name + ".npz")
            base_folder = "trufor_columbia_authentic" if item["class"] == "authentic" else "trufor_columbia_spliced"
            baseline_path = REPO / "results" / base_folder / (item["sample_id"] + ".tif.npz")
            current = np.load(current_path); baseline = np.load(baseline_path)
            score = float(current["score"]); base_score = float(baseline["score"])
            row = {**item, "image_score": score, "baseline_image_score": base_score,
                   "score_delta": score - base_score,
                   "prediction": int(score >= cfg["image_threshold"]),
                   "baseline_prediction": int(base_score >= cfg["image_threshold"]),
                   "prediction_flipped": int((score >= cfg["image_threshold"]) !=
                                             (base_score >= cfg["image_threshold"]))}
            if item["class"] == "spliced":
                amap = np.asarray(current["map"], dtype=np.float32)
                base_map = np.asarray(baseline["map"], dtype=np.float32)
                ref = decode_boundary(dataset / "4cam_splc/edgemask" /
                                      (item["sample_id"] + "_edgemask_3.jpg"))
                if amap.shape != ref.shape or base_map.shape != ref.shape:
                    raise ValueError(f"shape mismatch {amap.shape}/{base_map.shape}/{ref.shape}")
                loc, pred = spatial(amap, ref, cfg["map_threshold"])
                row.update({"map_mae_from_baseline": float(np.abs(amap - base_map).mean()),
                            "map_binary_flip_rate": float((pred != (base_map >= cfg["map_threshold"])).mean()),
                            **loc})
            records.append(row)
        except Exception as exc:
            failures.append({**item, "error": str(exc)})

    summaries = {}
    for condition in cfg["conditions"]:
        subset = [r for r in records if r["condition"] == condition]
        y = np.array([int(r["class"] == "spliced") for r in subset])
        scores = np.array([r["image_score"] for r in subset])
        metrics = score_metrics(y, scores, cfg["image_threshold"])
        ci_rows = [{"source_group": r["source_group"], "_y": int(r["class"] == "spliced")} for r in subset]
        metrics["roc_auc_group_bootstrap_95ci"] = auc_ci(
            ci_rows, scores, cfg["bootstrap_replicates"], cfg["bootstrap_seed"])
        spliced = [r for r in subset if r["class"] == "spliced"]
        loc_fields = ["precision", "recall", "f1", "iou", "mcc", "prediction_prevalence",
                      "map_mae_from_baseline", "map_binary_flip_rate"]
        summaries[condition] = {
            "n": len(subset), "detection": metrics,
            "mean_score_delta": float(np.mean([r["score_delta"] for r in subset])),
            "image_prediction_flip_rate": float(np.mean([r["prediction_flipped"] for r in subset])),
            "spatial_spliced_macro": {k: float(np.mean([r[k] for r in spliced])) for k in loc_fields}}
    result = {"status": "complete" if len(records) == 1452 and not failures else "incomplete",
              "n_expected": 1452, "n_evaluated": len(records), "thresholds_frozen": True,
              "image_threshold": cfg["image_threshold"], "map_threshold": cfg["map_threshold"],
              "by_condition": summaries, "failures": failures,
              "weights_md5": "7bee48f3476c75616c3c5721ab256ff8"}
    out = REPO / "results/trufor_full_stress_audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    keys = sorted(set().union(*(r.keys() for r in records)))
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
