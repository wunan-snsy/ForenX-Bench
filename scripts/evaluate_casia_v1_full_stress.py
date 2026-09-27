#!/usr/bin/env python3
"""Evaluate frozen TruFor and MVSS-Net CASIA v1 pressure-test predictions."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_casia_v1_models import localize, prediction_path, source_group
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci


def stress_prediction(model, item):
    return REPO / "results" / f"{model}_casia_v1_full_stress_outputs" / item["condition"] / item["class"] / (Path(item["path"]).name + ".npz")


def evaluate(model, manifest, config, root):
    records, failures = [], []
    for item in manifest:
        try:
            current = np.load(stress_prediction(model, item)); base = np.load(prediction_path(model, item))
            score, base_score = float(current["score"]), float(base["score"])
            record = {**item, "source_group": source_group(item["sample_id"], item["class"]),
                      "score": score, "baseline_score": base_score, "score_delta": score - base_score,
                      "prediction": int(score >= config["image_threshold"]),
                      "baseline_prediction": int(base_score >= config["image_threshold"])}
            record["prediction_flipped"] = int(record["prediction"] != record["baseline_prediction"])
            if item["mask_path"]:
                ref = np.asarray(Image.open(root / item["mask_path"]).convert("L")) > 127
                amap, base_map = np.asarray(current["map"], dtype=np.float32), np.asarray(base["map"], dtype=np.float32)
                if amap.shape != ref.shape or base_map.shape != ref.shape:
                    raise ValueError(f"map shape mismatch {amap.shape}/{base_map.shape}/{ref.shape}")
                record.update(localize(amap, ref, config["map_threshold"]))
                record["map_mae_from_baseline"] = float(np.abs(amap - base_map).mean())
                record["map_binary_flip_rate"] = float(((amap >= config["map_threshold"]) != (base_map >= config["map_threshold"])).mean())
            records.append(record)
        except Exception as exc:
            failures.append({"sample_id": item["sample_id"], "condition": item["condition"], "error": str(exc)})
    fields = ["precision", "recall", "f1", "iou", "mcc", "reference_prevalence", "prediction_prevalence", "map_mae_from_baseline", "map_binary_flip_rate"]
    result = {}
    for condition in config["conditions"]:
        subset = [r for r in records if r["condition"] == condition]
        labels = np.array([r["class"] == "tampered" for r in subset], dtype=int)
        scores = np.array([r["score"] for r in subset])
        detection = score_metrics(labels, scores, config["image_threshold"])
        ci_rows = [{"source_group": r["source_group"], "_y": int(r["class"] == "tampered")} for r in subset]
        detection["roc_auc_group_bootstrap_95ci"] = auc_ci(ci_rows, scores, config["bootstrap_replicates"], config["bootstrap_seed"])
        localized = [r for r in subset if "f1" in r]
        by_type = {}
        for kind in ("CM", "Sp"):
            typed = [r for r in localized if r["manipulation"].lower() == kind.lower()]
            by_type[kind] = {"n": len(typed), **{key: float(np.mean([r[key] for r in typed])) for key in fields}}
        result[condition] = {"n_detection": len(subset), "n_localization": len(localized), "detection": detection,
                             "mean_score_delta": float(np.mean([r["score_delta"] for r in subset])),
                             "image_prediction_flip_rate": float(np.mean([r["prediction_flipped"] for r in subset])),
                             "localization_macro": {key: float(np.mean([r[key] for r in localized])) for key in fields},
                             "localization_by_type": by_type}
    return {"status": "complete" if len(records) == 6884 and not failures else "incomplete", "n_evaluated": len(records),
            "by_condition": result, "failures": failures}


def main():
    config = json.loads((REPO / "configs/casia_v1_full_stress.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPO / "results/casia_v1_full_stress_inputs/manifest.json").read_text(encoding="utf-8"))
    root = REPO / "data/external/casia_v1_corrected"
    result = {"dataset": "CASIA v1 corrected + Pham et al. ground truth", "n_expected_per_model": 6884,
              "thresholds_frozen": True, "image_threshold": config["image_threshold"], "map_threshold": config["map_threshold"],
              "conditions": config["conditions"], "models": {model: evaluate(model, manifest, config, root) for model in ("trufor", "mvss")}}
    output = REPO / "results/casia_v1_full_stress_audit"; output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
