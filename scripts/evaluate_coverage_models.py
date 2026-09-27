#!/usr/bin/env python3
"""Evaluate frozen TruFor and MVSS-Net on audited COVERAGE pairs."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_casia_v1_models import localize
from run_columbia_blocks_baseline import REPO, score_metrics, auc_ci


def path_for(model, item):
    return REPO / "results" / f"{model}_coverage_{item['class']}" / (Path(item["relative_path"]).name + ".npz")


def evaluate(model, manifest):
    rows, failures = [], []
    root = REPO / "data/external/COVERAGE"
    for item in manifest:
        try:
            data = np.load(path_for(model, item)); score = float(data["score"])
            row = {"sample_id": item["sample_id"], "class": item["class"], "pair_id": int(item["pair_id"]), "score": score, "source_group": f"pair:{item['pair_id']}"}
            if item["mask_path"]:
                mask = np.asarray(Image.open(root / item["mask_path"]).convert("L")) > 127
                amap = np.asarray(data["map"], dtype=np.float32)
                if amap.shape != mask.shape: raise ValueError(f"map shape mismatch {amap.shape}/{mask.shape}")
                row.update(localize(amap, mask, .5))
            rows.append(row)
        except Exception as exc: failures.append({"sample_id": item["sample_id"], "error": str(exc)})
    labels = np.array([r["class"] == "tampered" for r in rows], dtype=int); scores = np.array([r["score"] for r in rows])
    detection = score_metrics(labels, scores, .5)
    detection["roc_auc_pair_bootstrap_95ci"] = auc_ci([{ "source_group": r["source_group"], "_y": int(r["class"] == "tampered")} for r in rows], scores, 2000, 20260926)
    localized = [r for r in rows if "f1" in r]; fields = ["precision", "recall", "f1", "iou", "mcc", "reference_prevalence", "prediction_prevalence"]
    return {"status": "complete" if len(rows) == 200 and not failures else "incomplete", "n_detection": len(rows), "n_localization": len(localized), "threshold": .5, "detection": detection, "localization_macro": {field: float(np.mean([r[field] for r in localized])) for field in fields}, "failures": failures}


def main():
    with (REPO / "manifests/coverage_manifest.csv").open(encoding="utf-8") as handle: manifest = list(csv.DictReader(handle))
    result = {"dataset": "COVERAGE corrected release", "n_detection_expected": 200, "n_localization_expected": 91, "thresholds_frozen": True, "models": {m: evaluate(m, manifest) for m in ("trufor", "mvss")}}
    out = REPO / "results/coverage_evaluation"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
