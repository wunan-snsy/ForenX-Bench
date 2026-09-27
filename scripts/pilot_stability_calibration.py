#!/usr/bin/env python3
"""Holdout pilot for transform-stability based error-risk estimation.

This pilot does not alter either detector. It learns a small logistic risk model
from clean score margin and four predeclared CASIA stress-score deviations.
The held-out split is deterministic by source-group identifier.
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from evaluate_casia_v1_models import prediction_path, source_group
from run_columbia_blocks_baseline import REPO


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float((pos[:, None] > neg).mean() + .5 * (pos[:, None] == neg).mean())


def brier(labels, probabilities): return float(np.mean((np.asarray(labels) - np.asarray(probabilities)) ** 2))


def fit_logistic(x, y, steps=3000, learning_rate=.08, ridge=.02):
    mean, scale = x.mean(0), x.std(0); scale[scale == 0] = 1
    z = (x - mean) / scale; z = np.c_[np.ones(len(z)), z]; y = np.asarray(y, dtype=float)
    w = np.zeros(z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        grad = z.T @ (p - y) / len(y); grad[1:] += ridge * w[1:]
        w -= learning_rate * grad
    return mean, scale, w


def predict(model, x):
    mean, scale, w = model; z = np.c_[np.ones(len(x)), (x - mean) / scale]
    return 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))


def main():
    with (REPO / "manifests/casia_v1_manifest.csv").open(encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    stress = json.loads((REPO / "results/casia_v1_full_stress_inputs/manifest.json").read_text(encoding="utf-8"))
    grouped = {}
    for row in stress: grouped.setdefault(row["sample_id"], []).append(row)
    result = {"protocol": {"dataset": "CASIA v1 corrected", "split": "deterministic source-group hash < 60 for train; otherwise holdout", "threshold": .5, "conditions": ["jpeg_q80", "jpeg_q40", "resize_0.5", "resize_0.25"], "method": "L2-regularized logistic error-risk model using clean score margin, mean absolute score deviation, max score deviation, and threshold flip fraction"}, "models": {}}
    for model in ("trufor", "mvss"):
        records = []
        for row in original:
            base = float(np.load(prediction_path(model, row))["score"]); prediction = int(base >= .5); label = int(row["class"] == "tampered")
            derivatives = sorted(grouped[row["sample_id"]], key=lambda x: x["condition"])
            values = []
            for item in derivatives:
                path = REPO / "results" / f"{model}_casia_v1_full_stress_outputs" / item["condition"] / item["class"] / (Path(item["path"]).name + ".npz")
                values.append(float(np.load(path)["score"]))
            deviations = np.abs(np.asarray(values) - base)
            group = source_group(row["sample_id"], row["class"])
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
            records.append({"sample_id": row["sample_id"], "group": group, "split": "train" if bucket < 60 else "holdout", "correct": int(prediction == label), "baseline_score": base, "features": [abs(base - .5), float(deviations.mean()), float(deviations.max()), float(np.mean((np.asarray(values) >= .5) != prediction))]})
        train = [r for r in records if r["split"] == "train"]; test = [r for r in records if r["split"] == "holdout"]
        model_fit = fit_logistic(np.asarray([r["features"] for r in train]), [1-r["correct"] for r in train])
        risk = predict(model_fit, np.asarray([r["features"] for r in test])); errors = np.asarray([1-r["correct"] for r in test])
        ordered = np.argsort(risk)
        coverage = .80; kept = ordered[:round(len(test)*coverage)]
        result["models"][model] = {"n_train": len(train), "n_holdout": len(test), "holdout_errors": int(errors.sum()), "error_risk_auc": auc(errors, risk), "error_risk_brier": brier(errors, risk), "baseline_error_rate": float(errors.mean()), "risk_at_80pct_coverage": float(errors[kept].mean()), "relative_error_reduction_at_80pct_coverage": float(1-errors[kept].mean()/errors.mean()) if errors.mean() else 0., "mean_predicted_error_risk": float(risk.mean()), "features": ["clean_score_margin", "mean_abs_stress_score_delta", "max_abs_stress_score_delta", "stress_threshold_flip_fraction"]}
    output = REPO / "results/stability_calibration_pilot"; output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
