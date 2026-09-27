"""Evaluate the dual-task CASIA v2 model on held-out test images.

The localization threshold is selected on the validation split only. Results
are internal experimental evidence: inspect split leakage before publication.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_columbia_blocks_baseline import REPO, score_metrics
from train_forenx_dualbranch_v1 import DataV1, NetV1


def infer(split, net, device):
    ds = DataV1(split)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    rows = []
    net.eval()
    with torch.no_grad():
        for x, mask, label, valid in loader:
            logits, cls = net(x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            masks = mask.numpy()[:, 0]
            labels = label.numpy().astype(int)
            valids = valid.numpy().astype(bool)
            class_scores = torch.sigmoid(cls[:, 0]).cpu().numpy()
            for i in range(len(labels)):
                rows.append({"score": float(class_scores[i]), "label": int(labels[i]),
                             "prob": probs[i], "mask": masks[i], "valid": bool(valids[i])})
    return rows


def localization(rows, threshold):
    tp = fp = fn = 0
    per_image = []
    for row in rows:
        if not row["valid"]:
            continue
        pred = row["prob"] >= threshold
        ref = row["mask"] >= 0.5
        a = int(np.logical_and(pred, ref).sum())
        b = int(np.logical_and(pred, np.logical_not(ref)).sum())
        c = int(np.logical_and(np.logical_not(pred), ref).sum())
        tp += a
        fp += b
        fn += c
        per_image.append({"f1": 2*a/max(1, 2*a+b+c), "iou": a/max(1, a+b+c)})
    return {"n_images_with_masks": len(per_image),
            "pixel_f1": 2*tp/max(1, 2*tp+fp+fn),
            "pixel_iou": tp/max(1, tp+fp+fn),
            "image_macro_f1": float(np.mean([r["f1"] for r in per_image])) if per_image else None,
            "image_macro_iou": float(np.mean([r["iou"] for r in per_image])) if per_image else None}


def main():
    ckpt_path = REPO / "results/forenx_dualbranch_v1/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NetV1().to(device)
    net.load_state_dict(ckpt["model"])
    val = infer("validation", net, device)
    test = infer("test", net, device)

    thresholds = np.arange(0.05, 0.951, 0.025)
    candidates = [(localization(val, float(t))["pixel_f1"], float(t)) for t in thresholds]
    best_f1, threshold = max(candidates)
    scores = np.array([r["score"] for r in test])
    labels = np.array([r["label"] for r in test])
    det = score_metrics(labels, scores, 0.5)
    report = {
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_selection": "highest validation image-level AUC",
        "validation_image_auc": float(score_metrics(
            np.array([r["label"] for r in val]), np.array([r["score"] for r in val]), 0.5
        )["roc_auc"]),
        "validation_localization_threshold": threshold,
        "validation_pixel_f1_at_selected_threshold": best_f1,
        "test_detection": det,
        "test_localization": localization(test, threshold),
        "mask_coverage": {"validation": sum(r["valid"] for r in val),
                          "test": sum(r["valid"] for r in test)},
        "warning": "CASIA v2 split audit required: authentic source images may occur across splits independently of tampered source-pair groups.",
    }
    out = REPO / "results/forenx_dualbranch_v1/test_evaluation.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
