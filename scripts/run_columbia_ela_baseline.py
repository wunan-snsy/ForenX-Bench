#!/usr/bin/env python3
"""Group-split Error Level Analysis baseline on Columbia Uncompressed."""
import csv
import io
import json
import numpy as np
from PIL import Image, ImageFilter
from run_columbia_blocks_baseline import REPO, fit, predict, threshold_dev, score_metrics, auc_ci
from run_columbia_full_detection import rows, split


def ela_features(path, quality):
    with Image.open(path) as source:
        image = source.convert("RGB")
    buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=quality, optimize=False, progressive=False)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    x = np.asarray(image, dtype=np.float32) / 255.0
    y = np.asarray(recompressed, dtype=np.float32) / 255.0
    residual = np.abs(x - y)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    blur = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0
    highpass = np.abs(gray - blur)
    values = []
    for plane in [residual[..., 0], residual[..., 1], residual[..., 2], residual.mean(2), highpass]:
        values.extend([float(plane.mean()), float(plane.std()),
                       *[float(np.percentile(plane, q)) for q in (50, 75, 90, 95, 99)]])
    # JPEG block-position signature: mean ELA at each position in an 8x8 lattice.
    ela_gray = residual.mean(2)
    for row in range(8):
        for col in range(8):
            values.append(float(ela_gray[row::8, col::8].mean()))
    # Coarse spatial distribution of residual energy.
    for band in np.array_split(ela_gray, 4, axis=0):
        for cell in np.array_split(band, 4, axis=1):
            values.extend([float(cell.mean()), float(cell.std())])
    return np.asarray(values, dtype=np.float64)


def main():
    cfg = json.loads((REPO / "configs/columbia_ela_baseline.json").read_text(encoding="utf-8"))
    data = rows(REPO / cfg["dataset_root"])
    partitions = split(data, cfg["split_ratios"], cfg["seed"])
    cache = {item["sample_id"]: ela_features(item["path"], cfg["jpeg_recompression_quality"])
             for item in data}
    xx = lambda items: np.stack([cache[item["sample_id"]] for item in items])
    yy = lambda items: np.asarray([item["_y"] for item in items])
    model = fit(xx(partitions["train"]), yy(partitions["train"]), cfg)
    threshold = threshold_dev(yy(partitions["dev"]), predict(model, xx(partitions["dev"])))
    test = partitions["test"]; scores = predict(model, xx(test)); labels = yy(test)
    metrics = score_metrics(labels, scores, threshold)
    metrics["roc_auc_group_bootstrap_95ci"] = auc_ci(
        test, scores, cfg["bootstrap_replicates"], cfg["bootstrap_seed"])
    result = {"status": "complete", "method": "ELA q90 + high-pass statistics + L2 logistic regression",
              "feature_dimensions": int(len(next(iter(cache.values())))),
              "counts": {name: {"n": len(items), "authentic": sum(x["_y"] == 0 for x in items),
                                "spliced": sum(x["_y"] == 1 for x in items)}
                         for name, items in partitions.items()},
              "threshold": threshold, "threshold_selected_on": "dev", "test_metrics": metrics}
    out = REPO / "results/columbia_ela_baseline"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["sample_id", "class", "source_group", "score", "prediction"])
        for item, score in zip(test, scores):
            writer.writerow([item["sample_id"], item["class"], item["source_group"], float(score),
                             "spliced" if score >= threshold else "authentic"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
