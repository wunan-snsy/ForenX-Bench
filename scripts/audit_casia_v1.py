#!/usr/bin/env python3
"""Audit corrected CASIA v1 images and Pham et al. ground-truth masks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main():
    root = REPO / "data/external/casia_v1_corrected"
    authentic = sorted((root / "Au").rglob("*.jpg")); tampered = sorted((root / "Tp").rglob("*.jpg"))
    masks = sorted((root / "CASIA 1.0 groundtruth").rglob("*.png"))
    mask_index = {p.stem.removesuffix("_gt").lower(): p for p in masks}
    records, shape_errors = [], []
    for label, paths in (("authentic", authentic), ("tampered", tampered)):
        for path in paths:
            mask = mask_index.get(path.stem.lower()) if label == "tampered" else None
            with Image.open(path) as image: width, height = image.size
            if mask:
                with Image.open(mask) as m:
                    if m.size != (width, height): shape_errors.append(path.stem)
            records.append({"sample_id": path.stem, "class": label,
                            "manipulation": path.parent.name if label == "tampered" else "none",
                            "relative_path": path.relative_to(root).as_posix(),
                            "mask_path": mask.relative_to(root).as_posix() if mask else "",
                            "width": width, "height": height, "sha256": digest(path),
                            "mask_sha256": digest(mask) if mask else ""})
    out = REPO / "manifests"; out.mkdir(exist_ok=True)
    with (out / "casia_v1_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    missing = [r["sample_id"] for r in records if r["class"] == "tampered" and not r["mask_path"]]
    audit = {"status": "complete", "dataset": "CASIA v1 corrected image bundle + Pham et al. ground truth",
             "authentic": len(authentic), "tampered": len(tampered), "masks": len(masks),
             "matched_masks": len(tampered) - len(missing), "missing_masks": missing,
             "shape_mismatches": shape_errors, "mask_foreground_rule": "grayscale > 127",
             "image_archive_sha256": "326355571cf3137d061b7322dd8d50e258afa3eada32c71718d645e4d9ca270c",
             "groundtruth_archive_sha256": "c552558f661be4b105dea60a7a6e9ae11bb6bc41497dfc7815fdd670ab723a1d"}
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
