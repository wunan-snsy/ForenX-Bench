#!/usr/bin/env python3
"""Audit the corrected COVERAGE release and write a local-only manifest."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_columbia_blocks_baseline import REPO


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shape(path):
    with Image.open(path) as image:
        return image.size


def main():
    root = REPO / "data/external/COVERAGE"; images = root / "image"; masks = root / "mask"
    rows, failures = [], []
    for number in range(1, 101):
        original, forged = images / f"{number}.tif", images / f"{number}t.tif"
        annotations = {kind: masks / f"{number}{kind}.tif" for kind in ("copy", "paste", "forged")}
        missing = [str(path) for path in (original, forged, *annotations.values()) if not path.is_file()]
        if missing:
            failures.append({"id": number, "error": "missing", "paths": missing}); continue
        try:
            original_size, forged_size = shape(original), shape(forged)
            masks_size = {kind: shape(path) for kind, path in annotations.items()}
            image_consistent = original_size == forged_size
            mask_consistent = all(size == forged_size for size in masks_size.values())
            if not image_consistent:
                raise ValueError(f"paired-image dimensions: original={original_size}, forged={forged_size}")
            if not mask_consistent:
                failures.append({"id": number, "error": "mask_dimension_mismatch", "image": forged_size, "masks": masks_size})
            union = np.asarray(Image.open(annotations["forged"]).convert("L")) > 127 if mask_consistent else None
            rows.extend([
                {"sample_id": f"coverage_{number:03d}", "class": "authentic", "pair_id": number,
                 "relative_path": original.relative_to(root).as_posix(), "mask_path": "", "width": original_size[0], "height": original_size[1], "sha256": digest(original), "mask_sha256": ""},
                {"sample_id": f"coverage_{number:03d}t", "class": "tampered", "pair_id": number,
                 "relative_path": forged.relative_to(root).as_posix(), "mask_path": annotations["forged"].relative_to(root).as_posix() if mask_consistent else "", "width": forged_size[0], "height": forged_size[1], "sha256": digest(forged), "mask_sha256": digest(annotations["forged"]) if mask_consistent else "", "mask_foreground_pixels": int(union.sum()) if union is not None else ""}
            ])
        except Exception as exc:
            failures.append({"id": number, "error": str(exc)})
    manifest = REPO / "manifests/coverage_manifest.csv"; manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "class", "pair_id", "relative_path", "mask_path", "width", "height", "sha256", "mask_sha256", "mask_foreground_pixels"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    result = {"dataset": "COVERAGE corrected release", "archive_sha256": digest(REPO / "data/external/COVERAGE.zip"),
              "n_authentic": sum(row["class"] == "authentic" for row in rows), "n_tampered": sum(row["class"] == "tampered" for row in rows),
              "n_masked": sum(bool(row["mask_path"]) for row in rows), "mask_semantics": "author-supplied forged-region mask; grayscale > 127",
              "status": "complete_with_localization_exclusions" if len(rows) == 200 else "incomplete", "localization_exclusions": len(failures), "failures": failures}
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
