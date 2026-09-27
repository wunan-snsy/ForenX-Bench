#!/usr/bin/env python3
"""Create compact, resumable CASIA v1 stress-test derivatives."""
import csv
import json
from pathlib import Path

from PIL import Image
from run_columbia_blocks_baseline import REPO


def main():
    config = json.loads((REPO / "configs/casia_v1_full_stress.json").read_text(encoding="utf-8"))
    root = REPO / "data/external/casia_v1_corrected"
    destination = REPO / "results/casia_v1_full_stress_inputs"
    with (REPO / "manifests/casia_v1_manifest.csv").open(encoding="utf-8") as handle:
        originals = list(csv.DictReader(handle))
    records, created, reused = [], 0, 0
    for item in originals:
        for condition, spec in config["conditions"].items():
            target = destination / condition / item["class"] / f"{item['sample_id']}{spec['extension']}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size:
                reused += 1
            else:
                with Image.open(root / item["relative_path"]) as source:
                    image = source.convert("RGB")
                    if spec["type"] == "jpeg":
                        image.save(target, "JPEG", quality=spec["quality"], progressive=False, optimize=False)
                    else:
                        reduced = (max(1, round(image.width * spec["scale"])),
                                   max(1, round(image.height * spec["scale"])))
                        image.resize(reduced, Image.Resampling.LANCZOS).resize(
                            image.size, Image.Resampling.BICUBIC).save(
                                target, "JPEG", quality=spec["quality"], progressive=False, optimize=False)
                created += 1
            records.append({key: item[key] for key in ("sample_id", "class", "manipulation", "relative_path", "mask_path")}
                           | {"condition": condition, "path": target.relative_to(REPO).as_posix()})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_images": len(originals), "derivatives": len(records), "created": created,
                      "reused": reused, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
