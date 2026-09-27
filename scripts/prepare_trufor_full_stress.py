#!/usr/bin/env python3
"""Create resumable full-dataset Columbia stress inputs for frozen TruFor."""
import json
from pathlib import Path
from PIL import Image
from run_columbia_blocks_baseline import REPO
from run_columbia_full_detection import rows


def main():
    cfg = json.loads((REPO / "configs/trufor_full_stress.json").read_text(encoding="utf-8"))
    dataset = REPO / "data/external/columbia_uncompressed"
    out = REPO / "results/trufor_full_stress_inputs"
    manifest, created, reused = [], 0, 0
    for item in rows(dataset):
        for condition, spec in cfg["conditions"].items():
            target = out / condition / item["class"] / f"{item['sample_id']}{spec['extension']}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size:
                reused += 1
            else:
                with Image.open(item["path"]) as source:
                    image = source.convert("RGB")
                    if spec["type"] == "jpeg":
                        image.save(target, "JPEG", quality=spec["quality"], progressive=False, optimize=False)
                    else:
                        size = (max(1, round(image.width * spec["scale"])),
                                max(1, round(image.height * spec["scale"])))
                        image.resize(size, Image.Resampling.LANCZOS).resize(
                            image.size, Image.Resampling.BICUBIC).save(target)
                created += 1
            manifest.append({"sample_id": item["sample_id"], "class": item["class"],
                             "source_group": item["source_group"], "condition": condition,
                             "path": target.relative_to(REPO).as_posix()})
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_images": len(rows(dataset)), "derivatives": len(manifest),
                      "created": created, "reused": reused, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
