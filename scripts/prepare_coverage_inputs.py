#!/usr/bin/env python3
"""Create class-separated hard-link inputs without changing COVERAGE images."""
import csv, os, shutil
from pathlib import Path
from run_columbia_blocks_baseline import REPO

def main():
    root = REPO / "data/external/COVERAGE"; out = REPO / "results/coverage_inputs"
    with (REPO / "manifests/coverage_manifest.csv").open(encoding="utf-8") as h: rows = list(csv.DictReader(h))
    for row in rows:
        src = root / row["relative_path"]; dst = out / row["class"] / src.name; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try: os.link(src, dst)
            except OSError: shutil.copy2(src, dst)
    print(f"prepared {len(rows)} class-separated COVERAGE inputs")
if __name__ == "__main__": main()
