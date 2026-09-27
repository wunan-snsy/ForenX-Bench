#!/usr/bin/env python3
"""Extract the user-supplied CASIA v2 images and ground-truth archives."""
import zipfile
from pathlib import Path
from run_columbia_blocks_baseline import REPO

out = REPO / "data/external/casia_v2_raw"
out.mkdir(parents=True, exist_ok=True)
for archive in (REPO / "data/external/archive.zip", REPO / "data/external/CASIA2.0_Groundtruth.zip"):
    with zipfile.ZipFile(archive) as source:
        source.extractall(out)
print("complete")
