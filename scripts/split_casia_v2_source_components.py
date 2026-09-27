"""Leakage-resistant CASIA v2 split by connected source-image families.

Every tampered sample joins the connected component of its two source images;
any available authentic originals join the same component. Components are
assigned deterministically. The largest component is assigned to training.
"""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from run_columbia_blocks_baseline import REPO

rows = list(csv.DictReader((REPO / "manifests/casia_v2_manifest.csv").open(encoding="utf-8")))
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

def normalize_source(token):
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", token)
    return f"{m.group(1)}_{m.group(2)}" if m else token

def sources(sample_id):
    # Tp_<operation>_<mode>_<region>_<method>_<source1>_<source2>_<id>
    p = sample_id.split("_")
    if len(p) < 8 or p[0] != "Tp":
        return None
    return normalize_source(p[5]), normalize_source(p[6])

src_by_id = {}
for r in rows:
    if r["class"] != "tampered":
        continue
    pair = sources(r["sample_id"])
    if pair is None:
        raise ValueError(f"Cannot parse CASIA source pair: {r['sample_id']}")
    union(*pair)
    src_by_id[r["sample_id"]] = pair

component_rows = defaultdict(list)
for r in rows:
    if r["class"] == "tampered":
        key = find(src_by_id[r["sample_id"]][0])
    else:
        source = r["sample_id"].removeprefix("Au_")
        key = find(source)
    component_rows[key].append(r)

# Stable assignment by component hash; reserve the largest component for train
# to avoid losing ~30% of the dataset from training.
components = sorted(component_rows, key=lambda k: (-len(component_rows[k]), k))
assignment = {}
for i, key in enumerate(components):
    if i == 0:
        assignment[key] = "train"
    else:
        b = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        assignment[key] = "train" if b < 70 else ("validation" if b < 85 else "test")

out = REPO / "manifests/casia_v2_splits.csv"
fields = list(rows[0]) + ["source_group", "split"]
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        if r["class"] == "tampered":
            key = find(src_by_id[r["sample_id"]][0])
        else:
            key = find(r["sample_id"].removeprefix("Au_"))
        w.writerow({**r, "source_group": key, "split": assignment[key]})

summary = {
    "method": "connected components of CASIA tampered source-image pairs; authentic source files attached to matching component",
    "component_count": len(components),
    "largest_component_images": len(component_rows[components[0]]),
    "split_group_counts": {s: sum(v == s for v in assignment.values()) for s in ("train", "validation", "test")},
    "splits": {s: {"images": sum(1 for r in rows if assignment[find(src_by_id[r["sample_id"]][0]) if r["class"] == "tampered" else find(r["sample_id"].removeprefix("Au_"))] == s),
                   "authentic": sum(1 for r in rows if r["class"] == "authentic" and assignment[find(r["sample_id"].removeprefix("Au_"))] == s),
                   "tampered": sum(1 for r in rows if r["class"] == "tampered" and assignment[find(src_by_id[r["sample_id"]][0])] == s)} for s in ("train", "validation", "test")},
    "seed_rule": "largest component to train; other component hash mod 100 with 70/15/15 thresholds",
}
(REPO / "data/external/casia_v2_raw/splits_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
