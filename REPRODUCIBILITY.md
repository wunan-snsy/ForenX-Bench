# Reproduction boundaries

## Public release contents

The public repository contains original evaluation code, configs, aggregate result tables/figures, experiment summaries, and manuscript sources. It intentionally excludes original datasets, masks, local file-level manifests and hashes, model checkpoints, upstream model source trees, full prediction arrays, evidence records tied to individual samples, and per-sample replay logs. Generated local data/results directories are git-ignored.

## Required assets

Before inference, obtain every dataset directly from an authorized source under current provider terms. The CASIA v1 image bundle used in the study came from a third-party distribution whose exact download URL and license terms were not preserved; users must independently verify an authorized source. Obtain model source and checkpoint artifacts separately. See [DATA_AND_RIGHTS.md](DATA_AND_RIGHTS.md), [data/SOURCES.csv](data/SOURCES.csv), and [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

The scripts expect assets at project-specific local paths. Run the relevant `audit_*.py` script to create a local file-level manifest; these manifests are intentionally not versioned. Review and set paths in the configs and scripts to match your approved copies. The run commands in README are examples, not a turnkey clean-clone installation: a compatible runtime, dependencies, upstream code, weights, and sometimes a GPU are required.

## Replay result and limits

The CASIA v1 independent replay selected 30 authentic, 30 copy-move, and 30 splicing cases per model (90 per model; seed 20260929). The 180 attempts reproduced both score and localization map within absolute tolerance `1e-5` after local source-hash checks. This replay ran in a separate process on the same host and software installation. It supports deterministic same-environment replay for the sampled cases; it does not establish cross-machine equivalence or correctness of the predictions. Per-sample replay records are not included in the public release because publication rights for derived sample-level records were not verified.

The project does not ship a complete dependency lockfile or sealed container image. The aggregate archive cannot reconstruct excluded inputs, model artifacts, or sample-level records. See experiment reports for dataset-specific denominators, annotation caveats, and stress conditions.
