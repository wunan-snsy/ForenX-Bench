# ForenX-Bench

An auditable evaluation of frozen image-manipulation detectors across datasets, controlled post-processing, localization annotations, and independent replay.

## v1.0.0 experiment scope

| Dataset | Detection images | Localization images | Protocol |
|---|---:|---:|---|
| Columbia Uncompressed | 363 | 180 | Clean evaluation and JPEG/resize stress tests |
| CASIA v1 corrected | 1,721 | 920 | Clean evaluation and JPEG/resize stress tests |
| COVERAGE corrected | 200 | 91 | Paired copy-move detection and localization |

The headline metrics, stress results, and limitations are in [the aggregate experiment summary](reports/v1.0_experiment_summary.md). The plots and their aggregate source tables are in [`reports/figures/`](reports/figures/). No per-sample identifiers, paths, hashes, masks, predictions, or replay logs are published.

The current public-draft manuscript is available as [LaTeX source](ForenX-Bench.tex) and a [compiled PDF](ForenX-Bench.pdf). Author and affiliation details are intentionally omitted from this public draft.

## Reproduction

This repository is not a one-command downloader. Obtain datasets, upstream implementations, model weights, and any required approvals directly from their providers. Review [Data, model, and rights](DATA_AND_RIGHTS.md) and [Model provenance](MODEL_PROVENANCE.md) first. The original datasets and model weights are not included.

The documented run scripts assume a compatible Python/PyTorch environment, GPU where required, and upstream code/checkpoints at the paths configured by each script. First obtain the relevant data and weights, run the dataset audit to generate a local manifest, then run inference and evaluation. Local manifests and sample-level outputs are git-ignored and remain on the user's machine.

Examples for an already prepared local environment:

```powershell
python scripts/audit_casia_v1.py
python scripts/audit_coverage.py
powershell -ExecutionPolicy Bypass -File scripts/run_casia_v1_full_stress.ps1 -Model both -Gpu 0
powershell -ExecutionPolicy Bypass -File scripts/run_coverage_models.ps1 -Model both -Gpu 0
```

Check the relevant scripts and configs for expected local paths and arguments. A successful rerun requires the exact dataset variants and compatible upstream model artifacts; the aggregate archive alone cannot reconstruct excluded inputs or sample-level records.

## Reproduction boundaries

- No cross-machine reproducibility claim is made: the 180 independent replay attempts ran in separate processes on the same host/software installation.
- Columbia's reference mask has a dataset-specific edge/source-region meaning and is not a complete tampered-area mask.
- Nine COVERAGE reference masks were dimension-mismatched and excluded only from localization; all 200 images remain in detection.
- The work evaluates two frozen models on three existing datasets; it does not introduce a detector, prove causal forensic attribution, or establish legal admissibility.
- CASIA v1 acquisition terms were not preserved, and the MVSS-Net code/checkpoint terms remain unresolved. These are disclosed in the rights statement and limit reuse claims.

## License and citation

The [MIT License](LICENSE) covers original project software code only. It does not grant rights to third-party data, masks, model code or weights; see [DATA_AND_RIGHTS.md](DATA_AND_RIGHTS.md). Cite the benchmark and the original dataset/model publications using [CITATION.cff](CITATION.cff) and the references in the paper.

## Contents

- `scripts/`, `configs/`: evaluation and audit code/configuration
- `reports/`: aggregate reports and figures only
- `sections/`, `ForenX-Bench.tex`, `references.bib`: manuscript source
- `data/SOURCES.csv`, `DATA_AND_RIGHTS.md`: provenance and access/rights limits
- `MODEL_PROVENANCE.md`, `REPRODUCIBILITY.md`: model and replay boundaries
