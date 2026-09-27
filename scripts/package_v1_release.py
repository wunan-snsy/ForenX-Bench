#!/usr/bin/env python3
"""Build a public-source archive without datasets or sample-level artifacts."""
import shutil
from pathlib import Path

from run_columbia_blocks_baseline import REPO


ROOT_FILES = (
    ".gitignore", "README.md", "CHANGELOG.md", "CITATION.cff", "LICENSE",
    "DATA_AND_RIGHTS.md", "DATASET_SELECTION.md", "EXPERIMENT_PLAN.md",
    "MODEL_PROVENANCE.md", "REPRODUCIBILITY.md", "ForenX-Bench.tex",
    "ForenX-Bench.pdf",
    "references.bib", "data/SOURCES.csv", "docs/coverage_acquisition.md",
    "reports/v1.0_experiment_summary.md", "reports/v1.0_experiment_summary.json",
)
FIGURE_FILES = (
    "evaluation_workflow.pdf", "cross_dataset_comparison.pdf",
    "stress_sensitivity.pdf", "cross_dataset_comparison_source.csv",
    "stress_sensitivity_source.csv", "casia_localization_by_type_source.csv",
    "figure_qa.md",
)


def copy_file(relative_path: str, stage: Path) -> None:
    source = REPO / relative_path
    if not source.is_file():
        raise FileNotFoundError(f"Required release file is missing: {source}")
    target = stage / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    staging = REPO / "tmp/public-v1.0.0/ForenX-Bench-v1.0.0"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for relative_path in ROOT_FILES:
        copy_file(relative_path, staging)
    for source in sorted((REPO / "sections").glob("*.tex")):
        copy_file(source.relative_to(REPO).as_posix(), staging)
    for source in sorted((REPO / "configs").glob("*.json")):
        copy_file(source.relative_to(REPO).as_posix(), staging)
    for source in sorted((REPO / "scripts").glob("*.py")) + sorted((REPO / "scripts").glob("*.ps1")):
        copy_file(source.relative_to(REPO).as_posix(), staging)
    for name in FIGURE_FILES:
        copy_file(f"reports/figures/{name}", staging)

    dist = REPO / "dist"
    dist.mkdir(exist_ok=True)
    archive_base = dist / "ForenX-Bench-v1.0.0"
    archive = Path(str(archive_base) + ".zip")
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive_base), "zip", staging.parent, staging.name)
    print(archive)


if __name__ == "__main__":
    main()
