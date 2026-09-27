#!/usr/bin/env python3
"""Build public-safe v1.0 experiment summaries from completed audit files."""
import json
from pathlib import Path
from run_columbia_blocks_baseline import REPO


def read(path):
    with (REPO / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    casia = read("results/casia_v1_evaluation/summary.json")
    casia_stress = read("results/casia_v1_full_stress_audit/summary.json")
    coverage = read("results/coverage_evaluation/summary.json")
    columbia = read("results/trufor_columbia_detection/summary.json")
    columbia_boundary = read("results/trufor_columbia_boundary/summary.json")
    mvss_columbia = read("results/mvss_columbia_evaluation/summary.json")
    replay = read("results/casia_v1_replay/summary_v1.1.json")
    rows = []
    for model, result in (("TruFor", columbia), ("MVSS-Net", mvss_columbia)):
        detection = result["metrics"] if model == "TruFor" else result["detection"]
        rows.append({"dataset": "Columbia Uncompressed", "condition": "clean", "model": model,
                     "n_detection": detection["n"], "n_localization": 180 if model == "TruFor" else 180,
                     "auc": detection["roc_auc"], "detection_f1": detection.get("f1_spliced"),
                     "localization_f1": columbia_boundary["macro"]["f1"] if model == "TruFor" else result["spatial_spliced_macro"]["f1"]})
    for model, result in casia["models"].items():
        detection = result["detection"]
        rows.append({"dataset": "CASIA v1 corrected", "condition": "clean", "model": "TruFor" if model == "trufor" else "MVSS-Net",
                     "n_detection": result["n_detection"], "n_localization": result["n_localization"], "auc": detection["roc_auc"],
                     "detection_f1": detection["f1_spliced"], "localization_f1": result["localization_macro"]["f1"]})
    for model, result in coverage["models"].items():
        detection = result["detection"]
        rows.append({"dataset": "COVERAGE corrected", "condition": "clean", "model": "TruFor" if model == "trufor" else "MVSS-Net",
                     "n_detection": result["n_detection"], "n_localization": result["n_localization"], "auc": detection["roc_auc"],
                     "detection_f1": detection["f1_spliced"], "localization_f1": result["localization_macro"]["f1"]})
    stress_rows = []
    for model, result in casia_stress["models"].items():
        for condition, values in result["by_condition"].items():
            stress_rows.append({"dataset": "CASIA v1 corrected", "model": "TruFor" if model == "trufor" else "MVSS-Net", "condition": condition,
                                "n_detection": values["n_detection"], "n_localization": values["n_localization"],
                                "auc": values["detection"]["roc_auc"], "detection_f1": values["detection"]["f1_spliced"],
                                "localization_f1": values["localization_macro"]["f1"], "mean_score_delta": values["mean_score_delta"],
                                "image_prediction_flip_rate": values["image_prediction_flip_rate"]})
    replay_summary = {model: {"attempted": values["attempted"], "successful_outputs": values["successful_outputs"],
                              "replay_success_rate": values["RSR"], "replay_success_rate_wilson95": values["RSR_wilson95"],
                              "agreement_with_original": values["agreement_within_tolerance"], "evidence_agreement_rate": values["EGA"],
                              "evidence_agreement_rate_wilson95": values["EGA_wilson95"]}
                      for model, values in replay["by_model"].items()}
    output = {"release": "v1.0.0", "status": "experiment_aggregate_complete", "results": rows, "casia_stress": stress_rows,
              "casia_v1_independent_replay": {"schema_version": replay["schema_version"], "selection": replay["selection"],
                                              "tolerance": replay["tolerance"], "models": replay_summary,
                                              "by_stratum": replay["by_stratum"]},
              "release_readiness": {"project_license": "MIT for original project software code only", "third_party_rights": "datasets, model source, and checkpoints are excluded and remain governed by upstream terms; see DATA_AND_RIGHTS.md", "sample_level_materials": "excluded because publication rights were not verified"},
              "coverage_data_quality": {"n_detection": 200, "n_localization": 91, "localization_excluded_mask_dimension_mismatch": 9},
              "provenance": {"source": "project experiment reports and local result summaries; sample-level source files are excluded from the public release"}}
    release = REPO / "release/v1.0.0"; release.mkdir(parents=True, exist_ok=True)
    summary_json = json.dumps(output, indent=2) + "\n"
    (release / "experiment_summary.json").write_text(summary_json, encoding="utf-8")
    (REPO / "reports/v1.0_experiment_summary.json").write_text(summary_json, encoding="utf-8")
    def metric(value): return "NA" if value is None else f"{value:.4f}"
    lines = ["# ForenX-Bench v1.0.0: Experiment Summary", "", "## Scope", "", "Frozen TruFor and MVSS-Net inference on Columbia Uncompressed, CASIA v1 corrected, and COVERAGE corrected. Image-level threshold is fixed at 0.5. Localization results use only samples with valid, dimension-aligned reference masks.", "", "## Clean-image results", "", "| Dataset | Model | Detection n | Localization n | ROC-AUC | Detection F1 | Localization macro F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['dataset']} | {r['model']} | {r['n_detection']} | {r['n_localization']} | {metric(r['auc'])} | {metric(r['detection_f1'])} | {metric(r['localization_f1'])} |" for r in rows]
    lines += ["", "## CASIA v1 pressure test", "", "| Model | Condition | ROC-AUC | Detection F1 | Localization macro F1 | Mean score change | Prediction flip rate |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['condition']} | {metric(r['auc'])} | {metric(r['detection_f1'])} | {metric(r['localization_f1'])} | {metric(r['mean_score_delta'])} | {metric(r['image_prediction_flip_rate'])} |" for r in stress_rows]
    lines += ["", "## CASIA v1 independent replay", "", "A seeded, stratified sample of 30 authentic, 30 copy-move, and 30 splicing images was replayed independently for each frozen model (90 attempts per model; seed 20260929). All sampled source hashes matched local manifests. At score and localization-map absolute-error tolerances of 1e-5, both TruFor and MVSS-Net reproduced all 90 sampled outputs (RSR=90/90; EGA=90/90 per model; Wilson 95% interval 0.959–1.000). Replay ran in a separate process on the same host and software environment; this does not establish cross-machine equivalence. Sample identifiers and replay records are omitted from the public release because permission to publish them was not verified.", "", "## Data-quality disclosure", "", "COVERAGE contributes all 200 images to detection. Nine tampered images with author-supplied masks whose dimensions do not match their images are excluded from localization metrics; therefore COVERAGE localization uses 91 samples.", "", "## Reproduction and evidence", "", "The repository includes code, configurations, aggregate results, and local audit procedures. Original inputs, model weights, per-file manifests and hashes, and sample-level evidence/replay records are excluded. Obtain data and model assets separately under their providers' current terms, generate manifests locally, and follow the documented run protocol.", "", "## License and third-party rights", "", "The MIT License applies to original project software code only. It does not license datasets, annotations, model implementations, checkpoints, manuscript text, or third-party assets. CASIA v1 image acquisition terms and permissions for the specific model checkpoints remain unresolved; these assets are not redistributed. See `DATA_AND_RIGHTS.md`, `data/SOURCES.csv`, and `MODEL_PROVENANCE.md`.", ""]
    report = REPO / "reports/v1.0_experiment_summary.md"; report.write_text("\n".join(lines), encoding="utf-8")
    (release / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"release_json": str(release / "experiment_summary.json"), "report": str(report), "n_clean_rows": len(rows), "n_stress_rows": len(stress_rows)}, indent=2))


if __name__ == "__main__": main()
