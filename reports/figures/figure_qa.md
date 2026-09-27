# Manuscript figures: content and quality notes

## Figure roles

| Figure | Type | Claim / question | Evidence |
|---|---|---|---|
| 1. Evaluation workflow | Protocol schematic (not a detector mechanism) | What inputs, frozen models, evaluation paths, evidence records and replay checks were used? | Project evaluation protocol; symbols are schematic, not dataset examples or model outputs. |
| 2. Cross-dataset and manipulation-type results | Quantitative data figure | How do image-level AUCs vary across datasets, and how does CASIA v1 mask F1 vary by operation type? | Archived clean-evaluation JSON; panel A includes dataset-specific bootstrap 95% intervals. |
| 3. Stress sensitivity | Quantitative data figure | How do controlled transforms change AUC and fixed-threshold image decisions? | Archived full-image stress JSON; AUC deltas use clean-condition AUC from archived evaluation summaries. |

## Outputs

Each figure is exported as editable-text PDF and SVG, and as 600 dpi PNG and TIFF. The Python builder is `scripts/make_manuscript_figures.py`. Quantitative source tables are `cross_dataset_comparison_source.csv`, `casia_localization_by_type_source.csv`, and `stress_sensitivity_source.csv` in this directory.

The workflow schematic follows the provided reference's numbered pastel-stage layout. It was redrawn in Python with neutral vector symbols; it does not use generated sample photos that could be mistaken for dataset images. The two data figures are plotted directly from archived result JSON files.

## Checks

- The manuscript compiles with `pdflatex` to a 15-page PDF; figure references resolve and no LaTeX undefined-reference or overfull/underfull warnings were found in the final log.
- Figure source preflight: ready, 0 warnings, 0 failures.
- PDF text-size audit: all three figures pass a 5 pt minimum check.
- Quantitative-panel alignment checks pass at 1.5 pt tolerance. Figure 2 intentionally gives panel A more width to accommodate three dataset estimates and intervals; the reason is stored with its alignment record in `tmp/figure_qa/`.
- Rendered collision audit: Figures 2 and 3 pass with no warnings. Figure 1 has six `text-fill-edge` warnings because its stage numerals are intentionally centered inside filled numbered badges, matching the requested reference style. Visual inspection at manuscript size confirmed the numerals and badge boundaries are clear; there are no text-text or text-stroke failures.
- Manuscript-page renders were visually inspected for Figures 1–3 after compilation.

## Use and submission status

These figures are integrated into the current manuscript draft. Their design follows a general high-end computer-vision / Nature-inspired visual language; no target journal has been selected, so final journal-specific figure specifications and any AI-use policy checks remain to be done before submission.
