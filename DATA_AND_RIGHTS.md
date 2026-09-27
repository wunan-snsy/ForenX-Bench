# Data, model, and rights statement

This repository contains evaluation software and aggregate research results. It does not distribute the original benchmark images, masks, model source trees, or model checkpoints.

## Dataset access and limits

| Resource | Access route | What this project distributes | Known terms and project limits |
|---|---|---|---|
| Columbia Uncompressed Image Splicing Detection Evaluation Dataset | [Columbia DVMM dataset page](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/); the page requires a request form | Aggregate metrics only; no images, masks, sample paths, or per-image predictions | Obtain access from Columbia and provide the requested credit. The page does not grant blanket redistribution rights. |
| CASIA v1 images | The original CASIA owners' server is reported unavailable. The corrected bundle used for this study came from a third-party distribution; the exact acquisition URL and terms were not preserved in the experiment archive. | Aggregate metrics only; no images, masks, sample manifest, or per-image predictions | The dataset and corrected copy have no project-level redistribution permission. Users must obtain a lawful copy independently and verify the applicable terms. |
| CASIA v1 ground-truth masks | [namtpham/casia1groundtruth](https://github.com/namtpham/casia1groundtruth) identifies its ground truth and cites the CASIA and Pham et al. works | Aggregate metrics only; no masks, sample manifest, or per-image predictions | The upstream repository provides access and citation information but no license granting this project redistribution rights. |
| COVERAGE | [Dataset authors' repository](https://github.com/wenbihan/coverage) | Aggregate metrics only; no images, masks, per-image paths, hashes, or predictions | The authors state that the data is copyrighted and may be used only for non-commercial research; cite their paper. These terms do not grant redistribution rights. |
| Columbia 2004 image-block set | [Columbia DVMM dataset page](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm) | No sample-level data or results from this pilot are included in the public release | The accompanying agreement limits use to research and prohibits redistribution. Obtain and follow the original agreement. |

The table describes the routes and terms observed by the project; it is not legal advice and does not replace current provider terms or written permission. Dataset access, publication of results, and redistribution are separate questions. Before use, check current terms with the provider and your institution.

## Model and checkpoint access

- [TruFor source repository](https://github.com/grip-unina/TruFor): the upstream source includes a restrictive license for informational/nonprofit use. This project does not redistribute that code. Permission for the specific checkpoint used in the experiments was not separately preserved; obtain it from the upstream authors and follow any checkpoint-specific terms.
- [MVSS-Net source repository](https://github.com/dong03/MVSS-Net): no license was found in the inspected upstream repository snapshot, and the exact snapshot commit was not recorded. The checkpoint-specific terms were not preserved. No upstream code or weights are redistributed here. Absence of a license must not be treated as permission.

The model identifiers and hashes recorded in project provenance are for research traceability only. They do not authorize access, use, modification, or redistribution.

## What the repository license covers

The root [MIT License](LICENSE) applies to original software code authored for this project. It does not grant rights to third-party datasets, masks, sample identifiers, model implementations, weights, or other assets. Original manuscript text, figures, and aggregate result tables are not relicensed by the code license unless a separate notice says otherwise. Third-party papers and citations retain their respective rights.

Public aggregate tables describe the experiments conducted by the authors. Their publication does not create a license to the underlying data, models, or checkpoints, and downstream users remain responsible for respecting upstream terms.

## Public release policy

The public repository excludes original images and masks, file-level manifests and hashes, model weights, upstream model source trees, full prediction arrays, sample-level evidence records, and replay logs. Local reruns may create such files under ignored paths; those generated files are not part of this public release. The public release retains aggregate metrics, figures, protocol/configuration files, and code needed to audit or adapt the pipeline.

If you believe an item in this repository infringes a right or violates a dataset condition, contact the repository owner with the file path and the relevant provider terms so it can be reviewed promptly.
