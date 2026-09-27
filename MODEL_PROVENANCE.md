# Model and software provenance

This file records the frozen models used in the v1.0.0 experiments. Upstream code, checkpoints, and local per-sample outputs are not included in the public repository. Dataset and model permissions are separate; see [DATA_AND_RIGHTS.md](DATA_AND_RIGHTS.md).

| Model | Upstream source | Recorded code revision | Weight identifier | Training provenance | Public-release status |
|---|---|---|---|---|---|
| TruFor | [GRIP-UNINA/TruFor](https://github.com/grip-unina/TruFor) | `ae54475df6f41a491d7615100feb19263dec13f7` | MD5 `7bee48f3476c75616c3c5721ab256ff8` | Author-released inference weights; this project did not train them | Code and weights excluded. The inspected code license restricts use; checkpoint-specific permission was not retained. |
| MVSS-Net | [dong03/MVSS-Net](https://github.com/dong03/MVSS-Net) | Not captured; local snapshot had no Git metadata | SHA-256 `080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50` | Author-released CASIAv2 checkpoint; upstream README describes CASIAv2 training | Code and weights excluded. Exact source revision and applicable licenses remain unresolved. |

The identifiers and hashes support provenance of the local experiment artifacts; they do not grant access, use, modification, or redistribution rights. Do not interpret the project MIT License as applying to either upstream model or checkpoint.
