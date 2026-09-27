# COVERAGE acquisition and local audit

Obtain the corrected COVERAGE release from the [dataset authors' repository](https://github.com/wenbihan/coverage), which links to the authors' download location and states the data-use terms. Do not rely on mirrors or redistribute the archive. The authors state that the data is copyrighted and may only be used for non-commercial research; cite their ICIP 2016 paper.

After obtaining the archive under the current provider terms, keep it outside version control, extract it to the local path expected by the audit script, and run `python scripts/audit_coverage.py --help` to review its options. The expected release contains original/tampered pairs and masks. The authors corrected some original/tampered labels in the 27 September 2017 update; do not use an earlier package without checking its version.

The audit creates a local manifest with filenames and hashes. That manifest is ignored by Git and must not be published unless the dataset provider confirms that these derived sample-level records may be shared. The public repository contains only aggregate metrics.
