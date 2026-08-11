# Dataset Integrity and Reproducibility

## Purpose

This document explains how to verify that the supplied evaluation datasets are the same frozen artefacts used by the academic release.

The primary benchmark contains 190 examples in:

```text
data/core/core.jsonl
```

The supplementary obfuscation stress test contains 24 examples in:

```text
data/obfuscation/obfuscated_malicious.jsonl
```

---

## 1. Primary benchmark freeze

The supplied freeze artefact is:

```text
data/core/frozen_ids.json
```

The academic release currently verifies:

```text
Frozen total: 190
Current total: 190
SHA-256:
1dbcac787550f78dc686718b6603d36117b7967435a2af45d3669efd29a644c1
```

The freeze artefact records the stable example IDs and the SHA-256 of the core JSONL file.

Two checks are therefore performed:

1. **ID stability** — no example has been added, removed or renamed.
2. **Content stability** — the SHA-256 detects any change to the bytes of `core.jsonl`, including changes to payloads, questions, expected answers or metadata.

---

## 2. Verify the primary benchmark

From the repository root:

```bash
PYTHONPATH=src:. python scripts/freeze_ids.py --verify
```

A valid release should report:

```text
Frozen total: 190
Current total: 190
All 190 example IDs match the frozen set ✓
Dataset SHA-256 also matches frozen record ✓
Safe to run experiments.
```

If the IDs or SHA-256 differ, do not treat the modified dataset as the supplied frozen benchmark.

---

## 3. Verify the obfuscation dataset

The obfuscation dataset has its own checksum file:

```text
data/obfuscation/obfuscated_malicious.jsonl.sha256
```

On macOS:

```bash
shasum -a 256 -c data/obfuscation/obfuscated_malicious.jsonl.sha256
```

Expected result:

```text
data/obfuscation/obfuscated_malicious.jsonl: OK
```

The supplied 24-row obfuscation dataset should be used directly for reproduction. The dataset-construction script is retained for provenance/regeneration purposes but is not required to reproduce the reported stress-test runs.

---

## 4. Offline software validation

The release includes tests that can be run without API credentials.

Recommended commands:

```bash
python -m compileall src tests scripts experiments_obfuscation -q

PYTHONPATH=src:. python -m pytest -q

PYTHONPATH=src:. python scripts/freeze_ids.py --verify

shasum -a 256 -c data/obfuscation/obfuscated_malicious.jsonl.sha256
```

For the validated academic release, the test suite contains 66 tests.

---

## 5. Live reproduction

Live condition runs require the appropriate API credential supplied locally by the examiner/user.

No API credential should be included in the release archive.

If no credential is present, a live runner should stop with an explicit missing-key error rather than silently using another credential.

Fresh live runs write new runtime logs. These new logs are reproduction artefacts and do not replace the reviewed/scored final dissertation outputs supplied in the final-results directories.

---

## 6. Raw versus authoritative results

Keep these concepts separate:

- **Frozen dataset** — the input examples used across conditions.
- **Fresh runtime logs** — outputs from a new reproduction run.
- **Reviewed/scored final results** — the authoritative outputs used to calculate dissertation metrics.
- **Helper flags** — automated review aids, not final human scores.

A fresh rerun can legitimately differ in latency, token usage, cost and individual model outputs because hosted model behaviour and infrastructure can vary over time. The supplied final reviewed results remain the evidence for the reported dissertation metrics.

---

## 7. No real external tool side effects

The experimental tool environment is mocked. Tool calls are recorded and simulated for evaluation.

The reproduction package does not need access to real email, cloud-drive or external-action services to reproduce the experimental tool-call behaviour.

---

## 8. Quick check

A minimal offline integrity check is:

```bash
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:. python scripts/freeze_ids.py --verify
shasum -a 256 -c data/obfuscation/obfuscated_malicious.jsonl.sha256
```

A successful run establishes that:

- the code passes its test suite;
- the primary benchmark matches the frozen 190-example artefact;
- the supplementary obfuscation dataset matches its stored SHA-256.
