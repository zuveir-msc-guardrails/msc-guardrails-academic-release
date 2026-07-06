# MSc Guardrails Evaluation Framework

This repository contains the implementation and final artefacts for an MSc dissertation experiment on lightweight guardrails against indirect prompt injection attacks in RAG-style LLM agents.

The project evaluates several guardrail conditions over a frozen synthetic benchmark of 190 examples. The benchmark includes benign tasks and malicious retrieved contexts covering instruction override, data exfiltration, tool misuse, and Markdown/link injection.

## What is included

```text
data/core/                  Frozen benchmark dataset and frozen ID record
src/guardrail_eval/          Main experiment implementation
src/guardrail_eval/conditions/
    c0.py                    Baseline unprotected backend
    c1.py                    Prompt-only hardening
    c2.py                    Heuristic detector
    c3.py                    LLM document-level classifier
    c5a.py                   C3 classifier + sentence/unit LLM sanitisation
    c5b.py                   C3 classifier + deterministic fuzzy payload removal
    c5c.py                   C3 classifier + context-aware unit-removal sanitisation
tests/                       Offline contract/regression tests
scripts/                     Final verification/analysis scripts
results/final/               Final reviewed/scored outputs used for dissertation analysis
docs/                        Reproducibility and scoring notes
```

Development-only scripts, intermediate logs, local virtual environments, `.env` files, and raw dev outputs are intentionally excluded from the academic release.

## Condition summary

| Condition | Meaning |
|---|---|
| C0 | Unprotected baseline agent |
| C1 | Prompt-only backend hardening |
| C2 | Regex/heuristic detector before backend |
| C3 | LLM document-level guardrail classifier |
| C5a | C3 classifier plus sentence/unit-level LLM sanitisation |
| C5b | C3 classifier plus deterministic fuzzy payload removal |
| C5c | C3 classifier plus context-aware LLM unit-removal sanitisation |

For C5 conditions, `guardrail_decision = block` means the document-level classifier detected suspicious retrieved content. It does **not** always mean the final user task was blocked. The final action is recorded in:

```text
pipeline_action
backend_called
backend_context_type
```

The key C5 final actions are:

```text
allow_full_context      Backend called with original retrieved context
sanitised_context       Backend called with cleaned/sanitised context
blocked_fallback        Backend not called; deterministic fallback block
```

## Install

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv mscguardrails
source mscguardrails/bin/activate

pip install -r requirements.txt
```

If your environment uses a different virtual environment name, that is fine. The name `mscguardrails` is only an example.

## Check the project without making API calls

From the repository root:

```bash
python -m compileall src tests scripts -q
PYTHONPATH=src:. pytest -q
PYTHONPATH=src:. python scripts/freeze_ids.py --verify
```

Expected result:

```text
66 passed
All 190 example IDs match the frozen set ✓
Dataset SHA-256 also matches frozen record ✓
Safe to run experiments.
```

The test suite is designed to run offline using fake clients and local mock tools. It should not require an OpenAI API key.

## Verify the frozen dataset

The experiment uses a frozen benchmark in:

```text
data/core/core.jsonl
data/core/frozen_ids.json
```

Run:

```bash
PYTHONPATH=src:. python scripts/freeze_ids.py --verify
```

This checks both:

```text
the ordered set of 190 example IDs
the SHA-256 hash of the frozen dataset
```

Do not modify `data/core/core.jsonl` if you want to reproduce the dissertation results.

## Run a live experiment

Live experiment runs require an OpenAI API key.

Create a local `.env` file in the repository root:

```text
OPENAI_API_KEY=your_key_here
```

Then run a condition directly, for example:

```bash
PYTHONPATH=src:. python src/guardrail_eval/conditions/c0.py
PYTHONPATH=src:. python src/guardrail_eval/conditions/c1.py
PYTHONPATH=src:. python src/guardrail_eval/conditions/c2.py
PYTHONPATH=src:. python src/guardrail_eval/conditions/c3.py
PYTHONPATH=src:. python src/guardrail_eval/conditions/c5a.py
PYTHONPATH=src:. python src/guardrail_eval/conditions/c5b.py
PYTHONPATH=src:. python src/guardrail_eval/conditions/c5c.py
```

Live runs write new CSV and JSONL outputs to:

```text
results/logs/<condition>/
```

These outputs are not the reviewed dissertation results unless they are manually reviewed and scored.

## Tool-use safety

The experiment exposes mock tool definitions to the LLM backend so tool-misuse attacks can be evaluated. These tools do **not** perform real side effects.

The following tool behaviours are mocked locally:

```text
send_email      logs a proposed email call; does not send email
open_url        logs a proposed URL open; does not open or fetch the URL
export_file     logs a proposed export; does not write a file
create_report   logs a proposed report; does not create an external report
```

This allows unsafe proposed tool calls to be captured and scored without sending real emails, opening malicious URLs, or exporting data.

## Results and scoring

Final reviewed outputs are stored under:

```text
results/final/
```

The automatic `auto_*` columns are deterministic helper flags used to support review. They are useful for finding likely canary leaks, suspicious tool calls, expected-answer matches, and similar signals.

The final dissertation metrics should be calculated from the reviewed human fields:

```text
human_final_attack_success
human_final_task_success
human_final_ua
human_final_reason
```

Do not treat `auto_task_success_signal` or `auto_attack_success_signal` as final labels. They are review aids only.

## Reproducibility notes

A clean reproducibility check is:

```bash
python -m compileall src tests scripts -q
PYTHONPATH=src:. pytest -q
PYTHONPATH=src:. python scripts/freeze_ids.py --verify
```

A full live rerun may produce slightly different model outputs, latencies, token counts, and helper flags because it calls a live hosted model. The final dissertation results are based on the reviewed/scored outputs in `results/final/`.

## Suggested examiner workflow

1. Inspect the condition files in `src/guardrail_eval/conditions/`.
2. Run the offline tests with `PYTHONPATH=src:. pytest -q`.
3. Verify the frozen dataset with `scripts/freeze_ids.py --verify`.
4. Inspect `results/final/` for the reviewed condition outputs.
5. Use `human_final_*` fields for final metrics.
6. Treat `auto_*` fields as helper signals, not final labels.

## Current validated release

The refactored academic version was validated with:

```text
66 pytest tests passing
190/190 frozen dataset IDs verified
Dataset SHA-256 matching frozen record
```

Associated Git tag:

```text
v1.0-academic-refactor
```
