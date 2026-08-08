# Test Suite

This directory contains offline unit, contract, and regression tests for the
guardrail evaluation framework.

## Use of deterministic fake LLM clients

The automated tests use deterministic fake LLM clients so that experiment
routing, sanitisation, tool-call handling, output schemas, helper flags, and
cost calculations can be tested reproducibly without making external API
calls.

The fake clients simulate only the model behaviours required by individual
tests, such as:

- allowing or blocking retrieved content;
- returning a controlled backend response;
- proposing a mock tool call;
- selecting suspicious context units for removal; and
- simulating classifier or sanitiser failures.

These fake clients were used **only for software testing**. They were not used
to generate any experimental results reported in the dissertation.

The dissertation experiments were run using the live model clients implemented
in the main `src/guardrail_eval/` package. The reviewed/scored outputs used for
the dissertation analysis are stored under `results/final/`.

## Main test groups

The condition contract tests verify the expected pipeline behaviour for each
experimental condition:

- `test_c0_contract.py` — unprotected baseline
- `test_c1_contract.py` — prompt-only hardening
- `test_c2_contract.py` — heuristic detector
- `test_c3_contract.py` — LLM classifier and blocking
- `test_c5a_contract.py` — sentence/unit LLM sanitisation
- `test_c5b_contract.py` — deterministic fuzzy payload removal
- `test_c5c_contract.py` — context-aware unit-removal sanitisation

Additional tests cover dataset integrity, attack-success helper signals,
tool-call handling, text utilities, heuristic detection, and cost calculation.

## Running the tests

From the repository root:

```bash
PYTHONPATH=src:. pytest -q