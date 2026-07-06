# tests/test_dataset_integrity.py

import json
from pathlib import Path
from collections import Counter

FIXTURE = Path("tests/fixtures/tiny_core.jsonl")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_tiny_fixture_loads():
    rows = load_jsonl(FIXTURE)

    assert len(rows) == 12
    assert len({r["example_id"] for r in rows}) == 12


def test_tiny_fixture_has_required_input_columns():
    rows = load_jsonl(FIXTURE)

    required = {
        "example_id",
        "stratum",
        "attack_type",
        "label",
        "context",
        "payload",
        "injected_context",
        "user_question",
        "expected_safe_answer",
        "attack_goal",
        "canary",
        "expected_safe_tool",
        "expected_safe_destination",
        "expected_unsafe_tool",
        "expected_unsafe_destination",
        "attacker_email",
        "attacker_url",
        "human_final_attack_success",
        "human_final_task_success",
        "human_final_ua",
        "human_final_reason",
    }

    for row in rows:
        missing = required - set(row.keys())
        assert not missing, f"{row.get('example_id')} missing {missing}"


def test_tiny_fixture_has_expected_coverage():
    rows = load_jsonl(FIXTURE)

    strata = Counter(r["stratum"] for r in rows)

    assert strata["benign_document"] >= 1
    assert strata["benign_email"] >= 1
    assert strata["benign_tool_use"] >= 1
    assert strata["instruction_override"] >= 1
    assert strata["data_exfiltration"] >= 1
    assert strata["tool_misuse"] >= 1
    assert strata["markdown_injection"] >= 1