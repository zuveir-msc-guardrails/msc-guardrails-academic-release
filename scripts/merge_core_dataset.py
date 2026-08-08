"""
merge_core_dataset.py
---------------------
Merges all completed dataset strata into the canonical core dataset.

Input files:
    data/agentdojo_curated/agentdojo_tool_examples.jsonl   (45 examples)
    data/bipia/bipia_examples.jsonl                        (90 examples)
    data/constructed/full_benign_emails_markdown_injection.jsonl             (55 examples)

Output files:
    data/core/core.jsonl
    data/core/core.csv

Canonical schema
----------------
Every example in the merged dataset has exactly these fields in this order.
Missing fields from any source are filled with their default value.

    example_id                  str   — stable unique ID across all strata
    split                       str   — "core" for all examples in this run
    stratum                     str   — instruction_override | data_exfiltration |
                                        tool_misuse | benign_document |
                                        benign_tool_use | benign_email |
                                        markdown_injection
    attack_type                 str   — naive | ignore | fake_completion |
                                        escape_separation | none
    label                       str   — malicious | benign
    source                      str   — BIPIA | AgentDojo | Constructed
    source_suite                str   — workspace (AgentDojo) | constructed (Constructed) | "" (BIPIA)
    context_source              str   — email | table | cloud_drive_file |
                                        calendar_event | markdown_document
    context_split               str   — test | train | workspace | constructed | ""
    context_index               str   — normalised to string in all cases
    payload_category            str   — attack category or "none"
    payload_index               str   — normalised to string
    source_task_id              str   — AgentDojo task ID or ""
    source_vector_key           str   — AgentDojo vector key or ""
    context                     str   — clean retrieved context text
    payload                     str   — injection payload; "" for benign
    injected_context            str   — context with payload embedded
    user_question               str   — the legitimate user task/question
    expected_safe_answer        str   — correct answer if agent is not compromised
    attack_goal                 str   — attacker objective; "" for benign
    canary                      str   — CANARY-DE-NNN for DE; "" otherwise
    synthetic_private_context   str   — private system context for DE; "" otherwise
    expected_safe_tool          str   — correct tool to call (benign tool-use)
    expected_safe_destination   str   — correct tool destination (benign tool-use)
    expected_safe_tool_arguments dict — expected tool arguments (benign tool-use)
    expected_unsafe_tool        str   — tool attacker wants misused; "" for benign
    expected_unsafe_destination str   — attacker target; "" for benign
    attacker_email              str   — for send_email attacks; "" otherwise
    attacker_url                str   — for open_url / URL-reproduction attacks; "" otherwise
    selection_notes             str   — provenance note from builder; "" if absent
    llm_judge_attack_success    str   — blank until judge pass
    llm_judge_task_success      str   — blank until judge pass
    llm_judge_ua                str   — blank until judge pass
    llm_judge_reason            str   — blank until judge pass
    human_final_attack_success  str   — blank until human scoring pass
    human_final_task_success    str   — blank until human scoring pass
    human_final_ua              str   — blank until human scoring pass
    human_final_reason          str   — blank until human scoring pass
    needs_human_review          bool  — True until scoring pass complete

Run from project root:
    python scripts/merge_core_dataset.py

Use --dry-run to validate without writing output:
    python scripts/merge_core_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ── Canonical field order ─────────────────────────────────────────────────────

CANONICAL_FIELDS = [
    "example_id",
    "split",
    "stratum",
    "attack_type",
    "label",
    "source",
    "source_suite",
    "context_source",
    "context_split",
    "context_index",
    "payload_category",
    "payload_index",
    "source_task_id",
    "source_vector_key",
    "context",
    "payload",
    "injected_context",
    "user_question",
    "expected_safe_answer",
    "attack_goal",
    "canary",
    "synthetic_private_context",
    "expected_safe_tool",
    "expected_safe_destination",
    "expected_safe_tool_arguments",
    "expected_unsafe_tool",
    "expected_unsafe_destination",
    "attacker_email",
    "attacker_url",
    "selection_notes",
    "llm_judge_attack_success",
    "llm_judge_task_success",
    "llm_judge_ua",
    "llm_judge_reason",
    "human_final_attack_success",
    "human_final_task_success",
    "human_final_ua",
    "human_final_reason",
    "needs_human_review",
]

# Default value for each field when missing from a source file.
FIELD_DEFAULTS: dict[str, Any] = {
    "example_id":                   "",
    "split":                        "core",
    "stratum":                      "",
    "attack_type":                  "",
    "label":                        "",
    "source":                       "",
    "source_suite":                 "",
    "context_source":               "",
    "context_split":                "",
    "context_index":                "",
    "payload_category":             "",
    "payload_index":                "",
    "source_task_id":               "",
    "source_vector_key":            "",
    "context":                      "",
    "payload":                      "",
    "injected_context":             "",
    "user_question":                "",
    "expected_safe_answer":         "",
    "attack_goal":                  "",
    "canary":                       "",
    "synthetic_private_context":    "",
    "expected_safe_tool":           "",
    "expected_safe_destination":    "",
    "expected_safe_tool_arguments": {},
    "expected_unsafe_tool":         "",
    "expected_unsafe_destination":  "",
    "attacker_email":               "",
    "attacker_url":                 "",
    "selection_notes":              "",
    "llm_judge_attack_success":     "",
    "llm_judge_task_success":       "",
    "llm_judge_ua":                 "",
    "llm_judge_reason":             "",
    "human_final_attack_success":   "",
    "human_final_task_success":     "",
    "human_final_ua":               "",
    "human_final_reason":           "",
    "needs_human_review":           True,
}

# Source-specific defaults for fields that may be blank in source builders.
SOURCE_CONTEXT_SPLIT_DEFAULTS = {
    "AgentDojo":    "workspace",
    "BIPIA":        "",             # already set per example where available
    "Constructed":  "constructed",
}

SOURCE_SUITE_DEFAULTS = {
    "AgentDojo":    "workspace",
    "BIPIA":        "",
    "Constructed":  "constructed",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            flat = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else ("" if value is None else str(value))
                )
                for key, value in record.items()
            }
            writer.writerow(flat)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(example: dict[str, Any]) -> dict[str, Any]:
    """
    Project any source example onto the canonical schema.

    Rules applied in order:
    1. Fill missing fields with FIELD_DEFAULTS.
    2. Normalise context_index to str.
    3. Normalise payload_index to str.
    4. Normalise payload None → "".
    5. Normalise attack_goal None → "".
    6. Set context_split default based on source if blank.
    7. Set source_suite default based on source if blank.
    8. Return only canonical fields in canonical order.
    """
    out: dict[str, Any] = {}

    for field in CANONICAL_FIELDS:
        if field in example:
            out[field] = example[field]
        else:
            out[field] = copy.deepcopy(FIELD_DEFAULTS[field])

    # Normalise types.
    out["context_index"] = str(out["context_index"]) if out["context_index"] is not None else ""
    out["payload_index"] = str(out["payload_index"]) if out["payload_index"] is not None else ""
    out["payload"] = out["payload"] if out["payload"] is not None else ""
    out["attack_goal"] = out["attack_goal"] if out["attack_goal"] is not None else ""

    # Set source-specific defaults.
    if not out["context_split"]:
        out["context_split"] = SOURCE_CONTEXT_SPLIT_DEFAULTS.get(out["source"], "")
    if not out["source_suite"]:
        out["source_suite"] = SOURCE_SUITE_DEFAULTS.get(out["source"], "")

    # Ensure expected_safe_tool_arguments is always a dict.
    if not isinstance(out["expected_safe_tool_arguments"], dict):
        out["expected_safe_tool_arguments"] = {}

    return out


# ── Validation ────────────────────────────────────────────────────────────────

# Expected counts per source in the final 190-example core dataset.
EXPECTED_COUNTS: dict[str, int] = {
    "AgentDojo":    45,
    "BIPIA":        90,
    "Constructed":  55,
}

# Expected stratum counts in the final merged dataset.
EXPECTED_STRATUM_COUNTS: dict[str, int] = {
    "tool_misuse":            20,
    "benign_tool_use":        25,
    "instruction_override":   25,
    "data_exfiltration":      25,
    "benign_document":        40,
    "markdown_injection":     20,
    "benign_email":           35,
}

VALID_STRATA = set(EXPECTED_STRATUM_COUNTS.keys())
VALID_ATTACK_TYPES = {"naive", "ignore", "fake_completion", "escape_separation", "none"}
VALID_LABELS = {"malicious", "benign"}

MARKDOWN_URL_CATEGORIES = {
    "external_url_link",
    "image_external_src",
    "encoded_url",
    "hidden_markdown_link",
}


def validate(examples: list[dict[str, Any]], sources: dict[str, int]) -> None:
    errors: list[str] = []

    # Total count.
    total = sum(EXPECTED_COUNTS.values())
    if len(examples) != total:
        errors.append(f"Total count: expected {total}, got {len(examples)}")

    # Per-source counts.
    source_counts = Counter(e["source"] for e in examples)
    for source, expected in EXPECTED_COUNTS.items():
        actual = source_counts.get(source, 0)
        if actual != expected:
            errors.append(f"Source '{source}': expected {expected}, got {actual}")

    unexpected_sources = sorted(set(source_counts) - set(EXPECTED_COUNTS))
    if unexpected_sources:
        errors.append(f"Unexpected sources: {unexpected_sources}")

    # The loaded raw-file source counts should match the normalised source counts.
    if source_counts != Counter(sources):
        errors.append(
            f"Normalised source counts differ from raw source counts: "
            f"normalised={dict(source_counts)}, raw={sources}"
        )

    # Per-stratum counts.
    stratum_counts = Counter(e["stratum"] for e in examples)
    for stratum, expected in EXPECTED_STRATUM_COUNTS.items():
        actual = stratum_counts.get(stratum, 0)
        if actual != expected:
            errors.append(f"Stratum '{stratum}': expected {expected}, got {actual}")

    unexpected_strata = sorted(set(stratum_counts) - VALID_STRATA)
    if unexpected_strata:
        errors.append(f"Unexpected strata: {unexpected_strata}")

    # No duplicate IDs.
    ids = [e["example_id"] for e in examples]
    dups = sorted({example_id for example_id in ids if ids.count(example_id) > 1})
    if dups:
        errors.append(f"Duplicate example_ids: {dups}")

    # All IDs non-empty.
    empty_ids = [i for i, e in enumerate(examples) if not e.get("example_id")]
    if empty_ids:
        errors.append(f"Empty example_id at positions: {empty_ids}")

    # All fields present in every example.
    for e in examples:
        missing = [field for field in CANONICAL_FIELDS if field not in e]
        if missing:
            errors.append(f"{e.get('example_id', '?')}: missing fields {missing}")

    # Valid attack_type, label, and stratum values.
    bad_types = [e["example_id"] for e in examples if e["attack_type"] not in VALID_ATTACK_TYPES]
    if bad_types:
        errors.append(f"Invalid attack_type in: {bad_types}")

    bad_labels = [e["example_id"] for e in examples if e["label"] not in VALID_LABELS]
    if bad_labels:
        errors.append(f"Invalid label in: {bad_labels}")

    bad_strata = [e["example_id"] for e in examples if e["stratum"] not in VALID_STRATA]
    if bad_strata:
        errors.append(f"Invalid stratum in: {bad_strata}")

    # All examples have split=core.
    wrong_split = [e["example_id"] for e in examples if e.get("split") != "core"]
    if wrong_split:
        errors.append(f"Wrong split value in: {wrong_split}")

    # Malicious examples have non-empty attack_goal.
    missing_goal = [
        e["example_id"] for e in examples
        if e["label"] == "malicious" and not e.get("attack_goal")
    ]
    if missing_goal:
        errors.append(f"Malicious examples missing attack_goal: {missing_goal}")

    # All examples have expected_safe_answer.
    missing_answer = [e["example_id"] for e in examples if not e.get("expected_safe_answer")]
    if missing_answer:
        errors.append(f"Missing expected_safe_answer: {missing_answer}")

    # Benign examples have empty payload after normalisation.
    benign_with_payload = [
        e["example_id"] for e in examples
        if e["label"] == "benign" and e.get("payload")
    ]
    if benign_with_payload:
        errors.append(f"Benign examples unexpectedly have payloads: {benign_with_payload}")

    # DE canary checks.
    de_examples = [e for e in examples if e["stratum"] == "data_exfiltration"]
    missing_canary = [e["example_id"] for e in de_examples if not e.get("canary")]
    if missing_canary:
        errors.append(f"DE examples missing canary: {missing_canary}")

    for e in de_examples:
        canary = e.get("canary", "")
        if canary and canary not in e.get("synthetic_private_context", ""):
            errors.append(f"{e['example_id']}: canary missing from synthetic_private_context")
        if canary and canary in e.get("context", ""):
            errors.append(f"{e['example_id']}: canary appears in retrieved context")

    # Markdown URL-reproduction metadata checks.
    markdown_url_examples = [
        e for e in examples
        if e["stratum"] == "markdown_injection"
        and e["payload_category"] in MARKDOWN_URL_CATEGORIES
    ]
    url_bad_destination = [
        e["example_id"] for e in markdown_url_examples
        if not e.get("attacker_url")
        or e.get("expected_unsafe_destination") != e.get("attacker_url")
    ]
    if url_bad_destination:
        errors.append(
            "Markdown URL examples require attacker_url and "
            f"expected_unsafe_destination == attacker_url: {url_bad_destination}"
        )

    # context_index is always a string.
    bad_index_type = [
        e["example_id"] for e in examples
        if not isinstance(e.get("context_index"), str)
    ]
    if bad_index_type:
        errors.append(f"Non-string context_index in: {bad_index_type}")

    if errors:
        print(f"\n  {len(errors)} validation error(s):")
        for error in errors:
            print(f"    ERROR: {error}")
        raise AssertionError("Validation failed. Fix errors before writing output.")


# ── Main ──────────────────────────────────────────────────────────────────────

INPUT_FILES = [
    Path("data/agentdojo_curated/agentdojo_tool_examples.jsonl"),
    Path("data/bipia/bipia_examples.jsonl"),
    Path("data/constructed/full_benign_emails_markdown_injection.jsonl"),
]

OUTPUT_JSONL = Path("data/core/core.jsonl")
OUTPUT_CSV = Path("data/core/core.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing output files.",
    )
    args = parser.parse_args()

    print("Loading source files...")
    raw_examples: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    for path in INPUT_FILES:
        records = load_jsonl(path)
        sources = Counter(record.get("source", "unknown") for record in records)
        print(f"  {path.name}: {len(records)} records — {dict(sources)}")
        raw_examples.extend(records)
        for src, count in sources.items():
            source_counts[src] = source_counts.get(src, 0) + count

    print(f"\nTotal raw examples: {len(raw_examples)}")

    print("\nNormalising to canonical schema...")
    examples = [normalise(example) for example in raw_examples]

    # Sort by example_id for deterministic output.
    examples = sorted(examples, key=lambda e: e["example_id"])

    print("\nValidating...")
    validate(examples, source_counts)
    print("  All validation checks passed ✓")

    print("\nMerged dataset summary:")
    stratum_counts = Counter(e["stratum"] for e in examples)
    for stratum in sorted(EXPECTED_STRATUM_COUNTS):
        count = stratum_counts.get(stratum, 0)
        print(f"  {stratum:<30} {count}")
    print(f"  {'TOTAL':<30} {len(examples)}")

    print("\nField normalisation applied:")
    for source in sorted(EXPECTED_COUNTS):
        source_examples = [e for e in examples if e["source"] == source]
        print(
            f"  {source:<12} context_split values: "
            f"{dict(Counter(e['context_split'] for e in source_examples))}"
        )

    if args.dry_run:
        print("\n--dry-run: skipping output write.")
        return

    print("\nWriting output...")
    write_jsonl(OUTPUT_JSONL, examples)
    write_csv(OUTPUT_CSV, examples)
    print(f"  Written: {OUTPUT_JSONL} ({len(examples)} records)")
    print(f"  Written: {OUTPUT_CSV}")
    print("\nNext step: freeze example IDs.")
    print("  python scripts/freeze_example_ids.py")


if __name__ == "__main__":
    main()
