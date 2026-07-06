#!/usr/bin/env python3
"""
summarise_guardrail_run.py
--------------------------
Create a saved run-summary file from a C2/C3 raw experiment CSV log.

Purpose:
    The experiment runners print useful pipeline summaries to the terminal, but
    that terminal output is not automatically persisted unless the command is
    piped through tee. This script recomputes the same summary from the raw CSV
    log and writes it to a text file.

Typical use for C3:
    python scripts/summarise_guardrail_run.py \
      --log results/logs/c0/c0_20260605_124503.csv

Optional explicit output path:
    python scripts/summarise_guardrail_run.py \
      --log results/logs/c3/c3_20260608_123942.csv \
      --out results/logs/c3/c3_20260608_123942_summary.txt

Works with raw C2/C3 logs that contain:
    - label
    - guardrail_decision
    - guardrail_error
    - auto_* helper fields
    - latency/token/cost fields

It also tolerates enriched review files where guardrail fields are prefixed
with log_, e.g. log_guardrail_decision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HELPER_FIELDS = [
    ("Canary leaks detected", "auto_canary_leaked"),
    ("Tool calls proposed", "auto_tool_call_proposed"),
    ("Attacker destination matches", "auto_attacker_destination_matched"),
    ("Expected answer string matches", "auto_expected_answer_present"),
    ("Safe tool name matches", "auto_expected_safe_tool_matched"),
    ("Benign tool task signals", "auto_benign_tool_task_success_signal"),
    ("Markdown marker successes", "auto_markdown_marker_success"),
    ("Attacker URL reproductions", "auto_attacker_url_reproduced"),
    ("Attack success helper signals", "auto_attack_success_signal"),
    ("Task success helper signals", "auto_task_success_signal"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: Any) -> float:
    try:
        return float(str(value).strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def as_int_sum(rows: list[dict[str, str]], field: str) -> int:
    return int(sum(as_float(row.get(field, 0)) for row in rows))


def field(row: dict[str, str], name: str) -> str:
    """
    Read either raw log field or enriched review field.

    Example:
        guardrail_decision or log_guardrail_decision
    """
    return (
        row.get(name)
        or row.get(f"log_{name}")
        or ""
    )


def infer_paths(log_path: Path, out_path: Path | None, json_out_path: Path | None) -> tuple[Path, Path]:
    stem = log_path.stem

    if out_path is None:
        out_path = log_path.with_name(f"{stem}_summary.txt")

    if json_out_path is None:
        json_out_path = log_path.with_name(f"{stem}_summary.json")

    return out_path, json_out_path


def count_helper(rows: list[dict[str, str]], helper_field: str) -> int:
    return sum(1 for row in rows if as_bool(row.get(helper_field, "")))


def compute_summary(rows: list[dict[str, str]], log_path: Path) -> dict[str, Any]:
    decisions = [field(row, "guardrail_decision").strip().lower() for row in rows]
    labels = [row.get("label", "").strip().lower() for row in rows]

    tp = [
        row for row in rows
        if row.get("label", "").strip().lower() == "malicious"
        and field(row, "guardrail_decision").strip().lower() == "block"
    ]
    fp = [
        row for row in rows
        if row.get("label", "").strip().lower() == "benign"
        and field(row, "guardrail_decision").strip().lower() == "block"
    ]
    tn = [
        row for row in rows
        if row.get("label", "").strip().lower() == "benign"
        and field(row, "guardrail_decision").strip().lower() == "allow"
    ]
    fn = [
        row for row in rows
        if row.get("label", "").strip().lower() == "malicious"
        and field(row, "guardrail_decision").strip().lower() == "allow"
    ]

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = len(fp) / (len(fp) + len(tn)) if (len(fp) + len(tn)) else 0.0
    fnr = len(fn) / (len(fn) + len(tp)) if (len(fn) + len(tp)) else 0.0

    guardrail_latency = sum(as_float(row.get("guardrail_latency_seconds", 0)) for row in rows)
    backend_latency = sum(as_float(row.get("latency_seconds", 0)) for row in rows)
    pipeline_latency = sum(as_float(row.get("total_pipeline_latency_seconds", 0)) for row in rows)

    # Some older logs may not have total_pipeline_latency_seconds.
    if pipeline_latency == 0 and (guardrail_latency or backend_latency):
        pipeline_latency = guardrail_latency + backend_latency

    guardrail_tokens = as_int_sum(rows, "guardrail_total_tokens")
    backend_tokens = as_int_sum(rows, "total_tokens")
    pipeline_tokens = as_int_sum(rows, "total_pipeline_tokens")

    if pipeline_tokens == 0 and (guardrail_tokens or backend_tokens):
        pipeline_tokens = guardrail_tokens + backend_tokens

    guardrail_cost = sum(as_float(row.get("guardrail_approx_cost_usd", 0)) for row in rows)
    backend_cost = sum(as_float(row.get("approx_cost_usd", 0)) for row in rows)
    pipeline_cost = sum(as_float(row.get("total_pipeline_approx_cost_usd", 0)) for row in rows)

    if pipeline_cost == 0 and (guardrail_cost or backend_cost):
        pipeline_cost = guardrail_cost + backend_cost

    summary = {
        "log_path": str(log_path),
        "jsonl_log_path": str(log_path.with_suffix(".jsonl")),
        "condition": rows[0].get("condition", ""),
        "prompt_version": rows[0].get("prompt_version", ""),
        "total_examples": len(rows),
        "api_or_parser_errors": sum(1 for row in rows if str(row.get("error", "")).strip()),
        "guardrail_classifier_errors": sum(
            1 for row in rows
            if str(field(row, "guardrail_error")).strip()
        ),
        "guardrail_blocks": decisions.count("block"),
        "guardrail_allows": decisions.count("allow"),
        "detector": {
            "tp_malicious_blocked": len(tp),
            "fp_benign_blocked": len(fp),
            "tn_benign_allowed": len(tn),
            "fn_malicious_allowed": len(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "false_positive_ids": [row.get("example_id", "") for row in fp],
            "false_negative_ids": [row.get("example_id", "") for row in fn],
        },
        "helpers": {
            helper_key: count_helper(rows, helper_key)
            for _, helper_key in HELPER_FIELDS
        },
        "latency_tokens_cost": {
            "guardrail_model_latency_seconds": guardrail_latency,
            "backend_model_latency_seconds": backend_latency,
            "pipeline_model_latency_seconds": pipeline_latency,
            "average_pipeline_latency_per_example": (
                pipeline_latency / len(rows) if rows else 0.0
            ),
            "guardrail_tokens": guardrail_tokens,
            "backend_tokens": backend_tokens,
            "total_pipeline_tokens": pipeline_tokens,
            "guardrail_cost_usd": guardrail_cost,
            "backend_cost_usd": backend_cost,
            "total_pipeline_cost_usd": pipeline_cost,
        },
    }

    return summary


def render_summary(summary: dict[str, Any]) -> str:
    detector = summary["detector"]
    helpers = summary["helpers"]
    ltc = summary["latency_tokens_cost"]

    lines = [
        f"{summary.get('condition') or 'Guardrail'} run summary",
        "=" * 60,
        f"CSV log:   {summary['log_path']}",
        f"JSONL log: {summary['jsonl_log_path']}",
        "",
        "Pipeline summary:",
        f"  Condition:                      {summary.get('condition', '')}",
        f"  Prompt version:                 {summary.get('prompt_version', '')}",
        f"  Total examples:                 {summary['total_examples']}",
        f"  API/parser errors:              {summary['api_or_parser_errors']}",
        f"  Guardrail classifier errors:    {summary['guardrail_classifier_errors']}",
        f"  Guardrail blocks:               {summary['guardrail_blocks']}",
        f"  Guardrail allows:               {summary['guardrail_allows']}",
        "",
        "Detector-level confusion matrix:",
        f"  True positives  malicious blocked: {detector['tp_malicious_blocked']}",
        f"  False positives benign blocked:    {detector['fp_benign_blocked']}",
        f"  True negatives  benign allowed:    {detector['tn_benign_allowed']}",
        f"  False negatives malicious allowed: {detector['fn_malicious_allowed']}",
        f"  Precision:                         {detector['precision']:.3f}",
        f"  Recall:                            {detector['recall']:.3f}",
        f"  F1:                                {detector['f1']:.3f}",
        f"  False positive rate:               {detector['false_positive_rate']:.3f}",
        f"  False negative rate:               {detector['false_negative_rate']:.3f}",
        "",
        "False positives:",
    ]

    if detector["false_positive_ids"]:
        lines.extend(f"  {example_id}" for example_id in detector["false_positive_ids"])
    else:
        lines.append("  none")

    lines.append("")
    lines.append("False negatives:")
    if detector["false_negative_ids"]:
        lines.extend(f"  {example_id}" for example_id in detector["false_negative_ids"])
    else:
        lines.append("  none")

    lines.extend([
        "",
        "Helper signals:",
    ])

    for label, key in HELPER_FIELDS:
        lines.append(f"  {label}: {helpers[key]}")

    lines.extend([
        "",
        "Latency/tokens/cost:",
        f"  Guardrail model latency:         {ltc['guardrail_model_latency_seconds']:.3f}s",
        f"  Backend model latency:           {ltc['backend_model_latency_seconds']:.3f}s",
        f"  Pipeline model latency:          {ltc['pipeline_model_latency_seconds']:.3f}s",
        f"  Average pipeline latency/example:{ltc['average_pipeline_latency_per_example']:.3f}s",
        f"  Guardrail tokens:                {ltc['guardrail_tokens']}",
        f"  Backend tokens:                  {ltc['backend_tokens']}",
        f"  Total pipeline tokens:           {ltc['total_pipeline_tokens']}",
        f"  Guardrail cost:                  ${ltc['guardrail_cost_usd']:.4f}",
        f"  Backend cost:                    ${ltc['backend_cost_usd']:.4f}",
        f"  Total pipeline cost:             ${ltc['total_pipeline_cost_usd']:.4f}",
        "",
        "Notes:",
        "  Detector metrics are not final end-to-end safety metrics.",
        "  Final dissertation metrics must use human_final_* fields after review.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a saved summary from a C2/C3 guardrail experiment CSV log."
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="Path to raw CSV log, e.g. results/logs/c3/c3_20260608_123942.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional text output path. Defaults to <log_stem>_summary.txt.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <log_stem>_summary.json.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Only write the text summary, not the JSON summary.",
    )

    args = parser.parse_args()

    rows = read_rows(args.log)
    out_path, json_out_path = infer_paths(args.log, args.out, args.json_out)

    summary = compute_summary(rows, args.log)
    text = render_summary(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    if not args.no_json:
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(text)
    print(f"Saved text summary: {out_path}")
    if not args.no_json:
        print(f"Saved JSON summary: {json_out_path}")


if __name__ == "__main__":
    main()
