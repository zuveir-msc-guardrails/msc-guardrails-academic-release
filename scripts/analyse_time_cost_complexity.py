#!/usr/bin/env python3
"""
analyse_time_cost_complexity.py
-------------------------------
Analyse time, token cost, and pipeline complexity across guardrail conditions.

Designed for this MSc Guardrails repo.

It reads the raw experiment log CSVs from:

    results/logs/<condition>/<condition>_*.csv

and writes dissertation-ready CSV/Markdown outputs to:

    results/analysis/time_cost_complexity

Supported by default:

    C0, C1, C2, C3, C5a, C5b, C5c

It reads C5c from:

    results/logs/c5c/c5c_*.csv

Use --c5c to point to a specific C5c log when you want to avoid accidentally analysing a later exploratory run.

What it computes:
    - total / mean / median / p95 pipeline latency
    - total / mean pipeline cost
    - total / mean token usage
    - component breakdown:
        guardrail detector
        sentence/context sanitiser
        backend agent
    - estimated LLM call count
    - pipeline action counts
    - sanitisation complexity:
        attempted
        succeeded
        fallback blocks
        units removed
        unit removal rate
    - per-row efficiency table

Important:
    This script is for runtime/cost/complexity analysis.
    Final safety/utility metrics still come from analyse_condition_results.py
    and the human_final_* fields.


Usage:
    python scripts/analyse_time_cost_complexity.py \
    --c0  results/logs/c0/c0_20260605_124503.csv \
    --c1  results/logs/c1/c1_20260607_110936.csv \
    --c2  results/logs/c2/c2_20260607_172130.csv \
    --c3  results/logs/c3/c3_20260608_123942.csv \
    --c5a results/logs/c5a/c5a_20260610_113959.csv \
    --c5b results/logs/c5b/c5b_20260609_125117.csv \
    --c5c results/logs/c5c/c5c_20260611_113203.csv \
    --outdir results/analysis/time_cost_complexity

"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONDITIONS = ["C0", "C1", "C2", "C3", "C5a", "C5b", "C5c"]

SANITISATION_CONDITIONS = {"C5a", "C5b", "C5c"}


@dataclass
class ConditionLog:
    condition: str
    path: Path


# ── Basic CSV / parsing helpers ─────────────────────────────────────────────

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def norm(value: Any) -> str:
    return str(value or "").strip()


def norm_lower(value: Any) -> str:
    return norm(value).lower()


def to_float(value: Any, default: float = 0.0) -> float:
    v = norm(value)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def to_int(value: Any, default: int = 0) -> int:
    v = norm(value)
    if not v:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def as_bool(value: Any) -> bool | None:
    v = norm_lower(value)
    if v in {"true", "1", "yes", "y", "checked"}:
        return True
    if v in {"false", "0", "no", "n", "unchecked"}:
        return False
    return None


def get_field(row: dict[str, str], name: str) -> str:
    """
    Resolve raw and review-prefixed fields.

    Examples:
        guardrail_decision or log_guardrail_decision
        pipeline_action or log_pipeline_action
    """
    return norm(row.get(name) or row.get(f"log_{name}") or "")


def get_num(row: dict[str, str], name: str) -> float:
    return to_float(get_field(row, name))


def get_int(row: dict[str, str], name: str) -> int:
    return to_int(get_field(row, name))


def get_bool(row: dict[str, str], name: str) -> bool | None:
    return as_bool(get_field(row, name))


def rate(n: int | float, d: int | float) -> float:
    return float(n) / float(d) if d else 0.0


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    """
    Simple percentile using nearest-rank interpolation.
    """
    cleaned = sorted(v for v in values if v is not None)
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return cleaned[0]
    idx = round((len(cleaned) - 1) * p)
    return cleaned[max(0, min(idx, len(cleaned) - 1))]


# ── Log discovery ───────────────────────────────────────────────────────────

def infer_latest_log(condition: str, logs_dir: Path) -> Path | None:
    c = condition.lower()
    candidates = list((logs_dir / c).glob(f"{c}_*.csv"))
    if not candidates:
        candidates = list((logs_dir / condition).glob(f"{condition}_*.csv"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def condition_sort_key(condition: str, order: list[str]) -> int:
    mapping = {c: i for i, c in enumerate(order)}
    return mapping.get(condition, 999)


# ── Row-level efficiency and complexity ─────────────────────────────────────

def pipeline_action(row: dict[str, str], condition: str) -> str:
    action = get_field(row, "pipeline_action")
    if action:
        return action.lower()

    decision = get_field(row, "guardrail_decision").lower()
    if condition in {"C2", "C3"} and decision == "block":
        return "blocked"
    if decision == "allow":
        return "allow_full_context"

    return "unknown"


def backend_called(row: dict[str, str]) -> bool:
    explicit = get_bool(row, "backend_called")
    if explicit is not None:
        return explicit

    # Fallback inference: backend latency/tokens/cost present means backend ran.
    return (
        get_num(row, "latency_seconds") > 0
        or get_int(row, "total_tokens") > 0
        or bool(get_field(row, "full_output"))
    )


def row_component_values(row: dict[str, str]) -> dict[str, float | int]:
    """
    Extract or infer per-row component costs/times/tokens.

    Naming convention in the current runners:
        guardrail_*             = document-level detector
        sentence_guardrail_*    = C5a/C5c sanitisation LLM calls
        latency_seconds         = backend agent latency
        total_tokens            = backend agent tokens
        approx_cost_usd         = backend agent cost
        total_pipeline_*        = already-summed totals when available
    """
    guardrail_latency = get_num(row, "guardrail_latency_seconds")
    guardrail_tokens = get_int(row, "guardrail_total_tokens")
    guardrail_cost = get_num(row, "guardrail_approx_cost_usd")

    sanitiser_latency = get_num(row, "sentence_guardrail_latency_seconds")
    sanitiser_tokens = get_int(row, "sentence_guardrail_total_tokens")
    sanitiser_cost = get_num(row, "sentence_guardrail_approx_cost_usd")

    backend_latency = get_num(row, "latency_seconds")
    backend_tokens = get_int(row, "total_tokens")
    backend_cost = get_num(row, "approx_cost_usd")

    # Prefer explicit pipeline totals when present. Otherwise sum components.
    explicit_pipeline_latency = get_num(row, "total_pipeline_latency_seconds")
    explicit_pipeline_tokens = get_int(row, "total_pipeline_tokens")
    explicit_pipeline_cost = get_num(row, "total_pipeline_approx_cost_usd")

    pipeline_latency = explicit_pipeline_latency or (guardrail_latency + sanitiser_latency + backend_latency)
    pipeline_tokens = explicit_pipeline_tokens or (guardrail_tokens + sanitiser_tokens + backend_tokens)
    pipeline_cost = explicit_pipeline_cost or (guardrail_cost + sanitiser_cost + backend_cost)

    return {
        "guardrail_latency_seconds": guardrail_latency,
        "guardrail_total_tokens": guardrail_tokens,
        "guardrail_approx_cost_usd": guardrail_cost,
        "sanitiser_latency_seconds": sanitiser_latency,
        "sanitiser_total_tokens": sanitiser_tokens,
        "sanitiser_approx_cost_usd": sanitiser_cost,
        "backend_latency_seconds": backend_latency,
        "backend_total_tokens": backend_tokens,
        "backend_approx_cost_usd": backend_cost,
        "total_pipeline_latency_seconds": pipeline_latency,
        "total_pipeline_tokens": pipeline_tokens,
        "total_pipeline_approx_cost_usd": pipeline_cost,
    }


def estimated_llm_calls(row: dict[str, str], condition: str) -> int:
    """
    Estimate row-level LLM calls.

    C0/C1:
        usually backend only.

    C2:
        regex guardrail + backend if allowed.
        Regex is not counted as an LLM call.

    C3:
        document-level LLM classifier + backend if allowed.

    C5a:
        document-level LLM classifier
        + one LLM call per sentence/unit when sanitisation attempted
        + backend if allowed or sanitised.

    C5b:
        document-level LLM classifier
        + deterministic fuzzy sanitisation, no extra LLM call
        + backend if allowed or sanitised.

    C5c:
        document-level LLM classifier
        + one context-aware LLM sanitisation call when sanitisation attempted
        + backend if allowed or sanitised.
    """
    calls = 0

    # Backend call.
    if backend_called(row):
        calls += 1

    # Document-level LLM classifier. C2 regex blocks are not LLM detector calls.
    if condition in {"C3", "C5a", "C5b", "C5c"}:
        # Count it if this condition architecture includes LLM detector.
        calls += 1
    elif get_int(row, "guardrail_total_tokens") > 0 or get_num(row, "guardrail_approx_cost_usd") > 0:
        calls += 1

    sanitisation_attempted = get_bool(row, "sanitisation_attempted") is True
    if condition == "C5a" and sanitisation_attempted:
        # Current C5a independently classifies each unit.
        units = get_int(row, "sentence_units_total")
        calls += units
    elif condition == "C5c" and sanitisation_attempted:
        # C5c uses one context-aware call over all numbered units.
        calls += 1
    elif condition not in {"C5a", "C5c"}:
        # If some future runner logs sanitiser tokens, count it as one extra LLM
        # stage unless the condition is known deterministic like C5b.
        if condition != "C5b" and get_int(row, "sentence_guardrail_total_tokens") > 0:
            calls += 1

    return calls


def complexity_path(row: dict[str, str], condition: str) -> str:
    action = pipeline_action(row, condition)
    decision = get_field(row, "guardrail_decision").lower()

    if condition == "C0":
        return "backend_only"
    if condition == "C1":
        return "security_prompt_backend"
    if condition == "C2":
        if decision == "block" or action in {"blocked", "blocked_fallback"}:
            return "regex_block"
        return "regex_allow_backend"
    if condition == "C3":
        if decision == "block" or action in {"blocked", "blocked_fallback"}:
            return "llm_classifier_block"
        return "llm_classifier_allow_backend"
    if condition == "C5a":
        if action == "sanitised_context":
            return "llm_classifier_sentence_llm_sanitise_backend"
        if action == "blocked_fallback":
            return "llm_classifier_sentence_llm_fallback_block"
        return "llm_classifier_allow_backend"
    if condition == "C5b":
        if action == "sanitised_context":
            return "llm_classifier_deterministic_sanitise_backend"
        if action == "blocked_fallback":
            return "llm_classifier_deterministic_fallback_block"
        return "llm_classifier_allow_backend"
    if condition == "C5c":
        if action == "sanitised_context":
            return "llm_classifier_context_aware_sanitise_backend"
        if action == "blocked_fallback":
            return "llm_classifier_context_aware_fallback_block"
        return "llm_classifier_allow_backend"

    return action or "unknown"


def complexity_score_for_path(path: str) -> int:
    """
    A simple architectural complexity score.

    This is not a performance metric. It is a descriptive score for comparing
    how many moving parts a row passed through.

    Approximate interpretation:
        1 = backend-only
        2 = gate + backend or gate + block
        3 = LLM classifier + backend/block
        4 = classifier + deterministic sanitisation + backend/block
        5 = classifier + one context-aware sanitiser LLM + backend/block
        6+ = classifier + many independent unit LLM calls + backend/block
    """
    if path == "backend_only":
        return 1
    if path == "security_prompt_backend":
        return 1
    if path.startswith("regex_"):
        return 2
    if path.startswith("llm_classifier_allow") or path.startswith("llm_classifier_block"):
        return 3
    if "deterministic_sanitise" in path or "deterministic_fallback" in path:
        return 4
    if "context_aware" in path:
        return 5
    if "sentence_llm" in path:
        return 6
    return 3


def per_row_record(condition: str, row: dict[str, str]) -> dict[str, Any]:
    comps = row_component_values(row)
    action = pipeline_action(row, condition)
    path = complexity_path(row, condition)

    sentence_units_total = get_int(row, "sentence_units_total")
    sentence_units_blocked = get_int(row, "sentence_units_blocked")
    segments_removed = get_int(row, "sanitisation_segments_removed")
    units_removed = sentence_units_blocked or segments_removed

    return {
        "condition": condition,
        "example_id": row.get("example_id", ""),
        "label": norm_lower(row.get("label") or row.get("core_label")),
        "stratum": row.get("stratum") or row.get("core_stratum") or "",
        "attack_type": row.get("attack_type") or row.get("core_attack_type") or "",
        "guardrail_decision": get_field(row, "guardrail_decision").lower(),
        "pipeline_action": action,
        "backend_called": str(backend_called(row)).lower(),
        "complexity_path": path,
        "complexity_score": complexity_score_for_path(path),
        "estimated_llm_calls": estimated_llm_calls(row, condition),
        "sanitisation_attempted": str(get_bool(row, "sanitisation_attempted") is True).lower(),
        "sanitisation_succeeded": str(get_bool(row, "sanitisation_succeeded") is True).lower(),
        "sanitisation_fallback_to_block": str(get_bool(row, "sanitisation_fallback_to_block") is True).lower(),
        "sentence_units_total": sentence_units_total,
        "sentence_units_blocked": sentence_units_blocked,
        "sanitisation_segments_removed": segments_removed,
        "units_or_segments_removed": units_removed,
        **comps,
    }


# ── Aggregation ─────────────────────────────────────────────────────────────

def summarise_condition(condition: str, rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_rows = [per_row_record(condition, row) for row in rows]

    n = len(per_rows)
    latency_values = [to_float(row["total_pipeline_latency_seconds"]) for row in per_rows]
    cost_values = [to_float(row["total_pipeline_approx_cost_usd"]) for row in per_rows]
    token_values = [to_float(row["total_pipeline_tokens"]) for row in per_rows]
    llm_call_values = [to_float(row["estimated_llm_calls"]) for row in per_rows]
    complexity_values = [to_float(row["complexity_score"]) for row in per_rows]

    action_counts = Counter(row["pipeline_action"] for row in per_rows)
    path_counts = Counter(row["complexity_path"] for row in per_rows)

    backend_called_count = sum(1 for row in per_rows if row["backend_called"] == "true")
    final_block_count = action_counts["blocked_fallback"] + action_counts["blocked"]
    sanitised_count = action_counts["sanitised_context"]

    sentence_units_total = sum(to_int(row["sentence_units_total"]) for row in per_rows)
    sentence_units_blocked = sum(to_int(row["sentence_units_blocked"]) for row in per_rows)
    units_or_segments_removed = sum(to_int(row["units_or_segments_removed"]) for row in per_rows)

    summary = {
        "condition": condition,
        "n": n,
        "backend_called_count": backend_called_count,
        "backend_called_rate": rate(backend_called_count, n),
        "final_block_count": final_block_count,
        "final_block_rate": rate(final_block_count, n),
        "sanitised_context_count": sanitised_count,
        "sanitised_context_rate": rate(sanitised_count, n),

        "total_pipeline_latency_seconds": round(sum(latency_values), 3),
        "mean_pipeline_latency_seconds": round(safe_mean(latency_values), 3),
        "median_pipeline_latency_seconds": round(safe_median(latency_values), 3),
        "p95_pipeline_latency_seconds": round(percentile(latency_values, 0.95), 3),

        "total_pipeline_approx_cost_usd": round(sum(cost_values), 6),
        "mean_pipeline_approx_cost_usd": round(safe_mean(cost_values), 6),
        "median_pipeline_approx_cost_usd": round(safe_median(cost_values), 6),
        "p95_pipeline_approx_cost_usd": round(percentile(cost_values, 0.95), 6),

        "total_pipeline_tokens": int(sum(token_values)),
        "mean_pipeline_tokens": round(safe_mean(token_values), 1),
        "median_pipeline_tokens": round(safe_median(token_values), 1),
        "p95_pipeline_tokens": round(percentile(token_values, 0.95), 1),

        "estimated_total_llm_calls": int(sum(llm_call_values)),
        "mean_llm_calls_per_row": round(safe_mean(llm_call_values), 2),
        "median_llm_calls_per_row": round(safe_median(llm_call_values), 2),
        "p95_llm_calls_per_row": round(percentile(llm_call_values, 0.95), 2),

        "mean_complexity_score": round(safe_mean(complexity_values), 2),
        "median_complexity_score": round(safe_median(complexity_values), 2),
        "p95_complexity_score": round(percentile(complexity_values, 0.95), 2),

        "sentence_units_total_sum": sentence_units_total,
        "sentence_units_blocked_sum": sentence_units_blocked,
        "sentence_units_blocked_rate": rate(sentence_units_blocked, sentence_units_total),
        "units_or_segments_removed_sum": units_or_segments_removed,
        "mean_units_or_segments_removed_per_sanitised_row": round(
            units_or_segments_removed / sanitised_count, 2
        ) if sanitised_count else 0.0,
    }

    component_totals = {
        "condition": condition,
        "n": n,
        "guardrail_latency_seconds": round(sum(to_float(row["guardrail_latency_seconds"]) for row in per_rows), 3),
        "guardrail_tokens": sum(to_int(row["guardrail_total_tokens"]) for row in per_rows),
        "guardrail_cost_usd": round(sum(to_float(row["guardrail_approx_cost_usd"]) for row in per_rows), 6),

        "sanitiser_latency_seconds": round(sum(to_float(row["sanitiser_latency_seconds"]) for row in per_rows), 3),
        "sanitiser_tokens": sum(to_int(row["sanitiser_total_tokens"]) for row in per_rows),
        "sanitiser_cost_usd": round(sum(to_float(row["sanitiser_approx_cost_usd"]) for row in per_rows), 6),

        "backend_latency_seconds": round(sum(to_float(row["backend_latency_seconds"]) for row in per_rows), 3),
        "backend_tokens": sum(to_int(row["backend_total_tokens"]) for row in per_rows),
        "backend_cost_usd": round(sum(to_float(row["backend_approx_cost_usd"]) for row in per_rows), 6),

        "total_latency_seconds": summary["total_pipeline_latency_seconds"],
        "total_tokens": summary["total_pipeline_tokens"],
        "total_cost_usd": summary["total_pipeline_approx_cost_usd"],
    }

    action_rows = [
        {
            "condition": condition,
            "pipeline_action": action,
            "count": count,
            "rate": rate(count, n),
        }
        for action, count in sorted(action_counts.items())
    ]

    path_rows = [
        {
            "condition": condition,
            "complexity_path": path,
            "complexity_score": complexity_score_for_path(path),
            "count": count,
            "rate": rate(count, n),
        }
        for path, count in sorted(path_counts.items())
    ]

    return summary, component_totals, action_rows, path_rows


def pairwise_efficiency(summary_rows: list[dict[str, Any]], baseline: str = "C0") -> list[dict[str, Any]]:
    by_condition = {row["condition"]: row for row in summary_rows}
    base = by_condition.get(baseline)
    if not base:
        return []

    metrics = [
        "mean_pipeline_latency_seconds",
        "total_pipeline_latency_seconds",
        "mean_pipeline_approx_cost_usd",
        "total_pipeline_approx_cost_usd",
        "mean_pipeline_tokens",
        "total_pipeline_tokens",
        "mean_llm_calls_per_row",
        "estimated_total_llm_calls",
        "mean_complexity_score",
    ]

    rows = []
    for condition, row in by_condition.items():
        if condition == baseline:
            continue
        out = {
            "baseline_condition": baseline,
            "condition": condition,
        }
        for metric in metrics:
            base_value = to_float(base.get(metric))
            value = to_float(row.get(metric))
            out[f"{metric}_baseline"] = base_value
            out[f"{metric}_{condition}"] = value
            out[f"{metric}_delta"] = value - base_value
            out[f"{metric}_ratio_vs_baseline"] = rate(value, base_value)
        rows.append(out)

    return rows


# ── Markdown report ─────────────────────────────────────────────────────────

def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], percent_cols: set[str] | None = None) -> str:
    if not rows:
        return "_No rows._\n"

    percent_cols = percent_cols or set()
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []

    for row in rows:
        vals = []
        for key, _label in columns:
            val = row.get(key, "")
            if key in percent_cols:
                vals.append(pct(to_float(val)))
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + body) + "\n"


def render_markdown(
    log_inputs: list[ConditionLog],
    summary_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    baseline: str,
) -> str:
    lines = ["# Time, cost, and complexity analysis", ""]

    lines.append("## Input log files")
    for item in log_inputs:
        lines.append(f"- **{item.condition}**: `{item.path}`")
    lines.append("")

    lines.append("## Efficiency summary")
    lines.append(markdown_table(
        summary_rows,
        [
            ("condition", "Condition"),
            ("n", "n"),
            ("mean_pipeline_latency_seconds", "Mean latency (s)"),
            ("p95_pipeline_latency_seconds", "P95 latency (s)"),
            ("total_pipeline_latency_seconds", "Total latency (s)"),
            ("mean_pipeline_approx_cost_usd", "Mean cost ($)"),
            ("total_pipeline_approx_cost_usd", "Total cost ($)"),
            ("mean_pipeline_tokens", "Mean tokens"),
            ("total_pipeline_tokens", "Total tokens"),
            ("mean_llm_calls_per_row", "Mean LLM calls"),
            ("mean_complexity_score", "Complexity score"),
            ("final_block_rate", "Final block rate"),
            ("sanitised_context_rate", "Sanitised rate"),
            ("mean_units_or_segments_removed_per_sanitised_row", "Mean removed / sanitised row"),
        ],
        {"final_block_rate", "sanitised_context_rate"},
    ))

    lines.append("## Component totals")
    lines.append(markdown_table(
        component_rows,
        [
            ("condition", "Condition"),
            ("guardrail_tokens", "Detector tokens"),
            ("guardrail_cost_usd", "Detector cost"),
            ("sanitiser_tokens", "Sanitiser tokens"),
            ("sanitiser_cost_usd", "Sanitiser cost"),
            ("backend_tokens", "Backend tokens"),
            ("backend_cost_usd", "Backend cost"),
            ("total_tokens", "Total tokens"),
            ("total_cost_usd", "Total cost"),
        ],
    ))

    lines.append(f"## Efficiency deltas versus {baseline}")
    lines.append(markdown_table(
        deltas,
        [
            ("condition", "Condition"),
            ("mean_pipeline_latency_seconds_delta", "Mean latency Δ"),
            ("total_pipeline_approx_cost_usd_delta", "Total cost Δ"),
            ("mean_pipeline_tokens_delta", "Mean tokens Δ"),
            ("mean_llm_calls_per_row_delta", "Mean calls Δ"),
            ("mean_complexity_score_delta", "Complexity Δ"),
        ],
    ))

    lines.append("## Pipeline actions")
    lines.append(markdown_table(
        action_rows,
        [
            ("condition", "Condition"),
            ("pipeline_action", "Pipeline action"),
            ("count", "Count"),
            ("rate", "Rate"),
        ],
        {"rate"},
    ))

    lines.append("## Complexity paths")
    lines.append(markdown_table(
        complexity_rows,
        [
            ("condition", "Condition"),
            ("complexity_path", "Path"),
            ("complexity_score", "Score"),
            ("count", "Count"),
            ("rate", "Rate"),
        ],
        {"rate"},
    ))

    lines.append("## Interpretation notes")
    lines.append(
        "- `total_pipeline_*` uses explicit pipeline totals where available, otherwise it sums detector, sanitiser, and backend components."
    )
    lines.append(
        "- C5a estimates one extra LLM call per sentence-like unit when sanitisation is attempted."
    )
    lines.append(
        "- C5b does not add LLM calls for sanitisation because its removal step is deterministic fuzzy payload removal."
    )
    lines.append(
        "- C5c estimates one extra LLM sanitiser call for each sanitisation attempt because it uses a single context-aware unit-removal call."
    )
    lines.append(
        "- Use the selected minimal C5c log for dissertation comparisons. Exploratory prompt-v2 C5c logs can be analysed separately by passing --c5c and a different --outdir."
    )
    lines.append(
        "- Complexity score is descriptive rather than a final metric; use it to explain architectural cost/operational burden."
    )

    return "\n".join(lines) + "\n"


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse time, cost, token use, and pipeline complexity from C0/C1/C2/C3/C5a/C5b/C5c experiment logs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("results/logs"),
        help="Root directory containing condition log folders.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/analysis/time_cost_complexity"),
        help="Output directory.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Conditions to analyse, e.g. C0 C1 C2 C3 C5a C5b C5c.",
    )
    parser.add_argument("--c0", type=Path, default=None, help="Explicit C0 log CSV.")
    parser.add_argument("--c1", type=Path, default=None, help="Explicit C1 log CSV.")
    parser.add_argument("--c2", type=Path, default=None, help="Explicit C2 log CSV.")
    parser.add_argument("--c3", type=Path, default=None, help="Explicit C3 log CSV.")
    parser.add_argument("--c5a", type=Path, default=None, help="Explicit C5a log CSV.")
    parser.add_argument("--c5b", type=Path, default=None, help="Explicit C5b log CSV.")
    parser.add_argument("--c5c", type=Path, default=None, help="Explicit C5c log CSV. Recommended: use the selected minimal C5c log, not exploratory prompt-v2 logs.")
    parser.add_argument(
        "--baseline",
        default="C0",
        help="Baseline condition for efficiency deltas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    explicit = {
        "C0": args.c0,
        "C1": args.c1,
        "C2": args.c2,
        "C3": args.c3,
        "C5a": args.c5a,
        "C5b": args.c5b,
        "C5c": args.c5c,
    }

    requested_conditions = args.conditions
    order = requested_conditions

    inputs: list[ConditionLog] = []
    missing: list[str] = []

    for condition in requested_conditions:
        path = explicit.get(condition) or infer_latest_log(condition, args.logs_dir)
        if path:
            inputs.append(ConditionLog(condition=condition, path=path))
        else:
            missing.append(condition)

    if not inputs:
        raise SystemExit("No log CSVs found. Pass explicit --c0/--c1/--c2/--c3/--c5a/--c5b/--c5c paths or check --logs-dir.")

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    per_row_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for item in inputs:
        rows = read_csv(item.path)

        summary, components, actions, complexity_paths = summarise_condition(item.condition, rows)

        summary_rows.append(summary)
        component_rows.append(components)
        action_rows.extend(actions)
        complexity_rows.extend(complexity_paths)
        per_row_rows.extend(per_row_record(item.condition, row) for row in rows)

        manifest_rows.append({
            "condition": item.condition,
            "path": str(item.path),
            "rows": len(rows),
            "prompt_version": rows[0].get("prompt_version", "") or rows[0].get("log_prompt_version", ""),
        })

    summary_rows.sort(key=lambda r: condition_sort_key(r["condition"], order))
    component_rows.sort(key=lambda r: condition_sort_key(r["condition"], order))
    action_rows.sort(key=lambda r: (condition_sort_key(r["condition"], order), r["pipeline_action"]))
    complexity_rows.sort(key=lambda r: (condition_sort_key(r["condition"], order), r["complexity_path"]))
    per_row_rows.sort(key=lambda r: (condition_sort_key(r["condition"], order), r["example_id"]))

    deltas = pairwise_efficiency(summary_rows, baseline=args.baseline)
    deltas.sort(key=lambda r: condition_sort_key(r["condition"], order))

    write_csv(outdir / "efficiency_summary.csv", summary_rows)
    write_csv(outdir / "component_totals.csv", component_rows)
    write_csv(outdir / "pipeline_action_counts.csv", action_rows)
    write_csv(outdir / "complexity_paths.csv", complexity_rows)
    write_csv(outdir / "per_row_efficiency.csv", per_row_rows)
    write_csv(outdir / "efficiency_deltas_vs_baseline.csv", deltas)
    write_csv(outdir / "manifest.csv", manifest_rows)

    write_json(outdir / "efficiency_summary.json", {
        "manifest": manifest_rows,
        "missing_conditions": missing,
        "efficiency_summary": summary_rows,
        "component_totals": component_rows,
        "pipeline_action_counts": action_rows,
        "complexity_paths": complexity_rows,
        "efficiency_deltas_vs_baseline": deltas,
    })

    md = render_markdown(
        log_inputs=inputs,
        summary_rows=summary_rows,
        component_rows=component_rows,
        action_rows=action_rows,
        complexity_rows=complexity_rows,
        deltas=deltas,
        baseline=args.baseline,
    )
    (outdir / "time_cost_complexity_summary.md").write_text(md, encoding="utf-8")

    print("Time, cost, and complexity analysis complete")
    print("=" * 60)
    print(f"Output directory: {outdir}")
    if missing:
        print(f"Missing conditions skipped: {', '.join(missing)}")
    print()
    print("Wrote:")
    for name in [
        "manifest.csv",
        "efficiency_summary.csv",
        "component_totals.csv",
        "pipeline_action_counts.csv",
        "complexity_paths.csv",
        "per_row_efficiency.csv",
        "efficiency_deltas_vs_baseline.csv",
        "efficiency_summary.json",
        "time_cost_complexity_summary.md",
    ]:
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
