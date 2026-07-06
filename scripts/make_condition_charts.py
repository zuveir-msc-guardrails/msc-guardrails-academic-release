#!/usr/bin/env python3
"""
make_condition_charts.py
------------------------
Create dissertation-ready charts from the output of scripts/analyse_condition_results.py.

Expected input directory:
    results/analysis/c0_to_c5c

Expected analysis files:
    overall_metrics.csv
    metrics_by_stratum.csv
    detector_metrics.csv
    score_matrix.csv
    sanitisation_metrics.csv
    sanitisation_by_stratum.csv

Outputs:
    PNG charts into results/charts/c0_to_c5c by default
    chart_index.md with image links and short interpretation notes

Usage:
    python scripts/make_condition_charts.py \
      --analysis-dir results/analysis/c0_to_c5c_answerkey_corrected \
      --outdir results/charts/c0_to_c5c

Notes:
    - Uses matplotlib only.
    - Does not use seaborn.
    - Does not set custom colours, so matplotlib defaults are used.
    - Creates one chart per figure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = ["C0", "C1", "C2", "C3", "C5a", "C5b", "C5c"]
SANITISATION_CONDITION_ORDER = ["C5a", "C5b", "C5c"]

MALICIOUS_STRATA = [
    "data_exfiltration",
    "instruction_override",
    "markdown_injection",
    "tool_misuse",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required analysis file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Analysis file is empty: {path}")
    return df


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def condition_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "condition" not in df.columns:
        return df
    order = {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    return df.assign(_condition_order=df["condition"].map(order).fillna(999)).sort_values("_condition_order").drop(columns=["_condition_order"])


def pct_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0) * 100.0


def add_bar_labels(ax: plt.Axes, suffix: str = "%", decimals: int = 1) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        label = f"{height:.{decimals}f}{suffix}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def save_current_figure(outpath: Path) -> None:
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def write_chart_index(outdir: Path, chart_entries: list[dict[str, str]]) -> None:
    lines = ["# C0–C5b chart outputs, including C5a", ""]

    for entry in chart_entries:
        lines.append(f"## {entry['title']}")
        lines.append("")
        lines.append(entry["description"])
        lines.append("")
        lines.append(f"![{entry['title']}]({entry['filename']})")
        lines.append("")

    (outdir / "chart_index.md").write_text("\n".join(lines), encoding="utf-8")


def grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    outpath: Path,
    percent: bool = True,
    footer: str | None = None,
) -> None:
    x_labels = list(df[x_col])
    x = range(len(x_labels))
    width = 0.8 / max(len(y_cols), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), 5))

    for i, (col, label) in enumerate(zip(y_cols, labels)):
        values = pct_series(df[col]) if percent else pd.to_numeric(df[col], errors="coerce").fillna(0)
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.legend()

    if percent:
        ax.set_ylim(0, 105)

    if footer:
        ax.text(0, -0.18, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def simple_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
    percent: bool = True,
    footer: str | None = None,
) -> None:
    values = pct_series(df[y_col]) if percent else pd.to_numeric(df[y_col], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.1), 5))
    ax.bar(df[x_col], values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

    if percent:
        ax.set_ylim(0, 105)
        add_bar_labels(ax, suffix="%", decimals=1)
    else:
        add_bar_labels(ax, suffix="", decimals=0)

    if footer:
        ax.text(0, -0.18, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def horizontal_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    outpath: Path,
    percent: bool = True,
    footer: str | None = None,
) -> None:
    plot_df = df.copy()
    plot_df[value_col] = pct_series(plot_df[value_col]) if percent else pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.45)))
    ax.barh(plot_df[label_col], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    if percent:
        ax.set_xlim(0, 105)

    for patch in ax.patches:
        width = patch.get_width()
        label = f"{width:.1f}%" if percent else f"{width:.0f}"
        ax.annotate(
            label,
            (width, patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(4, 0),
            textcoords="offset points",
        )

    if footer:
        ax.text(0, -0.12, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def scatter_with_labels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    percent: bool = True,
    footer: str | None = None,
) -> None:
    """Create a labelled scatter plot for security/utility trade-off charts."""
    plot_df = df.copy()
    plot_df[x_col] = pct_series(plot_df[x_col]) if percent else pd.to_numeric(plot_df[x_col], errors="coerce").fillna(0)
    plot_df[y_col] = pct_series(plot_df[y_col]) if percent else pd.to_numeric(plot_df[y_col], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df[x_col], plot_df[y_col], s=90)

    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row[label_col]),
            (row[x_col], row[y_col]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if percent:
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)

    ax.grid(True, linewidth=0.5, alpha=0.35)

    if footer:
        ax.text(0, -0.15, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_overall_security_utility(overall: pd.DataFrame, outdir: Path) -> dict[str, str]:
    df = condition_sort(overall.copy())

    outpath = outdir / "01_overall_security_utility.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=["asr_malicious", "ua_rate_malicious", "task_success_rate_all_rows"],
        labels=["Malicious ASR", "Malicious UA", "Overall task success"],
        title="Security and utility by condition",
        ylabel="Rate",
        outpath=outpath,
        percent=True,
        footer="Lower ASR is better; higher UA and task success are better.",
    )

    return {
        "title": "Security and utility by condition",
        "description": "Compares malicious attack success rate, malicious utility under attack, and overall task success across C0, C1, C2, C3, C5a, and C5b.",
        "filename": outpath.name,
    }


def chart_asr_vs_ua_tradeoff(overall: pd.DataFrame, outdir: Path) -> dict[str, str]:
    """Plot the security/utility frontier: y = ASR, x = UA.

    Uses malicious ASR on the y-axis and overall UA on the x-axis. This gives a
    compact view of the trade-off: the best region is bottom-right.
    """
    required = {"condition", "asr_malicious", "ua_rate_all_rows"}
    missing = required - set(overall.columns)
    if missing:
        raise ValueError(f"overall_metrics.csv missing required columns for ASR-vs-UA chart: {sorted(missing)}")

    df = condition_sort(overall.copy())
    outpath = outdir / "02b_asr_vs_ua_tradeoff.png"

    scatter_with_labels(
        df=df,
        x_col="ua_rate_all_rows",
        y_col="asr_malicious",
        label_col="condition",
        title="Security-utility trade-off: ASR versus UA",
        xlabel="Overall utility under attack (UA)",
        ylabel="Targeted attack success rate (ASR)",
        outpath=outpath,
        percent=True,
        footer="Best region is bottom-right: low attack success and high utility under attack.",
    )

    return {
        "title": "Security-utility trade-off: ASR versus UA",
        "description": "Plots targeted malicious ASR on the y-axis against overall UA on the x-axis for each condition.",
        "filename": outpath.name,
    }


def chart_asr_malicious(overall: pd.DataFrame, outdir: Path) -> dict[str, str]:
    df = condition_sort(overall.copy())
    outpath = outdir / "02_malicious_asr_by_condition.png"

    simple_bar(
        df=df,
        x_col="condition",
        y_col="asr_malicious",
        title="Targeted attack success rate on malicious rows",
        ylabel="ASR on malicious rows",
        outpath=outpath,
        percent=True,
        footer="This is the primary attack-success metric: attack successes divided by malicious examples.",
    )

    return {
        "title": "Targeted ASR on malicious rows",
        "description": "Shows the targeted attack success rate for each condition using malicious rows only as the denominator.",
        "filename": outpath.name,
    }


def chart_task_success_breakdown(overall: pd.DataFrame, outdir: Path) -> dict[str, str]:
    df = condition_sort(overall.copy())
    outpath = outdir / "03_task_success_breakdown.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=[
            "task_success_rate_benign",
            "task_success_rate_malicious",
            "task_success_rate_all_rows",
        ],
        labels=[
            "Benign task success",
            "Malicious task success",
            "Overall task success",
        ],
        title="Task success by condition",
        ylabel="Task success rate",
        outpath=outpath,
        percent=True,
        footer="C5a and C5a, C5b, and C5c should be compared against C3 to show utility recovered after sanitisation.",
    )

    return {
        "title": "Task success by condition",
        "description": "Compares task success on benign rows, malicious rows, and all rows.",
        "filename": outpath.name,
    }


def chart_detector_metrics(detector: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if detector.empty:
        return None

    df = condition_sort(detector.copy())
    outpath = outdir / "04_detector_precision_recall_f1.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=["precision", "recall", "f1"],
        labels=["Precision", "Recall", "F1"],
        title="Detector metrics",
        ylabel="Rate",
        outpath=outpath,
        percent=True,
        footer="For C5a, C5b, and C5c, detector metrics are classifier-only because classifier blocks can lead to sanitised backend execution.",
    )

    return {
        "title": "Detector precision, recall, and F1",
        "description": "Compares detector precision, recall, and F1 for conditions that log guardrail decisions.",
        "filename": outpath.name,
    }


def chart_block_rates(overall: pd.DataFrame, outdir: Path) -> dict[str, str]:
    df = condition_sort(overall.copy())
    outpath = outdir / "05_classifier_vs_final_block_rate.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=["guardrail_block_rate", "final_block_rate"],
        labels=["Classifier/block decision rate", "Final full-block rate"],
        title="Classifier block rate versus final full-block rate",
        ylabel="Rate",
        outpath=outpath,
        percent=True,
        footer="C5a, C5b, and C5c separate classifier blocks from final blocks because sanitised rows still call the backend.",
    )

    return {
        "title": "Classifier block rate versus final full-block rate",
        "description": "Shows the difference between guardrail/classifier block decisions and actual end-to-end full blocks.",
        "filename": outpath.name,
    }


def chart_score_matrix(score_matrix: pd.DataFrame, outdir: Path) -> dict[str, str]:
    df = score_matrix.copy()
    df["outcome"] = (
        "AS=" + df["human_final_attack_success"].astype(str)
        + ", Task=" + df["human_final_task_success"].astype(str)
        + ", UA=" + df["human_final_ua"].astype(str)
    )

    pivot = df.pivot_table(
        index="condition",
        columns="outcome",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot = condition_sort(pivot)

    outcome_cols = [col for col in pivot.columns if col != "condition"]

    outpath = outdir / "06_score_matrix_counts.png"

    grouped_bar(
        df=pivot,
        x_col="condition",
        y_cols=outcome_cols,
        labels=outcome_cols,
        title="Score matrix counts by condition",
        ylabel="Row count",
        outpath=outpath,
        percent=False,
        footer="Counts of final human-scored attack/task/UA outcomes.",
    )

    return {
        "title": "Score matrix counts",
        "description": "Shows the final human-scored outcome combinations for each condition.",
        "filename": outpath.name,
    }


def chart_malicious_asr_by_stratum(metrics_by_stratum: pd.DataFrame, outdir: Path) -> dict[str, str]:
    df = metrics_by_stratum.copy()
    df = df[df["stratum"].isin(MALICIOUS_STRATA)].copy()
    df = condition_sort(df)

    pivot = df.pivot_table(
        index="stratum",
        columns="condition",
        values="asr_malicious",
        aggfunc="mean",
        fill_value=0,
    )

    for condition in CONDITION_ORDER:
        if condition not in pivot.columns:
            pivot[condition] = 0

    pivot = pivot[CONDITION_ORDER]
    pivot = pivot.reset_index()

    outpath = outdir / "07_asr_by_malicious_stratum.png"

    grouped_bar(
        df=pivot,
        x_col="stratum",
        y_cols=CONDITION_ORDER,
        labels=CONDITION_ORDER,
        title="Attack success rate by malicious stratum",
        ylabel="ASR on malicious rows",
        outpath=outpath,
        percent=True,
        footer="Compares where each condition is most or least vulnerable.",
    )

    return {
        "title": "ASR by malicious stratum",
        "description": "Compares malicious-row attack success rates by attack stratum and condition.",
        "filename": outpath.name,
    }



def chart_task_success_by_stratum_sanitisation_conditions(metrics_by_stratum: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = metrics_by_stratum.copy()
    df = df[df["condition"].isin(SANITISATION_CONDITION_ORDER)].copy()
    if df.empty:
        return None

    pivot = df.pivot_table(
        index="stratum",
        columns="condition",
        values="task_success_rate_all_rows",
        aggfunc="mean",
        fill_value=0,
    )

    available_conditions = [c for c in SANITISATION_CONDITION_ORDER if c in pivot.columns]
    if not available_conditions:
        return None

    pivot = pivot[available_conditions].reset_index()

    outpath = outdir / "08_c5a_c5b_task_success_by_stratum.png"

    grouped_bar(
        df=pivot,
        x_col="stratum",
        y_cols=available_conditions,
        labels=available_conditions,
        title="C5a, C5b, and C5c task success by stratum",
        ylabel="Task success rate",
        outpath=outpath,
        percent=True,
        footer="Compares utility preservation for LLM sentence-level sanitisation (C5a) and fuzzy payload removal (C5b).",
    )

    return {
        "title": "C5a, C5b, and C5c task success by stratum",
        "description": "Compares task success by stratum for C5a, C5b, and C5c.",
        "filename": outpath.name,
    }


def chart_sanitisation_flow_counts(sanitisation: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if sanitisation.empty:
        return None

    df = condition_sort(sanitisation.copy())
    df = df[df["condition"].isin(SANITISATION_CONDITION_ORDER)].copy()
    if df.empty:
        return None

    outpath = outdir / "09_c5a_c5b_c5c_sanitisation_flow_counts.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=[
            "sanitisation_attempted_count",
            "sanitisation_succeeded_count",
            "sanitisation_fallback_to_block_count",
            "backend_called_after_sanitisation_count",
        ],
        labels=[
            "Attempted",
            "Succeeded",
            "Fallback block",
            "Backend after sanitisation",
        ],
        title="C5a, C5b, and C5c sanitisation flow counts",
        ylabel="Row count",
        outpath=outpath,
        percent=False,
        footer="Shows how classifier-blocked rows flowed through sanitisation and fallback handling.",
    )

    return {
        "title": "C5a, C5b, and C5c sanitisation flow counts",
        "description": "Shows attempted, succeeded, fallback-blocked, and backend-after-sanitisation counts for C5a, C5b, and C5c.",
        "filename": outpath.name,
    }


def chart_sanitised_malicious_outcomes(sanitisation: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if sanitisation.empty:
        return None

    df = condition_sort(sanitisation.copy())
    df = df[df["condition"].isin(SANITISATION_CONDITION_ORDER)].copy()
    if df.empty:
        return None

    outpath = outdir / "10_c5a_c5b_c5c_sanitised_malicious_outcomes.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=[
            "malicious_sanitised_attack_success_rate",
            "malicious_sanitised_task_success_rate",
            "malicious_sanitised_ua_rate",
        ],
        labels=[
            "Attack success",
            "Task success",
            "UA",
        ],
        title="Outcomes on sanitised malicious rows",
        ylabel="Rate",
        outpath=outpath,
        percent=True,
        footer="Core utility-recovery comparison: lower attack success and higher task/UA are better.",
    )

    return {
        "title": "C5a, C5b, and C5c sanitised malicious outcomes",
        "description": "Shows attack success, task success, and UA among malicious rows sanitised by C5a, C5b, and C5c.",
        "filename": outpath.name,
    }


def chart_sanitised_ua_by_stratum(sanitisation_by_stratum: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if sanitisation_by_stratum.empty:
        return None

    df = sanitisation_by_stratum.copy()
    df = df[df["stratum"].isin(MALICIOUS_STRATA)].copy()
    df = df[df["condition"].isin(SANITISATION_CONDITION_ORDER)].copy()
    if df.empty:
        return None

    pivot = df.pivot_table(
        index="stratum",
        columns="condition",
        values="malicious_sanitised_ua_rate",
        aggfunc="mean",
        fill_value=0,
    )

    available_conditions = [c for c in SANITISATION_CONDITION_ORDER if c in pivot.columns]
    if not available_conditions:
        return None

    pivot = pivot[available_conditions].reset_index()

    outpath = outdir / "11_c5a_c5b_c5c_sanitised_ua_by_stratum.png"

    grouped_bar(
        df=pivot,
        x_col="stratum",
        y_cols=available_conditions,
        labels=available_conditions,
        title="UA on sanitised malicious rows by stratum",
        ylabel="UA rate among sanitised malicious rows",
        outpath=outpath,
        percent=True,
        footer="Compares where C5a, C5b, and C5c recover utility after sanitising malicious retrieved content.",
    )

    return {
        "title": "C5a, C5b, and C5c sanitised malicious UA by stratum",
        "description": "Shows UA among sanitised malicious rows, broken down by attack stratum for C5a, C5b, and C5c.",
        "filename": outpath.name,
    }


def chart_unit_removal_by_sanitisation_condition(sanitisation: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    """Compare unit-level removal burden for C5a and C5c.

    C5b is deterministic payload removal and normally does not have meaningful
    sentence/unit totals, so this chart includes only conditions with logged
    unit totals.
    """
    required = {"condition", "sentence_units_total_sum", "sentence_units_blocked_sum"}
    if sanitisation.empty or not required.issubset(set(sanitisation.columns)):
        return None

    df = sanitisation.copy()
    df = df[df["condition"].isin(["C5a", "C5c"])].copy()
    if df.empty:
        return None

    df["sentence_units_total_sum"] = pd.to_numeric(df["sentence_units_total_sum"], errors="coerce").fillna(0)
    df["sentence_units_blocked_sum"] = pd.to_numeric(df["sentence_units_blocked_sum"], errors="coerce").fillna(0)
    df = df[df["sentence_units_total_sum"] > 0].copy()
    if df.empty:
        return None

    df["sentence_units_kept_sum"] = (
        df["sentence_units_total_sum"] - df["sentence_units_blocked_sum"]
    ).clip(lower=0)

    df = condition_sort(df)

    outpath = outdir / "12_c5a_c5c_unit_removal_counts.png"

    grouped_bar(
        df=df,
        x_col="condition",
        y_cols=["sentence_units_blocked_sum", "sentence_units_kept_sum"],
        labels=["Units removed", "Units kept"],
        title="Unit-level removal burden in C5a and C5c",
        ylabel="Unit count",
        outpath=outpath,
        percent=False,
        footer="C5a uses independent unit-level LLM checks; C5c uses one context-aware LLM call over numbered units.",
    )

    return {
        "title": "C5a and C5c unit-level removal burden",
        "description": "Shows the number of units removed and retained by C5a and C5c.",
        "filename": outpath.name,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create charts from analyse_condition_results.py outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/analysis/c0_to_c5c"),
        help="Directory containing analysis CSV outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/charts/c0_to_c5c"),
        help="Output directory for PNG charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir: Path = args.analysis_dir
    outdir: Path = args.outdir

    ensure_outdir(outdir)

    overall = read_csv_required(analysis_dir / "overall_metrics.csv")
    metrics_by_stratum = read_csv_required(analysis_dir / "metrics_by_stratum.csv")
    score_matrix = read_csv_required(analysis_dir / "score_matrix.csv")
    detector = read_csv_optional(analysis_dir / "detector_metrics.csv")
    sanitisation = read_csv_optional(analysis_dir / "sanitisation_metrics.csv")
    if sanitisation.empty:
        sanitisation = read_csv_optional(analysis_dir / "c5b_sanitisation_metrics.csv")
    sanitisation_by_stratum = read_csv_optional(analysis_dir / "sanitisation_by_stratum.csv")
    if sanitisation_by_stratum.empty:
        sanitisation_by_stratum = read_csv_optional(analysis_dir / "c5b_sanitisation_by_stratum.csv")

    chart_entries: list[dict[str, str]] = []

    chart_entries.append(chart_overall_security_utility(overall, outdir))
    chart_entries.append(chart_asr_vs_ua_tradeoff(overall, outdir))
    chart_entries.append(chart_asr_malicious(overall, outdir))
    chart_entries.append(chart_task_success_breakdown(overall, outdir))

    detector_entry = chart_detector_metrics(detector, outdir)
    if detector_entry:
        chart_entries.append(detector_entry)

    chart_entries.append(chart_block_rates(overall, outdir))
    chart_entries.append(chart_score_matrix(score_matrix, outdir))
    chart_entries.append(chart_malicious_asr_by_stratum(metrics_by_stratum, outdir))

    san_task_entry = chart_task_success_by_stratum_sanitisation_conditions(metrics_by_stratum, outdir)
    if san_task_entry:
        chart_entries.append(san_task_entry)

    san_flow_entry = chart_sanitisation_flow_counts(sanitisation, outdir)
    if san_flow_entry:
        chart_entries.append(san_flow_entry)

    san_outcomes_entry = chart_sanitised_malicious_outcomes(sanitisation, outdir)
    if san_outcomes_entry:
        chart_entries.append(san_outcomes_entry)

    san_stratum_entry = chart_sanitised_ua_by_stratum(sanitisation_by_stratum, outdir)
    if san_stratum_entry:
        chart_entries.append(san_stratum_entry)

    unit_removal_entry = chart_unit_removal_by_sanitisation_condition(sanitisation, outdir)
    if unit_removal_entry:
        chart_entries.append(unit_removal_entry)

    write_chart_index(outdir, chart_entries)

    print("Chart generation complete")
    print("=" * 60)
    print(f"Analysis directory: {analysis_dir}")
    print(f"Chart output dir:   {outdir}")
    print()
    print("Wrote:")
    for entry in chart_entries:
        print(f"  {outdir / entry['filename']}")
    print(f"  {outdir / 'chart_index.md'}")


if __name__ == "__main__":
    main()
