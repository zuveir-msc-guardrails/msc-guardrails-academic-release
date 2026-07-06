#!/usr/bin/env python3
"""
make_time_cost_complexity_charts.py
-----------------------------------
Create dissertation-ready charts from scripts/analyse_time_cost_complexity.py.

Default inputs:
    results/analysis/time_cost_complexity/
        efficiency_summary.csv
        component_totals.csv
        pipeline_action_counts.csv
        complexity_paths.csv
        efficiency_deltas_vs_baseline.csv

Optional effectiveness inputs:
    results/analysis/c0_to_c5c/
        overall_metrics.csv
        sanitisation_metrics.csv

Outputs:
    PNG charts and chart_index.md.

Usage:
    python scripts/make_time_cost_complexity_charts.py \
      --analysis-dir results/analysis/time_cost_complexity \
      --effectiveness-dir results/analysis/c0_to_c5c_answerkey_corrected \
      --outdir results/charts/time_cost_complexity

Notes:
    - Uses matplotlib only.
    - Does not use seaborn.
    - Does not set custom colours.
    - Creates one chart per figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = ["C0", "C1", "C2", "C3", "C5a", "C5b", "C5c"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
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
    return (
        df.assign(_condition_order=df["condition"].map(order).fillna(999))
        .sort_values("_condition_order")
        .drop(columns=["_condition_order"])
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def add_bar_labels(ax: plt.Axes, suffix: str = "", decimals: int = 2) -> None:
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


def add_horizontal_labels(ax: plt.Axes, suffix: str = "", decimals: int = 2) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if pd.isna(width):
            continue
        label = f"{width:.{decimals}f}{suffix}"
        ax.annotate(
            label,
            (width, patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(4, 0),
            textcoords="offset points",
        )


def save_current_figure(outpath: Path) -> None:
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def write_chart_index(outdir: Path, entries: list[dict[str, str]]) -> None:
    lines = ["# Time, cost, and complexity chart outputs", ""]
    for entry in entries:
        lines.append(f"## {entry['title']}")
        lines.append("")
        lines.append(entry["description"])
        lines.append("")
        lines.append(f"![{entry['title']}]({entry['filename']})")
        lines.append("")
    (outdir / "chart_index.md").write_text("\n".join(lines), encoding="utf-8")


def simple_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
    label_suffix: str = "",
    label_decimals: int = 2,
    footer: str | None = None,
) -> None:
    plot_df = condition_sort(df.copy())
    values = numeric(plot_df[y_col])

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.15), 5))
    ax.bar(plot_df[x_col], values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    add_bar_labels(ax, suffix=label_suffix, decimals=label_decimals)

    if footer:
        ax.text(0, -0.18, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    outpath: Path,
    label_suffix: str = "",
    footer: str | None = None,
) -> None:
    plot_df = condition_sort(df.copy())
    x_labels = list(plot_df[x_col])
    x = range(len(x_labels))
    width = 0.8 / max(len(y_cols), 1)

    fig, ax = plt.subplots(figsize=(max(9, len(x_labels) * 1.35), 5))

    for i, (col, label) in enumerate(zip(y_cols, labels)):
        values = numeric(plot_df[col])
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.legend()

    if footer:
        ax.text(0, -0.18, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def stacked_bar(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    outpath: Path,
    footer: str | None = None,
) -> None:
    plot_df = condition_sort(df.copy())
    x_labels = list(plot_df[x_col])
    x = range(len(x_labels))

    fig, ax = plt.subplots(figsize=(max(9, len(x_labels) * 1.35), 5))

    bottom = pd.Series([0.0] * len(plot_df))
    for col, label in zip(y_cols, labels):
        values = numeric(plot_df[col])
        ax.bar(x, values, bottom=bottom, label=label)
        bottom = bottom + values

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.legend()

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
    label_suffix: str = "",
    label_decimals: int = 2,
    footer: str | None = None,
) -> None:
    plot_df = df.copy()
    plot_df[value_col] = numeric(plot_df[value_col])
    plot_df = plot_df.sort_values(value_col)

    fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.45)))
    ax.barh(plot_df[label_col], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    add_horizontal_labels(ax, suffix=label_suffix, decimals=label_decimals)

    if footer:
        ax.text(0, -0.12, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def labelled_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    x_percent: bool = False,
    y_percent: bool = False,
    footer: str | None = None,
    zoom_to_data: bool = False,
) -> None:
    """Create a labelled scatter plot with one point per condition.

    The label offsets are deliberately condition-specific because several final
    guardrail conditions cluster around high UA and near-zero ASR. This keeps
    C5a, C5b, and C5c readable without changing the underlying point positions.
    """
    plot_df = condition_sort(df.copy())
    plot_df[x_col] = numeric(plot_df[x_col])
    plot_df[y_col] = numeric(plot_df[y_col])

    x_values = plot_df[x_col] * 100.0 if x_percent else plot_df[x_col]
    y_values = plot_df[y_col] * 100.0 if y_percent else plot_df[y_col]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    # Plot one point per condition. We do not specify colours, so matplotlib
    # defaults are used.
    for _, row in plot_df.iterrows():
        x = row[x_col] * 100.0 if x_percent else row[x_col]
        y = row[y_col] * 100.0 if y_percent else row[y_col]
        ax.scatter([x], [y], s=90)

    label_offsets = {
        "C0": (8, 6),
        "C1": (8, 6),
        "C2": (8, 6),
        "C3": (8, 6),
        "C5a": (8, -18),
        "C5b": (8, 4),
        "C5c": (8, 22),
    }

    for _, row in plot_df.iterrows():
        condition = str(row["condition"])
        x = row[x_col] * 100.0 if x_percent else row[x_col]
        y = row[y_col] * 100.0 if y_percent else row[y_col]
        xytext = label_offsets.get(condition, (6, 4))
        ax.annotate(
            condition,
            (x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5} if condition in {"C5a", "C5b", "C5c"} else None,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if zoom_to_data:
        x_min = float(x_values.min())
        x_max = float(x_values.max())
        y_min = float(y_values.min())
        y_max = float(y_values.max())

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)

        ax.set_xlim(max(0, x_min - x_span * 0.15), min(105 if x_percent else x_max + x_span * 0.15, x_max + x_span * 0.18))
        ax.set_ylim(max(0, y_min - y_span * 0.20), min(105 if y_percent else y_max + y_span * 0.20, y_max + y_span * 0.25))
    else:
        if x_percent:
            ax.set_xlim(0, 105)
        if y_percent:
            ax.set_ylim(0, 105)

    ax.grid(True, linewidth=0.5, alpha=0.35)

    if footer:
        ax.text(0, -0.17, footer, transform=ax.transAxes, fontsize=9, va="top")

    save_current_figure(outpath)


def merged_efficiency_effectiveness(summary: pd.DataFrame, effectiveness: pd.DataFrame) -> pd.DataFrame:
    """Join runtime/cost metrics with human-scored effectiveness metrics."""
    if effectiveness.empty:
        return pd.DataFrame()
    df = summary.merge(effectiveness, on="condition", how="inner")
    return condition_sort(df)


# ── Chart builders ───────────────────────────────────────────────────────────

def chart_total_cost(summary: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "01_total_cost_by_condition.png"
    simple_bar(
        df=summary,
        x_col="condition",
        y_col="total_pipeline_approx_cost_usd",
        title="Total estimated cost by condition",
        ylabel="Total estimated cost, USD",
        outpath=outpath,
        label_suffix="",
        label_decimals=4,
        footer="Shows the operational cost of each condition across the full run.",
    )
    return {
        "title": "Total estimated cost by condition",
        "description": "Compares total estimated API cost across conditions.",
        "filename": outpath.name,
    }


def chart_mean_cost(summary: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "02_mean_cost_per_row.png"
    simple_bar(
        df=summary,
        x_col="condition",
        y_col="mean_pipeline_approx_cost_usd",
        title="Mean estimated cost per row",
        ylabel="Mean cost per row, USD",
        outpath=outpath,
        label_decimals=5,
        footer="Useful for comparing expected per-query cost under each guardrail architecture.",
    )
    return {
        "title": "Mean estimated cost per row",
        "description": "Compares average estimated cost per example.",
        "filename": outpath.name,
    }


def chart_mean_latency(summary: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "03_mean_and_p95_latency.png"
    grouped_bar(
        df=summary,
        x_col="condition",
        y_cols=["mean_pipeline_latency_seconds", "p95_pipeline_latency_seconds"],
        labels=["Mean latency", "P95 latency"],
        title="Mean and P95 pipeline latency",
        ylabel="Seconds",
        outpath=outpath,
        footer="P95 latency helps show tail latency introduced by guardrails and sanitisation.",
    )
    return {
        "title": "Mean and P95 pipeline latency",
        "description": "Compares average and tail latency by condition.",
        "filename": outpath.name,
    }


def chart_total_tokens(summary: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "04_total_tokens_by_condition.png"
    simple_bar(
        df=summary,
        x_col="condition",
        y_col="total_pipeline_tokens",
        title="Total token use by condition",
        ylabel="Total tokens",
        outpath=outpath,
        label_decimals=0,
        footer="Token use explains much of the cost difference between conditions.",
    )
    return {
        "title": "Total token use by condition",
        "description": "Compares total pipeline token use across the full run.",
        "filename": outpath.name,
    }


def chart_component_costs(components: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "05_component_cost_breakdown.png"
    stacked_bar(
        df=components,
        x_col="condition",
        y_cols=["guardrail_cost_usd", "sanitiser_cost_usd", "backend_cost_usd"],
        labels=["Detector", "Sanitiser", "Backend"],
        title="Cost breakdown by pipeline component",
        ylabel="Estimated cost, USD",
        outpath=outpath,
        footer="Separates detector, sanitiser, and backend costs.",
    )
    return {
        "title": "Cost breakdown by component",
        "description": "Shows how detector, sanitiser, and backend calls contribute to total cost.",
        "filename": outpath.name,
    }


def chart_component_tokens(components: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "06_component_token_breakdown.png"
    stacked_bar(
        df=components,
        x_col="condition",
        y_cols=["guardrail_tokens", "sanitiser_tokens", "backend_tokens"],
        labels=["Detector", "Sanitiser", "Backend"],
        title="Token breakdown by pipeline component",
        ylabel="Tokens",
        outpath=outpath,
        footer="Highlights the extra token burden introduced by LLM-based sanitisation.",
    )
    return {
        "title": "Token breakdown by component",
        "description": "Shows token consumption by detector, sanitiser, and backend components.",
        "filename": outpath.name,
    }


def chart_llm_calls(summary: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "07_estimated_llm_calls.png"
    grouped_bar(
        df=summary,
        x_col="condition",
        y_cols=["estimated_total_llm_calls", "mean_llm_calls_per_row"],
        labels=["Total LLM calls", "Mean LLM calls per row"],
        title="Estimated LLM calls",
        ylabel="Count",
        outpath=outpath,
        footer="C5a is expected to be high because it classifies sentence-like units independently.",
    )
    return {
        "title": "Estimated LLM calls",
        "description": "Compares the number of estimated LLM calls required by each architecture.",
        "filename": outpath.name,
    }


def chart_complexity_score(summary: pd.DataFrame, outdir: Path) -> dict[str, str]:
    outpath = outdir / "08_mean_complexity_score.png"
    simple_bar(
        df=summary,
        x_col="condition",
        y_col="mean_complexity_score",
        title="Mean architectural complexity score",
        ylabel="Complexity score",
        outpath=outpath,
        label_decimals=2,
        footer="A descriptive score representing the number and type of pipeline stages.",
    )
    return {
        "title": "Mean architectural complexity score",
        "description": "Compares architectural complexity across conditions.",
        "filename": outpath.name,
    }


def chart_pipeline_actions(actions: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if actions.empty:
        return None

    pivot = actions.pivot_table(
        index="condition",
        columns="pipeline_action",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot = condition_sort(pivot)

    action_cols = [c for c in pivot.columns if c != "condition"]
    if not action_cols:
        return None

    outpath = outdir / "09_pipeline_action_counts.png"
    stacked_bar(
        df=pivot,
        x_col="condition",
        y_cols=action_cols,
        labels=action_cols,
        title="Pipeline action counts",
        ylabel="Row count",
        outpath=outpath,
        footer="Shows how often each condition allowed, sanitised, or blocked rows.",
    )
    return {
        "title": "Pipeline action counts",
        "description": "Shows final route counts such as allow, sanitised context, and blocked fallback.",
        "filename": outpath.name,
    }


def chart_complexity_paths(paths: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if paths.empty:
        return None

    plot_df = paths.copy()
    plot_df["label"] = plot_df["condition"].astype(str) + ": " + plot_df["complexity_path"].astype(str)

    outpath = outdir / "10_complexity_path_counts.png"
    horizontal_bar(
        df=plot_df,
        label_col="label",
        value_col="count",
        title="Complexity path counts",
        xlabel="Row count",
        outpath=outpath,
        label_decimals=0,
        footer="Shows which architectural path each row followed.",
    )
    return {
        "title": "Complexity path counts",
        "description": "Breaks each condition down into operational paths, such as classifier allow, sanitised backend, or fallback block.",
        "filename": outpath.name,
    }


def chart_asr_vs_ua(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"ua_rate_all_rows", "asr_malicious"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "11_asr_vs_overall_ua.png"

    labelled_scatter(
        df=df,
        x_col="ua_rate_all_rows",
        y_col="asr_malicious",
        title="Security-utility trade-off: ASR versus UA",
        xlabel="Overall utility under attack (UA, %)",
        ylabel="Targeted ASR on malicious rows (%)",
        outpath=outpath,
        x_percent=True,
        y_percent=True,
        footer="Best region is bottom-right: high UA and low attack success.",
    )

    return {
        "title": "ASR versus overall UA",
        "description": "Plots targeted malicious ASR against overall utility under attack. Best trade-off is bottom-right.",
        "filename": outpath.name,
    }


def chart_asr_vs_ua_zoomed(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"ua_rate_all_rows", "asr_malicious"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "12_asr_vs_overall_ua_zoomed.png"

    labelled_scatter(
        df=df,
        x_col="ua_rate_all_rows",
        y_col="asr_malicious",
        title="Security-utility trade-off: ASR versus UA (zoomed)",
        xlabel="Overall utility under attack (UA, %)",
        ylabel="Targeted ASR on malicious rows (%)",
        outpath=outpath,
        x_percent=True,
        y_percent=True,
        zoom_to_data=True,
        footer="Zoomed to the observed range so C5a, C5b, and C5c are distinguishable.",
    )

    return {
        "title": "ASR versus overall UA, zoomed",
        "description": "Zoomed security-utility trade-off chart to separate the high-UA, low-ASR conditions.",
        "filename": outpath.name,
    }


def chart_cost_vs_utility(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"mean_pipeline_approx_cost_usd", "ua_rate_all_rows"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "13_cost_vs_overall_ua.png"

    labelled_scatter(
        df=df,
        x_col="mean_pipeline_approx_cost_usd",
        y_col="ua_rate_all_rows",
        title="Cost versus overall utility under attack",
        xlabel="Mean estimated cost per row, USD",
        ylabel="Overall UA rate (%)",
        outpath=outpath,
        y_percent=True,
        footer="Best region is top-left: high UA at low cost.",
    )

    return {
        "title": "Cost versus overall UA",
        "description": "Plots mean cost per row against overall utility-under-attack rate. Useful for arguing cost-effectiveness.",
        "filename": outpath.name,
    }


def chart_latency_vs_utility(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"mean_pipeline_latency_seconds", "ua_rate_all_rows"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "14_latency_vs_overall_ua.png"

    labelled_scatter(
        df=df,
        x_col="mean_pipeline_latency_seconds",
        y_col="ua_rate_all_rows",
        title="Latency versus overall utility under attack",
        xlabel="Mean pipeline latency per row (seconds)",
        ylabel="Overall UA rate (%)",
        outpath=outpath,
        y_percent=True,
        footer="Best region is top-left: high UA at low latency.",
    )

    return {
        "title": "Latency versus overall UA",
        "description": "Plots mean pipeline latency per row against overall utility-under-attack rate.",
        "filename": outpath.name,
    }


def chart_complexity_vs_utility(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"mean_complexity_score", "ua_rate_all_rows"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "15_complexity_vs_overall_ua.png"

    labelled_scatter(
        df=df,
        x_col="mean_complexity_score",
        y_col="ua_rate_all_rows",
        title="Architectural complexity versus overall UA",
        xlabel="Mean architectural complexity score",
        ylabel="Overall UA rate (%)",
        outpath=outpath,
        y_percent=True,
        footer="Best region is top-left: high UA with lower architectural complexity.",
    )

    return {
        "title": "Complexity versus overall UA",
        "description": "Plots descriptive architectural complexity score against overall utility-under-attack rate.",
        "filename": outpath.name,
    }


def chart_llm_calls_vs_utility(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"mean_llm_calls_per_row", "ua_rate_all_rows"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "16_llm_calls_vs_overall_ua.png"

    labelled_scatter(
        df=df,
        x_col="mean_llm_calls_per_row",
        y_col="ua_rate_all_rows",
        title="LLM calls versus overall utility under attack",
        xlabel="Mean LLM calls per row",
        ylabel="Overall UA rate (%)",
        outpath=outpath,
        y_percent=True,
        footer="Best region is top-left: high UA with fewer model calls.",
    )

    return {
        "title": "LLM calls versus overall UA",
        "description": "Plots mean LLM calls per row against overall utility-under-attack rate.",
        "filename": outpath.name,
    }


def chart_cost_vs_asr(summary: pd.DataFrame, effectiveness: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    df = merged_efficiency_effectiveness(summary, effectiveness)
    required = {"mean_pipeline_approx_cost_usd", "asr_malicious"}
    if df.empty or not required.issubset(set(df.columns)):
        return None

    outpath = outdir / "17_cost_vs_malicious_asr.png"

    labelled_scatter(
        df=df,
        x_col="mean_pipeline_approx_cost_usd",
        y_col="asr_malicious",
        title="Cost versus targeted ASR",
        xlabel="Mean estimated cost per row, USD",
        ylabel="ASR on malicious rows (%)",
        outpath=outpath,
        y_percent=True,
        footer="Best region is bottom-left: low cost and low attack success.",
    )

    return {
        "title": "Cost versus targeted ASR",
        "description": "Plots mean cost per row against targeted attack success rate. Lower ASR is better.",
        "filename": outpath.name,
    }


def chart_sanitiser_cost(sanitisation: pd.DataFrame, outdir: Path) -> dict[str, str] | None:
    if sanitisation.empty or "sentence_guardrail_approx_cost_usd" not in sanitisation.columns:
        return None

    df = sanitisation.copy()
    df = df[df["condition"].isin(["C5a", "C5b", "C5c"])].copy()
    if df.empty:
        return None

    outpath = outdir / "18_sanitiser_cost_by_condition.png"
    simple_bar(
        df=df,
        x_col="condition",
        y_col="sentence_guardrail_approx_cost_usd",
        title="Sanitiser-stage cost",
        ylabel="Estimated sanitiser cost, USD",
        outpath=outpath,
        label_decimals=5,
        footer="C5a uses independent unit-level LLM checks; C5b is deterministic; C5c uses one context-aware LLM call per sanitised row.",
    )

    return {
        "title": "Sanitiser-stage cost",
        "description": "Compares additional sanitiser-stage cost for C5a, C5b, and C5c.",
        "filename": outpath.name,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create time/cost/complexity charts."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/analysis/time_cost_complexity"),
        help="Directory containing analyse_time_cost_complexity.py outputs.",
    )
    parser.add_argument(
        "--effectiveness-dir",
        type=Path,
        default=Path("results/analysis/c0_to_c5c"),
        help="Optional directory containing overall_metrics.csv and sanitisation_metrics.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/charts/time_cost_complexity"),
        help="Output directory for PNG charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir: Path = args.analysis_dir
    effectiveness_dir: Path = args.effectiveness_dir
    outdir: Path = args.outdir

    ensure_outdir(outdir)

    summary = read_csv_required(analysis_dir / "efficiency_summary.csv")
    components = read_csv_required(analysis_dir / "component_totals.csv")
    actions = read_csv_optional(analysis_dir / "pipeline_action_counts.csv")
    paths = read_csv_optional(analysis_dir / "complexity_paths.csv")

    effectiveness = read_csv_optional(effectiveness_dir / "overall_metrics.csv")
    sanitisation = read_csv_optional(effectiveness_dir / "sanitisation_metrics.csv")

    chart_entries: list[dict[str, str]] = []

    chart_entries.append(chart_total_cost(summary, outdir))
    chart_entries.append(chart_mean_cost(summary, outdir))
    chart_entries.append(chart_mean_latency(summary, outdir))
    chart_entries.append(chart_total_tokens(summary, outdir))
    chart_entries.append(chart_component_costs(components, outdir))
    chart_entries.append(chart_component_tokens(components, outdir))
    chart_entries.append(chart_llm_calls(summary, outdir))
    chart_entries.append(chart_complexity_score(summary, outdir))

    entry = chart_pipeline_actions(actions, outdir)
    if entry:
        chart_entries.append(entry)

    entry = chart_complexity_paths(paths, outdir)
    if entry:
        chart_entries.append(entry)

    for scatter_builder in [
        chart_asr_vs_ua,
        chart_asr_vs_ua_zoomed,
        chart_cost_vs_utility,
        chart_latency_vs_utility,
        chart_complexity_vs_utility,
        chart_llm_calls_vs_utility,
        chart_cost_vs_asr,
    ]:
        entry = scatter_builder(summary, effectiveness, outdir)
        if entry:
            chart_entries.append(entry)

    entry = chart_sanitiser_cost(sanitisation, outdir)
    if entry:
        chart_entries.append(entry)

    write_chart_index(outdir, chart_entries)

    print("Time/cost/complexity chart generation complete")
    print("=" * 60)
    print(f"Analysis directory:      {analysis_dir}")
    print(f"Effectiveness directory: {effectiveness_dir}")
    print(f"Chart output directory:  {outdir}")
    print()
    print("Wrote:")
    for entry in chart_entries:
        print(f"  {outdir / entry['filename']}")
    print(f"  {outdir / 'chart_index.md'}")


if __name__ == "__main__":
    main()
