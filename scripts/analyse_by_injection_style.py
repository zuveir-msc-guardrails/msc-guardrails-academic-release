#!/usr/bin/env python3

'''
Usage:

python scripts/analyse_by_injection_style.py \
  --outdir results/analysis/c0_to_c5c

'''

from pathlib import Path
import argparse
import pandas as pd


CONDITION_FILES = {
    "C0": "results/final/c0/c0_20260605_124503_review_scored_answerkey_corrected.csv",
    "C1": "results/final/c1/c1_20260607_110936_review_scored_answerkey_corrected.csv",
    "C2": "results/final/c2/c2_20260607_172130_review_scored_answerkey_corrected.csv",
    "C3": "results/final/c3/c3_20260608_123942_review_scored_answerkey_corrected.csv",
    "C5a": "results/final/c5a/c5a_20260610_113959_review_scored_answerkey_corrected.csv",
    "C5b": "results/final/c5b/c5b_20260609_125117_review_scored_answerkey_corrected.csv",
    "C5c": "results/final/c5c/c5c_20260611_113203_review_scored_answerkey_corrected.csv",
}

CONDITION_ORDER = ["C0", "C1", "C2", "C3", "C5a", "C5b", "C5c"]

STYLE_ORDER = ["naive", "ignore", "escape_separation", "fake_completion"]
STYLE_LABELS = {
    "naive": "Naive",
    "ignore": "Ignore",
    "escape_separation": "Escape-Separation",
    "fake_completion": "Fake Completion",
}
STYLE_DISPLAY_ORDER = ["Naive", "Ignore", "Escape-Separation", "Fake Completion"]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalise_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def infer_style_from_id(example_id: str) -> str | None:
    x = str(example_id).upper()

    if "NAIVE" in x:
        return "naive"
    if "IGNORE" in x:
        return "ignore"
    if "ESC" in x or "ESCAPE" in x:
        return "escape_separation"
    if "FAKE" in x or "FC" in x or "FAKE_COMPLETION" in x:
        return "fake_completion"

    return None


def normalise_style(value: str) -> str | None:
    x = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "naive": "naive",
        "ignore": "ignore",
        "ignore_previous": "ignore",
        "escape": "escape_separation",
        "escape_separation": "escape_separation",
        "escape_separated": "escape_separation",
        "fake": "fake_completion",
        "fake_completion": "fake_completion",
        "fakecompletion": "fake_completion",
    }

    return aliases.get(x)


def pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return round((num / den) * 100, 1)


def fmt_pct(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}%"


def fmt_int(value) -> str:
    if pd.isna(value):
        return ""
    return str(int(value))


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def ordered_pivot(
    df: pd.DataFrame,
    index_col: str,
    metric_col: str,
    output_path: Path,
    index_order: list[str] | None = None,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=index_col,
        columns="condition",
        values=metric_col,
        aggfunc="first",
    )

    cols = [c for c in CONDITION_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    if index_order is not None:
        existing = [x for x in index_order if x in pivot.index]
        pivot = pivot.loc[existing]

    pivot = pivot.reset_index()
    pivot.to_csv(output_path, index=False)
    return pivot


def load_condition(condition: str, path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {condition} file: {p}")

    df = pd.read_csv(p)

    id_col = find_col(df, ["example_id", "id", "record_id", "data_record_id"])
    label_col = find_col(df, ["label", "gold_label", "is_malicious"])
    stratum_col = find_col(df, ["stratum", "attack_stratum", "category"])
    style_col = find_col(df, ["injection_style", "attack_style", "style"])

    attack_col = find_col(df, ["human_final_attack_success", "attack_success"])
    task_col = find_col(df, ["human_final_task_success", "task_success"])
    ua_col = find_col(df, ["human_final_ua", "utility_under_attack", "ua"])

    decision_col = find_col(df, ["guardrail_decision", "detector_decision"])
    action_col = find_col(df, ["pipeline_action"])

    required = {
        "id": id_col,
        "attack_success": attack_col,
        "task_success": task_col,
        "ua": ua_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"{condition}: missing required columns {missing}. "
            f"Columns are: {list(df.columns)}"
        )

    work = df.copy()
    work["condition"] = condition

    # Malicious filter.
    if label_col:
        label_s = work[label_col].astype(str).str.strip().str.lower()
        malicious = label_s.isin(["malicious", "attack", "injected", "true", "1"])
    else:
        # Fall back to rows with recognisable injection style in ID.
        malicious = work[id_col].astype(str).str.upper().str.contains(
            "NAIVE|IGNORE|ESC|FAKE|FC",
            regex=True,
            na=False,
        )

    work = work.loc[malicious].copy()

    # Normalise injection style.
    if style_col:
        work["injection_style_norm"] = work[style_col].map(normalise_style)
    else:
        work["injection_style_norm"] = None

    missing_style = work["injection_style_norm"].isna()
    if missing_style.any():
        work.loc[missing_style, "injection_style_norm"] = work.loc[
            missing_style, id_col
        ].map(infer_style_from_id)

    work = work[work["injection_style_norm"].isin(STYLE_ORDER)].copy()

    work["attack_success_bool"] = normalise_bool(work[attack_col])
    work["task_success_bool"] = normalise_bool(work[task_col])
    work["ua_bool"] = normalise_bool(work[ua_col])

    if decision_col:
        work["guardrail_block_bool"] = work[decision_col].astype(str).str.lower().str.contains(
            "block|deny|reject",
            regex=True,
            na=False,
        )
    elif action_col:
        work["guardrail_block_bool"] = work[action_col].astype(str).str.lower().str.contains(
            "block|fallback",
            regex=True,
            na=False,
        )
    else:
        work["guardrail_block_bool"] = False

    if action_col:
        work["blocked_fallback_bool"] = (
            work[action_col].astype(str).str.lower().eq("blocked_fallback")
        )
    else:
        work["blocked_fallback_bool"] = False

    if stratum_col:
        work["stratum_norm"] = work[stratum_col].astype(str)
    else:
        work["stratum_norm"] = "all_malicious"

    return work


def build_by_style(full: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (condition, style), g in full.groupby(
        ["condition", "injection_style_norm"],
        sort=False,
    ):
        n = len(g)
        rows.append(
            {
                "condition": condition,
                "injection_style": STYLE_LABELS.get(style, style),
                "n": n,
                "ASR_%": pct(g["attack_success_bool"].sum(), n),
                "task_success_%": pct(g["task_success_bool"].sum(), n),
                "UA_%": pct(g["ua_bool"].sum(), n),
                "block_rate_%": pct(g["guardrail_block_bool"].sum(), n),
                "fallback_count": int(g["blocked_fallback_bool"].sum()),
            }
        )

    by_style = pd.DataFrame(rows)

    by_style["style_order"] = by_style["injection_style"].map(
        {
            "Naive": 0,
            "Ignore": 1,
            "Escape-Separation": 2,
            "Fake Completion": 3,
        }
    )
    by_style["condition_order"] = by_style["condition"].map(
        {c: i for i, c in enumerate(CONDITION_ORDER)}
    )

    by_style = by_style.sort_values(["condition_order", "style_order"]).drop(
        columns=["condition_order", "style_order"]
    )

    return by_style


def build_by_stratum_style(full: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (condition, stratum, style), g in full.groupby(
        ["condition", "stratum_norm", "injection_style_norm"],
        sort=False,
    ):
        n = len(g)
        rows.append(
            {
                "condition": condition,
                "stratum": stratum,
                "injection_style": STYLE_LABELS.get(style, style),
                "n": n,
                "ASR_%": pct(g["attack_success_bool"].sum(), n),
                "task_success_%": pct(g["task_success_bool"].sum(), n),
                "UA_%": pct(g["ua_bool"].sum(), n),
                "block_rate_%": pct(g["guardrail_block_bool"].sum(), n),
                "fallback_count": int(g["blocked_fallback_bool"].sum()),
            }
        )

    return pd.DataFrame(rows)


def build_dissertation_table(by_style: pd.DataFrame) -> pd.DataFrame:
    asr = by_style.pivot(index="injection_style", columns="condition", values="ASR_%")
    ua = by_style.pivot(index="injection_style", columns="condition", values="UA_%")
    fallback = by_style.pivot(
        index="injection_style",
        columns="condition",
        values="fallback_count",
    )
    n_by_style = by_style.groupby("injection_style")["n"].max()

    rows = []
    for style in STYLE_DISPLAY_ORDER:
        rows.append(
            {
                "Injection style": style,
                "n": int(n_by_style.get(style, 0)),
                "C0 ASR": fmt_pct(asr.loc[style, "C0"])
                if style in asr.index and "C0" in asr.columns
                else "",
                "C2 ASR": fmt_pct(asr.loc[style, "C2"])
                if style in asr.index and "C2" in asr.columns
                else "",
                "C3 ASR": fmt_pct(asr.loc[style, "C3"])
                if style in asr.index and "C3" in asr.columns
                else "",
                "C5a fallbacks": fmt_int(fallback.loc[style, "C5a"])
                if style in fallback.index and "C5a" in fallback.columns
                else "",
                "C5c ASR": fmt_pct(asr.loc[style, "C5c"])
                if style in asr.index and "C5c" in asr.columns
                else "",
                "C5c UA": fmt_pct(ua.loc[style, "C5c"])
                if style in ua.index and "C5c" in ua.columns
                else "",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/analysis/c0_to_c5c")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for condition, path in CONDITION_FILES.items():
        all_rows.append(load_condition(condition, path))

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(outdir / "by_injection_style_long_rows.csv", index=False)

    by_style = build_by_style(full)
    by_style.to_csv(outdir / "by_injection_style_metrics.csv", index=False)

    by_stratum_style = build_by_stratum_style(full)
    by_stratum_style.to_csv(
        outdir / "by_stratum_and_injection_style_metrics.csv",
        index=False,
    )

    dissertation_style_table = build_dissertation_table(by_style)
    dissertation_style_table.to_csv(
        outdir / "by_injection_style_dissertation_table.csv",
        index=False,
    )

    # Aggregate by-style pivots.
    by_style_asr_pivot = ordered_pivot(
        by_style,
        index_col="injection_style",
        metric_col="ASR_%",
        output_path=outdir / "by_injection_style_asr_pivot.csv",
        index_order=STYLE_DISPLAY_ORDER,
    )

    by_style_ua_pivot = ordered_pivot(
        by_style,
        index_col="injection_style",
        metric_col="UA_%",
        output_path=outdir / "by_injection_style_ua_pivot.csv",
        index_order=STYLE_DISPLAY_ORDER,
    )

    by_style_task_pivot = ordered_pivot(
        by_style,
        index_col="injection_style",
        metric_col="task_success_%",
        output_path=outdir / "by_injection_style_task_success_pivot.csv",
        index_order=STYLE_DISPLAY_ORDER,
    )

    by_style_block_pivot = ordered_pivot(
        by_style,
        index_col="injection_style",
        metric_col="block_rate_%",
        output_path=outdir / "by_injection_style_block_rate_pivot.csv",
        index_order=STYLE_DISPLAY_ORDER,
    )

    by_style_fallback_pivot = ordered_pivot(
        by_style,
        index_col="injection_style",
        metric_col="fallback_count",
        output_path=outdir / "by_injection_style_fallback_count_pivot.csv",
        index_order=STYLE_DISPLAY_ORDER,
    )

    # Within-stratum pivots.
    by_stratum_style = by_stratum_style.copy()
    by_stratum_style["stratum_style"] = (
        by_stratum_style["stratum"].astype(str)
        + " | "
        + by_stratum_style["injection_style"].astype(str)
    )

    by_stratum_style_asr_pivot = ordered_pivot(
        by_stratum_style,
        index_col="stratum_style",
        metric_col="ASR_%",
        output_path=outdir / "by_stratum_and_injection_style_asr_pivot.csv",
    )

    by_stratum_style_ua_pivot = ordered_pivot(
        by_stratum_style,
        index_col="stratum_style",
        metric_col="UA_%",
        output_path=outdir / "by_stratum_and_injection_style_ua_pivot.csv",
    )

    by_stratum_style_block_pivot = ordered_pivot(
        by_stratum_style,
        index_col="stratum_style",
        metric_col="block_rate_%",
        output_path=outdir / "by_stratum_and_injection_style_block_rate_pivot.csv",
    )

    # Markdown report.
    md = []
    md.append("# By-injection-style metrics\n")
    md.append("Malicious examples only. UA = task_success AND NOT attack_success.\n")

    md.append("\n\n## Dissertation compact by-style table\n")
    md.append(df_to_md_table(dissertation_style_table))

    md.append("\n\n## Full by-style metrics\n")
    md.append(df_to_md_table(by_style))

    md.append("\n\n## ASR pivot by injection style\n")
    md.append(df_to_md_table(by_style_asr_pivot))

    md.append("\n\n## UA pivot by injection style\n")
    md.append(df_to_md_table(by_style_ua_pivot))

    md.append("\n\n## Task-success pivot by injection style\n")
    md.append(df_to_md_table(by_style_task_pivot))

    md.append("\n\n## Block-rate pivot by injection style\n")
    md.append(df_to_md_table(by_style_block_pivot))

    md.append("\n\n## Fallback-count pivot by injection style\n")
    md.append(df_to_md_table(by_style_fallback_pivot))

    md.append("\n\n## Within-stratum by-injection-style metrics\n")
    md.append(df_to_md_table(by_stratum_style))

    md.append("\n\n## Within-stratum ASR pivot\n")
    md.append(df_to_md_table(by_stratum_style_asr_pivot))

    md.append("\n\n## Within-stratum UA pivot\n")
    md.append(df_to_md_table(by_stratum_style_ua_pivot))

    md.append("\n\n## Within-stratum block-rate pivot\n")
    md.append(df_to_md_table(by_stratum_style_block_pivot))

    md.append("\n")

    (outdir / "by_injection_style_metrics.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    outputs = [
        "by_injection_style_long_rows.csv",
        "by_injection_style_metrics.csv",
        "by_injection_style_dissertation_table.csv",
        "by_injection_style_asr_pivot.csv",
        "by_injection_style_ua_pivot.csv",
        "by_injection_style_task_success_pivot.csv",
        "by_injection_style_block_rate_pivot.csv",
        "by_injection_style_fallback_count_pivot.csv",
        "by_stratum_and_injection_style_metrics.csv",
        "by_stratum_and_injection_style_asr_pivot.csv",
        "by_stratum_and_injection_style_ua_pivot.csv",
        "by_stratum_and_injection_style_block_rate_pivot.csv",
        "by_injection_style_metrics.md",
    ]

    print("Wrote:")
    for name in outputs:
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()