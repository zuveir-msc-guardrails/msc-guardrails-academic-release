#!/usr/bin/env python3

'''
Usage:

python scripts/write_by_injection_style_markdown.py \
  --analysis-dir results/analysis/c0_to_c5c
'''

from pathlib import Path
import argparse
import pandas as pd


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def safe_float_percent(value: str | float | int) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None


def build_main_text_md(outdir: Path) -> None:
    compact = read_csv(outdir / "by_injection_style_dissertation_table.csv")

    lines = []
    lines.append("# By-injection-style results for dissertation main text\n")
    lines.append(
        "This table is intended for the main Results section. It keeps only the columns "
        "needed to show whether the guardrails are robust across injection styles rather "
        "than only against obvious appended payloads.\n"
    )

    lines.append("## Suggested table caption\n")
    lines.append(
        "**Table X. Guardrail robustness by injection style, malicious examples only.** "
        "ASR is attack success rate. UA is malicious-row utility under attack, defined as "
        "`task_success AND NOT attack_success`. C5a fallbacks indicate cases where "
        "sentence-level sanitisation failed to remove a unit and reverted to full blocking.\n"
    )

    lines.append("## Main table\n")
    lines.append(df_to_md_table(compact))

    # Generate short interpretation from the compact table.
    lines.append("\n## Suggested interpretation paragraph\n")

    c5c_asr_values = [
        safe_float_percent(v)
        for v in compact.get("C5c ASR", pd.Series(dtype=str)).tolist()
    ]
    c5c_ua_values = [
        safe_float_percent(v)
        for v in compact.get("C5c UA", pd.Series(dtype=str)).tolist()
    ]
    c5a_fallbacks = [
        int(v) for v in compact.get("C5a fallbacks", pd.Series(dtype=int)).fillna(0).tolist()
    ]

    all_c5c_zero_asr = all(v == 0.0 for v in c5c_asr_values if v is not None)
    min_c5c_ua = min([v for v in c5c_ua_values if v is not None], default=None)
    total_c5a_fallbacks = sum(c5a_fallbacks)

    sentence_1 = (
        "Table X shows that C5c maintained 0% ASR across all injection styles"
        if all_c5c_zero_asr
        else "Table X shows that C5c substantially reduced ASR across injection styles"
    )

    if min_c5c_ua is not None:
        sentence_1 += f" while preserving at least {min_c5c_ua:.1f}% malicious-row UA in every style group."
    else:
        sentence_1 += " while preserving malicious-row utility under attack."

    sentence_2 = (
        f"The {total_c5a_fallbacks} C5a fallback cases show the limitation of judging "
        "sentence-like units independently: benign-looking injected tasks can appear safe "
        "when stripped of document context."
        if total_c5a_fallbacks > 0
        else "C5a produced no fallback cases in this corrected scoring run, but it remains less context-aware than C5c."
    )

    sentence_3 = (
        "The by-style breakdown therefore strengthens the main C5 finding: the final method "
        "does not merely work on naïve appended payloads; it preserves the security outcome "
        "across different injection forms while recovering the utility lost by binary blocking."
    )

    lines.append(sentence_1 + " " + sentence_2 + " " + sentence_3 + "\n")

    (outdir / "by_injection_style_main_text.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_pivot_md(outdir: Path) -> None:
    asr = read_csv(outdir / "by_injection_style_asr_pivot.csv")
    ua = read_csv(outdir / "by_injection_style_ua_pivot.csv")
    task = read_csv(outdir / "by_injection_style_task_success_pivot.csv")
    block = read_csv(outdir / "by_injection_style_block_rate_pivot.csv")
    fallback = read_csv(outdir / "by_injection_style_fallback_count_pivot.csv")

    lines = []
    lines.append("# By-injection-style pivot tables\n")
    lines.append(
        "These tables are useful for analysis and appendix material. The main dissertation "
        "should normally use `by_injection_style_main_text.md` instead of including all "
        "pivot tables.\n"
    )

    lines.append("## ASR by injection style and condition\n")
    lines.append(df_to_md_table(asr))

    lines.append("\n## UA by injection style and condition\n")
    lines.append(df_to_md_table(ua))

    lines.append("\n## Task success by injection style and condition\n")
    lines.append(df_to_md_table(task))

    lines.append("\n## Block rate by injection style and condition\n")
    lines.append(df_to_md_table(block))

    lines.append("\n## Fallback count by injection style and condition\n")
    lines.append(df_to_md_table(fallback))

    (outdir / "by_injection_style_pivots.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_appendix_md(outdir: Path) -> None:
    detailed = read_csv(outdir / "by_stratum_and_injection_style_metrics.csv")

    optional_files = {
        "ASR pivot": outdir / "by_stratum_and_injection_style_asr_pivot.csv",
        "UA pivot": outdir / "by_stratum_and_injection_style_ua_pivot.csv",
        "Block-rate pivot": outdir / "by_stratum_and_injection_style_block_rate_pivot.csv",
    }

    lines = []
    lines.append("# Appendix: within-stratum by-injection-style metrics\n")
    lines.append(
        "This appendix table gives the full breakdown by both attack stratum and injection "
        "style. It is too detailed for the main Results section but is useful as audit "
        "evidence for the claim that the final method was evaluated across multiple attack "
        "forms and task strata.\n"
    )

    lines.append("## Full within-stratum table\n")
    lines.append(df_to_md_table(detailed))

    for title, path in optional_files.items():
        if path.exists():
            lines.append(f"\n## {title}\n")
            lines.append(df_to_md_table(pd.read_csv(path)))

    (outdir / "by_injection_style_appendix.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results/analysis/c0_to_c5c")
    args = parser.parse_args()

    outdir = Path(args.analysis_dir)

    build_main_text_md(outdir)
    build_pivot_md(outdir)
    build_appendix_md(outdir)

    print("Wrote:")
    print(f"  {outdir / 'by_injection_style_main_text.md'}")
    print(f"  {outdir / 'by_injection_style_pivots.md'}")
    print(f"  {outdir / 'by_injection_style_appendix.md'}")


if __name__ == "__main__":
    main()
