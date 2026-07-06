#!/usr/bin/env python3
"""
mcnemar_guardrail_pairs.py

Compare paired guardrail outcomes using McNemar's test.

Purpose:
    This script compares two or more guardrail conditions on the same examples.
    It is useful when the same benchmark rows were evaluated under multiple
    conditions, for example C3 versus C5a/C5b/C5c.

    In this dissertation, the main use is to compare malicious-row UA outcomes:
        human_final_ua = task_success AND NOT attack_success

    The script matches rows by example_id, builds a paired 2x2 table, and then
    applies McNemar's exact test to the discordant pairs.

Example:
python scripts/mcnemar_guardrail_pairs.py \
  --baseline results/final/c3/c3_20260608_123942_review_scored_answerkey_corrected.csv\
  --comparators results/final/c5a/c5a_20260610_113959_review_scored_answerkey_corrected.csv results/final/c5b/c5b_20260609_125117_review_scored_answerkey_corrected.csv results/final/c5c/c5c_20260611_113203_review_scored_answerkey_corrected.csv \
  --metric human_final_ua \
  --subset malicious \
  --out results/analysis/mcnemar/mcnemar_c3_vs_c5.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def parse_bool(x):
    """
    Convert common CSV boolean values into Python True/False.

    Review CSVs may store booleans as:
        true / false
        1 / 0
        yes / no
        y / n

    If the value is missing or cannot be recognised, return None.
    The caller later checks for None and raises an error.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def exact_mcnemar_p(b, c):
    """
    Calculate the exact two-sided McNemar p-value.

    McNemar's test uses only the discordant pairs:

        b = baseline failed, comparator succeeded
        c = baseline succeeded, comparator failed

    Under the null hypothesis, b and c are equally likely.
    Therefore, the exact test is equivalent to a two-sided binomial test
    with probability 0.5 over the discordant pairs.

    If b + c = 0, there are no discordant pairs and no evidence of a paired
    difference, so the p-value is returned as 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue


def read_review(path, metric, subset):
    """
    Read one human-reviewed condition CSV and return example_id + metric.

    Parameters:
        path:
            CSV file path for one condition, for example C3 or C5c.

        metric:
            The binary outcome to compare.
            For H5, this is usually human_final_ua.

        subset:
            Which rows to keep:
                all       = use all examples
                malicious = use malicious examples only
                benign    = use benign examples only

    The function checks:
        - the CSV has example_id and the requested metric column;
        - the requested subset can be inferred from label or stratum;
        - the metric column can be converted to booleans.
    """
    df = pd.read_csv(path)

    required = {"example_id", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if subset == "malicious":
        if "label" in df.columns:
            df = df[df["label"].astype(str).str.lower().eq("malicious")]
        elif "stratum" in df.columns:
            benign = {"benign_document", "benign_email", "benign_tool_use"}
            df = df[~df["stratum"].astype(str).str.lower().isin(benign)]
        else:
            raise ValueError(
                f"{path} has no label or stratum column, so malicious subset cannot be inferred."
            )

    elif subset == "benign":
        if "label" in df.columns:
            df = df[df["label"].astype(str).str.lower().eq("benign")]
        elif "stratum" in df.columns:
            benign = {"benign_document", "benign_email", "benign_tool_use"}
            df = df[df["stratum"].astype(str).str.lower().isin(benign)]
        else:
            raise ValueError(
                f"{path} has no label or stratum column, so benign subset cannot be inferred."
            )

    out = df[["example_id", metric]].copy()
    out[metric] = out[metric].map(parse_bool)

    if out[metric].isna().any():
        bad = out[out[metric].isna()]["example_id"].tolist()[:10]
        raise ValueError(f"{path} has non-boolean or missing {metric} values, e.g. {bad}")

    return out


def condition_name(path):
    """
    Infer a readable condition name from the input filename.

    Example:
        c3_20260608_review.csv -> C3
        c5c_20260611_review.csv -> C5C

    If no known condition token is found, use the filename stem.
    """
    name = Path(path).stem.lower()
    for c in ["c5c", "c5b", "c5a", "c3", "c2", "c1", "c0"]:
        if c in name:
            return c.upper()
    return Path(path).stem


def compare_pair(base_df, comp_df, base_name, comp_name, metric):
    """
    Compare one baseline condition against one comparator condition.

    Rows are matched by example_id. The resulting paired table is:

                          comparator success    comparator failure

        baseline success          a                    c

        baseline failure          b                    d

    For the dissertation H5 case:
        baseline   = C3
        comparator = C5a, C5b, or C5c
        metric     = human_final_ua

    The most important cells are:
        b = examples where C3 failed UA but C5 succeeded UA
        c = examples where C3 succeeded UA but C5 failed UA

    McNemar's exact test is computed from b and c only.
    """
    merged = base_df.merge(comp_df, on="example_id", suffixes=("_base", "_comp"))

    base_col = f"{metric}_base"
    comp_col = f"{metric}_comp"

    a = int(((merged[base_col] == True) & (merged[comp_col] == True)).sum())
    b = int(((merged[base_col] == False) & (merged[comp_col] == True)).sum())
    c = int(((merged[base_col] == True) & (merged[comp_col] == False)).sum())
    d = int(((merged[base_col] == False) & (merged[comp_col] == False)).sum())

    p = exact_mcnemar_p(b, c)

    return {
        "baseline": base_name,
        "comparator": comp_name,
        "metric": metric,
        "paired_n": len(merged),
        "both_success_a": a,
        "baseline_fail_comparator_success_b": b,
        "baseline_success_comparator_fail_c": c,
        "both_fail_d": d,
        "discordant_n": b + c,
        "exact_mcnemar_p": p,
        "baseline_success_rate": round((a + c) / len(merged), 4),
        "comparator_success_rate": round((a + b) / len(merged), 4),
        "difference_comparator_minus_baseline": round(((a + b) - (a + c)) / len(merged), 4),
    }


def main():
    """
    Parse command-line arguments, load the baseline and comparator CSVs,
    run paired McNemar comparisons, and save the output CSV.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--comparators", nargs="+", required=True)
    parser.add_argument("--metric", default="human_final_ua")
    parser.add_argument("--subset", choices=["all", "malicious", "benign"], default="malicious")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_name = condition_name(args.baseline)
    base_df = read_review(args.baseline, args.metric, args.subset)

    rows = []
    for comp_path in args.comparators:
        comp_name = condition_name(comp_path)
        comp_df = read_review(comp_path, args.metric, args.subset)
        rows.append(compare_pair(base_df, comp_df, base_name, comp_name, args.metric))

    result = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()