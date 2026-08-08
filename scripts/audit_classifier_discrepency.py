"""

Audit C3 vs C5 detector discrepancy.

Purpose:
- Load C3, C5a, C5b and C5c final corrected CSV files.
- Use the correct detector outcome columns:
    C3  -> c3_detector_outcome
    C5a -> c5a_detector_outcome
    C5b -> c5b_detector_outcome
    C5c -> c5c_detector_outcome
- Recompute TP, FP, TN, FN, precision, recall, F1, FPR, FNR.
- Identify which malicious example_ids explain the 88 vs 87 detector-count discrepancy.
- Export CSV audit files.
- Create simple diagnostic charts.

Run from:
    MSc_Guardrails/scripts

Command:
    python audit_classifier_discrepency.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FINAL_DIR = PROJECT_ROOT / "results" / "final"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "classifier_discrepancy_audit"

FILES = {
    "C3": FINAL_DIR / "c3" / "c3_20260608_123942_review_scored_answerkey_corrected.csv",
    "C5a": FINAL_DIR / "c5a" / "c5a_20260610_113959_review_scored_answerkey_corrected.csv",
    "C5b": FINAL_DIR / "c5b" / "c5b_20260609_125117_review_scored_answerkey_corrected.csv",
    "C5c": FINAL_DIR / "c5c" / "c5c_20260611_113203_review_scored_answerkey_corrected.csv",
}

DETECTOR_COLS = {
    "C3": "c3_detector_outcome",
    "C5a": "c5a_detector_outcome",
    "C5b": "c5b_detector_outcome",
    "C5c": "c5c_detector_outcome",
}

MALICIOUS_STRATA = {
    "instruction_override",
    "data_exfiltration",
    "tool_misuse",
    "markdown_injection",
}

CONDITION_ORDER = ["C3", "C5a", "C5b", "C5c"]


# ---------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------

def die(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def parse_bool(value) -> bool:
    """Parse common boolean-like values."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    s = str(value).strip().lower()

    true_values = {
        "true", "1", "1.0", "yes", "y", "t",
        "success", "succeeded",
    }

    false_values = {
        "false", "0", "0.0", "no", "n", "f",
        "fail", "failed", "none", "nan", "",
    }

    if s in true_values:
        return True
    if s in false_values:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def is_malicious_row(row: pd.Series) -> bool:
    """
    Determine whether row is malicious.

    Preferred:
    - label column if it contains malicious/benign style labels.
    Fallback:
    - stratum column.
    """
    if "label" in row.index and not pd.isna(row["label"]):
        label = str(row["label"]).strip().lower()

        if label in {"malicious", "attack", "unsafe"}:
            return True
        if label in {"benign", "safe", "clean"}:
            return False

    if "stratum" in row.index and not pd.isna(row["stratum"]):
        return str(row["stratum"]).strip().lower() in MALICIOUS_STRATA

    raise ValueError(
        "Could not determine malicious/benign row. "
        "Expected either a usable 'label' column or 'stratum' column."
    )


def detector_outcome_to_unsafe(value) -> bool:
    """
    Convert detector outcome label to classifier unsafe decision.

    Expected detector outcome values:
        TP_malicious_blocked
        TP_malicious_detected
        FP_benign_blocked
        FN_malicious_allowed
        TN_benign_allowed

    Detector unsafe = TP or FP.
    Detector safe/allowed = TN or FN.
    """
    if pd.isna(value):
        raise ValueError("Detector outcome is missing.")

    s = str(value).strip().lower()

    if s.startswith("tp") or s.startswith("fp"):
        return True

    if s.startswith("tn") or s.startswith("fn"):
        return False

    # Fallback support for safe/unsafe strings
    if "unsafe" in s or "blocked" in s or "detected" in s:
        return True

    if "safe" in s or "allowed" in s or "pass" in s:
        return False

    raise ValueError(f"Cannot parse detector outcome: {value!r}")


def detector_outcome_short(value) -> str:
    """Return TP/FP/TN/FN from detector outcome."""
    if pd.isna(value):
        return "MISSING"

    s = str(value).strip().upper()

    if s.startswith("TP"):
        return "TP"
    if s.startswith("FP"):
        return "FP"
    if s.startswith("TN"):
        return "TN"
    if s.startswith("FN"):
        return "FN"

    return "UNKNOWN"


def load_condition(audit_condition: str, path: Path) -> pd.DataFrame:
    """Load one condition CSV and validate required columns."""
    if not path.exists():
        die(
            f"Could not find file for {audit_condition}:\n{path}\n\n"
            "Check the FILES dictionary at the top of the script."
        )

    df = pd.read_csv(path)

    required = ["example_id", "stratum", "label", DETECTOR_COLS[audit_condition]]
    missing = [c for c in required if c not in df.columns]

    if missing:
        die(
            f"{audit_condition} file is missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df["audit_condition_loaded_as"] = audit_condition
    df["source_file"] = str(path)

    print(f"Loaded {audit_condition}: {path}")
    print(f"Rows: {len(df)}")
    print(f"Detector column: {DETECTOR_COLS[audit_condition]}")
    print()

    return df


def standardise_condition(audit_condition: str, df: pd.DataFrame) -> pd.DataFrame:
    """Create a standardised audit dataframe."""
    detector_col = DETECTOR_COLS[audit_condition]

    # Important: initialise with df.index so scalar assignments fill all rows
    out = pd.DataFrame(index=df.index)

    out["audit_condition"] = audit_condition
    out["example_id"] = df["example_id"].astype(str)
    out["stratum"] = df["stratum"].astype(str)
    out["attack_type"] = df["attack_type"].astype(str) if "attack_type" in df.columns else ""
    out["label"] = df["label"].astype(str)
    out["is_malicious"] = df.apply(is_malicious_row, axis=1).astype(bool)

    out["detector_col"] = detector_col
    out["detector_outcome_raw"] = df[detector_col].astype(str)
    out["detector_outcome_short"] = df[detector_col].apply(detector_outcome_short)
    out["classifier_unsafe"] = df[detector_col].apply(detector_outcome_to_unsafe).astype(bool)

    # Final human-scored outcomes
    for col in [
        "human_final_attack_success",
        "human_final_task_success",
        "human_final_ua",
    ]:
        if col in df.columns:
            out[col] = df[col]

    # Guardrail / pipeline metadata
    for col in [
        "guardrail_decision",
        "log_guardrail_decision",
        "guardrail_reason",
        "log_guardrail_reason",
        "pipeline_action",
        "backend_called",
        "backend_context_type",
        "sanitisation_attempted",
        "sanitisation_succeeded",
        "sanitisation_fallback_to_block",
        "sanitisation_strategy",
        "sanitisation_segments_removed",
        "sentence_units_total",
        "sentence_units_blocked",
        "core_payload",
        "core_attack_goal",
        "core_user_question",
        "core_injected_context",
        "agent_output",
        "log_agent_output",
        "full_output",
        "log_full_output",
    ]:
        if col in df.columns:
            out[col] = df[col]

    return out.reset_index(drop=True)

def compute_detector_metrics(std: pd.DataFrame) -> pd.DataFrame:
    """Compute detector metrics by audit condition."""
    rows = []

    if "audit_condition" not in std.columns:
        raise RuntimeError(
            "Missing audit_condition column. The standardised dataframe was not built correctly."
        )

    print("\nDEBUG: standardised rows:", len(std))
    print("DEBUG: audit_condition counts:")
    print(std["audit_condition"].value_counts(dropna=False).to_string())

    for audit_condition in CONDITION_ORDER:
        g = std[std["audit_condition"] == audit_condition].copy()

        print(f"\nDEBUG: {audit_condition} rows found:", len(g))

        if g.empty:
            continue

        tp = int(((g["is_malicious"]) & (g["classifier_unsafe"])).sum())
        fn = int(((g["is_malicious"]) & (~g["classifier_unsafe"])).sum())
        fp = int(((~g["is_malicious"]) & (g["classifier_unsafe"])).sum())
        tn = int(((~g["is_malicious"]) & (~g["classifier_unsafe"])).sum())

        n = tp + fp + tn + fn
        malicious_n = tp + fn
        benign_n = fp + tn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        fpr = fp / benign_n if benign_n else 0.0
        fnr = fn / malicious_n if malicious_n else 0.0
        block_rate = (tp + fp) / n if n else 0.0

        rows.append({
            "condition": audit_condition,
            "n": n,
            "malicious_n": malicious_n,
            "benign_n": benign_n,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "classifier_block_rate_pct": block_rate * 100,
            "precision_pct": precision * 100,
            "recall_pct": recall * 100,
            "f1_pct": f1 * 100,
            "fpr_pct": fpr * 100,
            "fnr_pct": fnr * 100,
        })

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        raise RuntimeError(
            "Detector metrics table is empty. This means no rows matched "
            "C3/C5a/C5b/C5c in the audit_condition column."
        )

    order_map = {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    metrics["condition_order"] = metrics["condition"].map(order_map)
    metrics = metrics.sort_values("condition_order").drop(columns=["condition_order"])

    return metrics


def compare_c3_to_c5(std: pd.DataFrame) -> pd.DataFrame:
    """Find rows where C3 and C5 detector decisions differ on malicious examples."""
    c3 = std[
        (std["audit_condition"] == "C3") &
        (std["is_malicious"])
    ][
        [
            "example_id",
            "stratum",
            "attack_type",
            "detector_outcome_raw",
            "detector_outcome_short",
            "classifier_unsafe",
        ]
    ].rename(
        columns={
            "detector_outcome_raw": "C3_detector_outcome_raw",
            "detector_outcome_short": "C3_detector_outcome_short",
            "classifier_unsafe": "C3_classifier_unsafe",
        }
    )

    all_diffs = []

    for audit_condition in ["C5a", "C5b", "C5c"]:
        c5 = std[
            (std["audit_condition"] == audit_condition) &
            (std["is_malicious"])
        ][
            [
                "example_id",
                "detector_outcome_raw",
                "detector_outcome_short",
                "classifier_unsafe",
            ]
        ].rename(
            columns={
                "detector_outcome_raw": f"{audit_condition}_detector_outcome_raw",
                "detector_outcome_short": f"{audit_condition}_detector_outcome_short",
                "classifier_unsafe": f"{audit_condition}_classifier_unsafe",
            }
        )

        merged = c3.merge(c5, on="example_id", how="outer")
        merged["comparison"] = f"C3_vs_{audit_condition}"
        merged["differs"] = (
            merged["C3_classifier_unsafe"] != merged[f"{audit_condition}_classifier_unsafe"]
        )

        all_diffs.append(merged[merged["differs"]])

    if not all_diffs:
        return pd.DataFrame()

    return pd.concat(all_diffs, ignore_index=True)


def make_discrepancy_wide(std: pd.DataFrame, discrepancy_ids: list[str]) -> pd.DataFrame:
    """Create one row per discrepant example_id with detector outcomes across all conditions."""
    rows = []

    for example_id in discrepancy_ids:
        g = std[std["example_id"] == example_id].copy()

        if g.empty:
            continue

        base = g.iloc[0]

        row = {
            "example_id": example_id,
            "stratum": base.get("stratum", ""),
            "attack_type": base.get("attack_type", ""),
            "label": base.get("label", ""),
        }

        for audit_condition in CONDITION_ORDER:
            cg = g[g["audit_condition"] == audit_condition]

            if cg.empty:
                row[f"{audit_condition}_detector_outcome"] = ""
                row[f"{audit_condition}_classifier_unsafe"] = ""
                row[f"{audit_condition}_attack_success"] = ""
                row[f"{audit_condition}_task_success"] = ""
                row[f"{audit_condition}_ua"] = ""
                row[f"{audit_condition}_pipeline_action"] = ""
                continue

            r = cg.iloc[0]
            row[f"{audit_condition}_detector_outcome"] = r.get("detector_outcome_raw", "")
            row[f"{audit_condition}_classifier_unsafe"] = r.get("classifier_unsafe", "")
            row[f"{audit_condition}_attack_success"] = r.get("human_final_attack_success", "")
            row[f"{audit_condition}_task_success"] = r.get("human_final_task_success", "")
            row[f"{audit_condition}_ua"] = r.get("human_final_ua", "")
            row[f"{audit_condition}_pipeline_action"] = r.get("pipeline_action", "")

        rows.append(row)

    return pd.DataFrame(rows)


def detector_outcome_counts(std: pd.DataFrame) -> pd.DataFrame:
    """Count raw detector outcomes by audit condition."""
    counts = (
        std.groupby(["audit_condition", "detector_outcome_raw"])
        .size()
        .reset_index(name="count")
        .sort_values(["audit_condition", "detector_outcome_raw"])
    )
    return counts


def malicious_false_negatives(std: pd.DataFrame) -> pd.DataFrame:
    """List malicious rows classified safe/allowed by the detector."""
    fn = std[
        (std["is_malicious"]) &
        (~std["classifier_unsafe"])
    ].copy()

    cols = [
        "audit_condition",
        "example_id",
        "stratum",
        "attack_type",
        "detector_outcome_raw",
        "human_final_attack_success",
        "human_final_task_success",
        "human_final_ua",
        "pipeline_action",
        "backend_called",
        "core_payload",
        "core_attack_goal",
        "core_user_question",
    ]
    cols = [c for c in cols if c in fn.columns]

    return fn[cols].sort_values(["audit_condition", "example_id"])


def make_charts(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Create simple diagnostic charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = metrics.set_index("condition").loc[
        [c for c in CONDITION_ORDER if c in set(metrics["condition"])]
    ]

    # Chart 1: TP/FN stacked bar for malicious examples
    ax = plot_df[["TP", "FN"]].plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
    )
    ax.set_title("Malicious detector outcomes by condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Count")
    ax.legend(title="Outcome")
    plt.tight_layout()
    plt.savefig(output_dir / "01_tp_fn_by_condition.png", dpi=200)
    plt.close()

    # Chart 2: Recall
    ax = plot_df["recall_pct"].plot(
        kind="bar",
        figsize=(8, 5),
    )
    ax.set_title("Detector recall by condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 105)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(output_dir / "02_recall_by_condition.png", dpi=200)
    plt.close()

    # Chart 3: Classifier/block rate
    ax = plot_df["classifier_block_rate_pct"].plot(
        kind="bar",
        figsize=(8, 5),
    )
    ax.set_title("Classifier/block rate by condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Classifier/block rate (%)")
    ax.set_ylim(0, 60)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(output_dir / "03_classifier_block_rate_by_condition.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n===================================================")
    print("LOADING FILES")
    print("===================================================\n")

    loaded = {}
    for audit_condition, path in FILES.items():
        loaded[audit_condition] = load_condition(audit_condition, path)

    print("\n===================================================")
    print("STANDARDISING ROWS")
    print("===================================================\n")

    standardised = []
    for audit_condition in CONDITION_ORDER:
        std = standardise_condition(audit_condition, loaded[audit_condition])
        standardised.append(std)

        print(
            f"{audit_condition}: "
            f"malicious={int(std['is_malicious'].sum())}, "
            f"benign={int((~std['is_malicious']).sum())}, "
            f"classifier_unsafe={int(std['classifier_unsafe'].sum())}"
        )

    std_all = pd.concat(standardised, ignore_index=True)
    std_all.to_csv(OUTPUT_DIR / "standardised_classifier_rows.csv", index=False)

    print("\n===================================================")
    print("RECOMPUTED DETECTOR METRICS")
    print("===================================================\n")

    metrics = compute_detector_metrics(std_all)

    metrics_display = metrics.copy()
    pct_cols = [
        "classifier_block_rate_pct",
        "precision_pct",
        "recall_pct",
        "f1_pct",
        "fpr_pct",
        "fnr_pct",
    ]

    missing_pct_cols = [c for c in pct_cols if c not in metrics_display.columns]
    if missing_pct_cols:
        raise RuntimeError(
            f"Metrics table is missing expected percentage columns: {missing_pct_cols}\n"
            f"Actual columns are: {list(metrics_display.columns)}"
        )

    metrics_display[pct_cols] = metrics_display[pct_cols].round(1)

    print(metrics_display.to_string(index=False))
    metrics_display.to_csv(OUTPUT_DIR / "detector_metrics_recomputed.csv", index=False)

    print("\n===================================================")
    print("RAW DETECTOR OUTCOME COUNTS")
    print("===================================================\n")

    outcome_counts = detector_outcome_counts(std_all)
    print(outcome_counts.to_string(index=False))
    outcome_counts.to_csv(OUTPUT_DIR / "detector_outcome_counts.csv", index=False)

    print("\n===================================================")
    print("MALICIOUS FALSE NEGATIVES")
    print("===================================================\n")

    fn_rows = malicious_false_negatives(std_all)

    if fn_rows.empty:
        print("No malicious false negatives found.")
    else:
        preview_cols = [
            "audit_condition",
            "example_id",
            "stratum",
            "attack_type",
            "detector_outcome_raw",
        ]
        preview_cols = [c for c in preview_cols if c in fn_rows.columns]
        print(fn_rows[preview_cols].to_string(index=False))

    fn_rows.to_csv(OUTPUT_DIR / "malicious_false_negatives.csv", index=False)

    print("\n===================================================")
    print("C3 VS C5 DISCREPANCIES")
    print("===================================================\n")

    diffs = compare_c3_to_c5(std_all)

    if diffs.empty:
        print("No C3-vs-C5 detector decision discrepancies found.")
        discrepancy_ids = []
    else:
        print(diffs.to_string(index=False))
        diffs.to_csv(OUTPUT_DIR / "c3_vs_c5_discrepancies.csv", index=False)

        discrepancy_ids = sorted(diffs["example_id"].dropna().unique().tolist())
        wide = make_discrepancy_wide(std_all, discrepancy_ids)

        print("\nWide discrepancy table:")
        print(wide.to_string(index=False))

        wide.to_csv(OUTPUT_DIR / "c3_vs_c5_discrepancy_wide.csv", index=False)

        detail = std_all[std_all["example_id"].isin(discrepancy_ids)].copy()
        detail = detail.sort_values(["example_id", "audit_condition"])
        detail.to_csv(OUTPUT_DIR / "c3_vs_c5_discrepancy_details_long.csv", index=False)

    print("\n===================================================")
    print("C3 CONSISTENCY CHECK")
    print("===================================================\n")

    c3 = std_all[(std_all["audit_condition"] == "C3") & (std_all["is_malicious"])].copy()
    c3_tp = int(c3["classifier_unsafe"].sum())
    c3_fn = int((~c3["classifier_unsafe"]).sum())

    print(f"C3 malicious rows: {len(c3)}")
    print(f"C3 classifier unsafe / TP count: {c3_tp}")
    print(f"C3 classifier safe / FN count: {c3_fn}")

    if "human_final_task_success" in c3.columns:
        try:
            task_success = c3["human_final_task_success"].apply(parse_bool)
            print(f"C3 malicious task_success count: {int(task_success.sum())}")

            allowed_success = c3[(~c3["classifier_unsafe"]) & (task_success)]
            print("\nC3 malicious rows allowed by detector and task_success=True:")

            cols = [
                "example_id",
                "stratum",
                "attack_type",
                "detector_outcome_raw",
                "human_final_task_success",
                "human_final_attack_success",
                "human_final_ua",
            ]

            if allowed_success.empty:
                print("None.")
            else:
                print(allowed_success[cols].to_string(index=False))

        except Exception as exc:
            print(f"Could not parse C3 human_final_task_success: {exc}")

    print("\n===================================================")
    print("CREATING CHARTS")
    print("===================================================\n")

    make_charts(metrics, OUTPUT_DIR)
    print("Charts created.")

    print("\n===================================================")
    print("OUTPUT FILES")
    print("===================================================\n")

    print(f"Output directory:\n{OUTPUT_DIR}\n")
    print("CSV files:")
    print("- standardised_classifier_rows.csv")
    print("- detector_metrics_recomputed.csv")
    print("- detector_outcome_counts.csv")
    print("- malicious_false_negatives.csv")
    print("- c3_vs_c5_discrepancies.csv")
    print("- c3_vs_c5_discrepancy_wide.csv")
    print("- c3_vs_c5_discrepancy_details_long.csv")
    print()
    print("Charts:")
    print("- 01_tp_fn_by_condition.png")
    print("- 02_recall_by_condition.png")
    print("- 03_classifier_block_rate_by_condition.png")

    print("\n===================================================")
    print("INTERPRETATION GUIDE")
    print("===================================================\n")

    print(
        "If the recomputed metrics show C3 TP=88/FN=2 and each C5 TP=87/FN=3, "
        "then the 88 vs 87 discrepancy is real in the files. Keep the table values "
        "and include the explanatory note after Table 2c."
    )
    print()
    print(
        "If any C5 condition recomputes to TP=88/FN=2, update Table 2c and related "
        "sanitisation/recovery counts for that condition."
    )
    print()
    print(
        "The most important file is c3_vs_c5_discrepancy_wide.csv. It shows which "
        "example_id(s) differ between C3 and each C5 condition."
    )


if __name__ == "__main__":
    main()