# src/guardrail_eval/log_writer.py

"""
Shared CSV/JSONL writer for experiment outputs.

Each condition runner builds schema-stable rows and passes them here for
serialisation. This module is condition agnostic and is responsible only for
writing already-built rows in two formats:

- CSV for manual review, spreadsheet checks, and dissertation tables.
- JSONL for programmatic inspection and preservation of structured row data.

Keeping output writing centralised avoids each condition implementing slightly
different CSV/JSONL behaviour.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def write_csv_jsonl(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    run_id: str,
    fieldnames: Sequence[str],
) -> tuple[Path, Path]:
    """
    Write one completed experiment run to both CSV and JSONL.

    Args:
        rows:
            Schema-stable row dictionaries produced by a condition runner.
            The row builders are responsible for ensuring that every expected
            output field exists, with optional/not-yet-reviewed fields left
            blank where appropriate.

        output_dir:
            Directory where the run files should be written. It is created if
            it does not already exist.

        run_id:
            File stem used for both output files, for example
            ``c5c_20260706_101245``.

        fieldnames:
            Ordered CSV schema from ``guardrail_eval.schemas``. This preserves
            column order across repeated runs and across review/scoring scripts.

    Returns:
        A pair of ``Path`` objects: ``(csv_path, jsonl_path)``.

    Important behaviour:
        ``csv.DictWriter`` is deliberately allowed to fail if a row contains an
        unexpected extra field. That is useful during refactoring because it
        exposes schema drift rather than silently dropping data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{run_id}.csv"
    jsonl_path = output_dir / f"{run_id}.jsonl"

    # CSV is the review-facing format. The explicit field order makes outputs
    # easier to compare across conditions and reruns.
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    # JSONL is the machine-facing companion format. It records the same row
    # dictionaries one-per-line and keeps non-ASCII text readable.
    with jsonl_path.open("w", encoding="utf-8") as json_file:
        for row in rows:
            json_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path
