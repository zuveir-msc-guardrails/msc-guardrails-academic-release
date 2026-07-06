# src/guardrail_eval/log_writer.py

import csv
import json
from pathlib import Path

def write_csv_jsonl(rows, output_dir, run_id, fieldnames):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{run_id}.csv"
    jsonl_path = output_dir / f"{run_id}.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as json_file:
        for row in rows:
            json_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path