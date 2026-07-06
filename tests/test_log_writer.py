import csv
import json

from guardrail_eval.log_writer import write_csv_jsonl


def test_write_csv_jsonl_writes_same_rows(tmp_path):
    rows = [
        {"example_id": "A", "condition": "C0"},
        {"example_id": "B", "condition": "C0"},
    ]
    fieldnames = ["example_id", "condition"]

    csv_path, jsonl_path = write_csv_jsonl(
        rows=rows,
        output_dir=tmp_path,
        run_id="test_run",
        fieldnames=fieldnames,
    )

    assert csv_path.exists()
    assert jsonl_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    jsonl_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]

    assert reader.fieldnames == fieldnames
    assert [r["example_id"] for r in csv_rows] == ["A", "B"]
    assert [r["example_id"] for r in jsonl_rows] == ["A", "B"]