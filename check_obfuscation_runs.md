python - <<'PY'
import json
from pathlib import Path

conditions = ["c1", "c2", "c3", "c5a", "c5b", "c5c"]
expected_rows = 24

print(
    f"{'Condition':<10} {'Rows':>6} {'Errors':>8} "
    f"{'File':<45} {'Status':>8}"
)
print("-" * 85)

overall_ok = True

for condition in conditions:
    log_dir = Path("results/obfuscation/logs") / condition

    files = sorted(
        log_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        print(
            f"{condition:<10} {'-':>6} {'-':>8} "
            f"{'NO JSONL FILE FOUND':<45} {'FAIL':>8}"
        )
        overall_ok = False
        continue

    path = files[0]

    rows = []
    parse_errors = []

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                parse_errors.append(
                    f"JSON line {lineno}: {e}"
                )

    row_errors = []

    for row in rows:
        example_id = row.get("example_id", "?")

        # Main backend / combined pipeline error.
        if row.get("error"):
            row_errors.append(
                f"{example_id}: error = {row['error']}"
            )

        # C3/C5 document classifier error.
        if row.get("guardrail_error"):
            row_errors.append(
                f"{example_id}: guardrail_error = "
                f"{row['guardrail_error']}"
            )

        # C5a/C5c unit-level sanitiser errors, where present.
        value = row.get("sentence_guardrail_errors")

        if value not in (None, "", [], {}, "[]", "{}"):
            row_errors.append(
                f"{example_id}: sentence_guardrail_errors = {value}"
            )

    total_errors = len(parse_errors) + len(row_errors)

    ok = (
        len(rows) == expected_rows
        and total_errors == 0
    )

    if not ok:
        overall_ok = False

    print(
        f"{condition:<10} "
        f"{len(rows):>6} "
        f"{total_errors:>8} "
        f"{path.name:<45} "
        f"{'PASS' if ok else 'FAIL':>8}"
    )

    for error in parse_errors:
        print(f"    {error}")

    for error in row_errors[:20]:
        print(f"    {error}")

    if len(row_errors) > 20:
        print(f"    ... {len(row_errors) - 20} more errors")

print("-" * 85)

if overall_ok:
    print(
        "PASS: All six obfuscation conditions produced "
        "24 rows with no logged API/parser errors."
    )
else:
    print(
        "FAIL: One or more obfuscation conditions require investigation."
    )
PY