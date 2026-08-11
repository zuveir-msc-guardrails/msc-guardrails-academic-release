python - <<'PY'
import json
from pathlib import Path

conditions = ["c0", "c1", "c2", "c3", "c5a", "c5b", "c5c"]
expected_rows = 190

print(f"{'Condition':<10} {'File':<55} {'Rows':>6} {'Errors':>8} {'Status':>10}")
print("-" * 95)

overall_ok = True

for condition in conditions:
    log_dir = Path("results/logs") / condition
    files = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        print(f"{condition:<10} {'NO JSONL FILE FOUND':<55} {'-':>6} {'-':>8} {'FAIL':>10}")
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
                parse_errors.append((lineno, str(e)))

    execution_errors = [
        r for r in rows
        if r.get("error")
    ]

    ok = (
        len(rows) == expected_rows
        and not execution_errors
        and not parse_errors
    )

    if not ok:
        overall_ok = False

    status = "PASS" if ok else "FAIL"

    print(
        f"{condition:<10} "
        f"{path.name:<55} "
        f"{len(rows):>6} "
        f"{len(execution_errors) + len(parse_errors):>8} "
        f"{status:>10}"
    )

    if execution_errors:
        print("  Execution errors:")
        for row in execution_errors[:10]:
            print(
                f"    {row.get('example_id', '?')}: "
                f"{row.get('error')}"
            )

    if parse_errors:
        print("  JSON parse errors:")
        for lineno, error in parse_errors[:10]:
            print(f"    line {lineno}: {error}")

print("-" * 95)

if overall_ok:
    print("PASS: All seven conditions produced 190 rows with no logged errors.")
else:
    print("FAIL: One or more conditions need investigation.")
PY