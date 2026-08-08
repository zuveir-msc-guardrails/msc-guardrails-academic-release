# Archived benchmark version

`core_pre_answer_corrections_20260605.jsonl` is an earlier 190-example
benchmark snapshot retained for provenance.

It contains the same 190 example IDs as the final benchmark. Thirteen rows
differ only in the `expected_safe_answer` field.

The authoritative frozen benchmark used by the academic release is:

    data/core/core.jsonl

Its SHA-256 is:

    1dbcac787550f78dc686718b6603d36117b7967435a2af45d3669efd29a644c1

This hash is recorded in `data/core/frozen_ids.json` and can be verified with:

    PYTHONPATH=src:. python3 scripts/freeze_ids.py --verify
