"""
freeze_example_ids.py
---------------------
Freezes the stable example IDs for the dissertation core dataset before
any experiment is run.

Two modes:

    freeze  (default)
        Reads data/core/core.jsonl, extracts all example_id values,
        writes them to data/core/frozen_ids.json with a timestamp and
        SHA-256 hash of data/core/core.jsonl, then prints a summary. Run this ONCE before the first experiment run.
        Commit frozen_ids.json to git immediately after.

    verify
        Reads data/core/core.jsonl and data/core/frozen_ids.json and
        checks that the current example IDs and core.jsonl SHA-256 hash exactly match the frozen record.
        Run this before every experiment run to confirm ID stability.
        Raises with a clear error if any ID is added, removed, changed, or if the dataset content hash has drifted.

Usage:
    python scripts/freeze_example_ids.py           # freeze mode
    python scripts/freeze_example_ids.py --verify  # verify mode when ids have been frozen

NFR2 requirement (pre-experiment specification):
    All 190 core examples must receive stable unique IDs before any
    experiment is run. IDs must not change between conditions.

The frozen_ids.json file is the audit trail that proves IDs were locked
before experiments began. The git commit timestamp establishes pre-run
status. Do not regenerate this file after experiments have started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# The dataset schema uses the canonical field name `example_id`.
# In this script, those values are treated as stable data record identifiers:
# one immutable ID per dataset row before experiments begin.
CORE_JSONL    = Path("data/core/core.jsonl")
FROZEN_IDS    = Path("data/core/frozen_ids.json")
EXPECTED_TOTAL = 190


def load_ids(path: Path) -> list[str]:
    """
    Load and return sorted values from the canonical `example_id` field.

    These values are the stable data record identifiers used to lock the
    dataset before experiments begin.

    Important implementation detail:
    - Do not filter out missing or empty IDs here.
    - Missing/empty IDs are represented as "" so the validation step can
      detect and report them explicitly.
    - If we filtered them out at load time, the later empty-ID validation
      would never see them.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    ids = [example.get("example_id", "") for example in examples]
    return sorted(ids)


def sha256_file(path: Path) -> str:
    """
    Return a SHA-256 hash of the exact core dataset file.

    The frozen ID list proves that example IDs did not drift.
    The dataset hash is a stronger audit signal: it also changes if any
    context, payload, expected answer, or metadata field changes while the
    example IDs stay the same.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(args: argparse.Namespace) -> None:
    """
    Read core.jsonl, extract IDs, write frozen_ids.json.
    Raises if frozen_ids.json already exists and --force is not set.
    """
    if FROZEN_IDS.exists() and not args.force:
        print(
            f"ERROR: {FROZEN_IDS} already exists.\n"
            "IDs are already frozen. Run --verify to check stability.\n"
            "If you genuinely need to re-freeze (e.g. dataset was rebuilt "
            "before any experiment ran), use --force. Document this in the "
            "methodology chapter if any experiment had already run."
        )
        sys.exit(1)

    print(f"Reading {CORE_JSONL}...")
    ids = load_ids(CORE_JSONL)

    # Validate before freezing
    if len(ids) != EXPECTED_TOTAL:
        print(
            f"ERROR: Expected {EXPECTED_TOTAL} examples, found {len(ids)}.\n"
            "Run the merge script first and confirm it passes validation."
        )
        sys.exit(1)

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        print(f"ERROR: Duplicate example_ids found: {duplicates}")
        sys.exit(1)

    empty = [i for i in ids if not i]
    if empty:
        print(f"ERROR: {len(empty)} empty example_id values found.")
        sys.exit(1)

    # Build the freeze record.
    # This stores both:
    #   1. the stable `example_id` list. These are treated as data record IDs
    #      for experiment reproducibility, while preserving the canonical schema
    #      field name used across the dataset;
    #   2. a SHA-256 hash of core.jsonl, which makes later content drift visible
    #      even if the IDs themselves do not change.
    freeze_record = {
        "_comment": (
            "Frozen `example_id` values for the dissertation core dataset. "
            "Written before the first experiment run. "
            "Commit this file to git immediately after generation. "
            "The git commit timestamp establishes pre-experiment status. "
            "Do not regenerate after experiments have started."
        ),
        "frozen_at":      datetime.now(timezone.utc).isoformat(),
        "total":          len(ids),
        "expected_total": EXPECTED_TOTAL,
        "source_file":    str(CORE_JSONL),
        "source_sha256":  sha256_file(CORE_JSONL),
        "example_ids":    ids,
    }

    FROZEN_IDS.parent.mkdir(parents=True, exist_ok=True)
    with FROZEN_IDS.open("w", encoding="utf-8") as f:
        json.dump(freeze_record, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  Frozen {len(ids)} data record IDs from the `example_id` field.")
    print(f"  Source SHA-256: {freeze_record['source_sha256']}")
    print(f"  Written: {FROZEN_IDS}")
    print()
    print("Data record ID prefix distribution:")
    
    prefixes = Counter(eid.split("-")[0] for eid in ids)
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix:<10} {count}")
    print()
    print("Next steps:")
    print("  git add data/core/frozen_ids.json")
    print("  git commit -m \"freeze example IDs — 190 examples locked before first experiment run\"")
    print()
    print("IMPORTANT: Do not regenerate frozen_ids.json after any experiment has run.")


def verify(args: argparse.Namespace) -> None:
    """
    Compare current core.jsonl IDs against frozen_ids.json.
    Raises with a clear diff if any mismatch is found.
    """
    if not FROZEN_IDS.exists():
        print(
            f"ERROR: {FROZEN_IDS} not found.\n"
            "Run without --verify first to freeze the IDs."
        )
        sys.exit(1)

    print(f"Loading frozen IDs from {FROZEN_IDS}...")
    with FROZEN_IDS.open(encoding="utf-8") as f:
        freeze_record = json.load(f)

    frozen = sorted(freeze_record["example_ids"])
    frozen_at = freeze_record.get("frozen_at", "unknown")
    frozen_total = freeze_record.get("total", len(frozen))
    frozen_hash = freeze_record.get("source_sha256", "")

    print(f"  Frozen at:    {frozen_at}")
    print(f"  Frozen total: {frozen_total}")
    if frozen_hash:
        print(f"  Frozen hash:  {frozen_hash}")

    print(f"\nLoading current IDs from {CORE_JSONL}...")
    current = load_ids(CORE_JSONL)
    current_hash = sha256_file(CORE_JSONL)
    print(f"  Current total: {len(current)}")
    print(f"  Current hash:  {current_hash}")

    frozen_set  = set(frozen)
    current_set = set(current)

    added   = sorted(current_set - frozen_set)
    removed = sorted(frozen_set - current_set)
    count_mismatch = len(current) != len(frozen)
    hash_mismatch = bool(frozen_hash) and current_hash != frozen_hash

    errors = []

    # ID drift is the hard failure for the stable-ID requirement.
    if removed:
        errors.append(f"IDs removed since freeze ({len(removed)}): {removed[:10]}"
                      + (" ..." if len(removed) > 10 else ""))
    if added:
        errors.append(f"IDs added since freeze ({len(added)}): {added[:10]}"
                      + (" ..." if len(added) > 10 else ""))
    if count_mismatch and not added and not removed:
        errors.append(
            f"Count mismatch: frozen={len(frozen)}, current={len(current)} "
            "(possible duplicate IDs in current dataset)"
        )

    # Content drift is also a failure because it means core.jsonl changed after
    # the freeze point, even if the example IDs are still identical.
    if hash_mismatch:
        errors.append(
            "Dataset content hash changed since freeze: "
            f"frozen={frozen_hash}, current={current_hash}"
        )

    if errors:
        print()
        print(f"  ID DRIFT DETECTED — {len(errors)} issue(s):")
        for err in errors:
            print(f"    ERROR: {err}")
        print()
        print("Do not run experiments until ID drift is resolved.")
        print("If the dataset was legitimately rebuilt before any experiment ran,")
        print("re-freeze with: python scripts/freeze_example_ids.py --force")
        sys.exit(1)

    print()
    print(f"  All {len(current)} example IDs match the frozen set ✓")
    if frozen_hash:
        print("  Dataset SHA-256 also matches frozen record ✓")
    print("  Safe to run experiments.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or verify dissertation core dataset example IDs."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify current IDs match frozen_ids.json instead of freezing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing frozen_ids.json. Only use if dataset was "
            "rebuilt before any experiment ran. Document in methodology chapter."
        ),
    )
    args = parser.parse_args()

    if args.verify:
        verify(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
