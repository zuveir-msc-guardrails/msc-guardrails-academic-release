from pathlib import Path
import json
import csv

BIPIA_ROOT = Path("external/BIPIA")

def preview_json(path: Path, max_items=3):
    print(f"\nJSON: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            for item in data[:max_items]:
                print(json.dumps(item, indent=2)[:1500])
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())[:30]}")
            print(json.dumps(data, indent=2)[:1500])
    except Exception as e:
        print(f"Could not read JSON: {e}")

def preview_jsonl(path: Path, max_items=3):
    print(f"\nJSONL: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_items:
                    break
                print(json.dumps(json.loads(line), indent=2)[:1500])
    except Exception as e:
        print(f"Could not read JSONL: {e}")

def preview_csv(path: Path, max_items=3):
    print(f"\nCSV: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print(f"Columns: {reader.fieldnames}")
            for i, row in enumerate(reader):
                if i >= max_items:
                    break
                print(row)
    except Exception as e:
        print(f"Could not read CSV: {e}")

def main():
    if not BIPIA_ROOT.exists():
        raise SystemExit("external/BIPIA not found. Clone BIPIA first.")

    files = list(BIPIA_ROOT.rglob("*"))
    data_files = [
        p for p in files
        if p.is_file() and p.suffix.lower() in [".json", ".jsonl", ".csv", ".tsv"]
    ]

    print(f"Found {len(data_files)} possible data files.\n")

    for path in data_files:
        print(path)

    print("\n" + "=" * 80)
    print("Previewing candidate files")
    print("=" * 80)

    for path in data_files[:30]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            preview_json(path)
        elif suffix == ".jsonl":
            preview_jsonl(path)
        elif suffix in [".csv", ".tsv"]:
            preview_csv(path)

if __name__ == "__main__":
    main()