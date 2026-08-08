"""
extract_agentdojo_contexts.py
-----------------------------
Extracts workspace contexts from the installed agentdojo package.
Writes curated JSONL artefacts to data/agentdojo_curated/.

Run from project root:
    python scripts/extract_agentdojo_contexts.py

Outputs:
    data/agentdojo_curated/workspace_clean_emails.jsonl
    data/agentdojo_curated/workspace_placeholder_emails.jsonl
    data/agentdojo_curated/workspace_calendar_events.jsonl
    data/agentdojo_curated/workspace_cloud_drive_files.jsonl
    data/agentdojo_curated/workspace_injection_vectors.jsonl
    
Inspect outputs:
in bash:    
    head -n 2 data/agentdojo_curated/workspace_clean_emails.jsonl
    head -n 2 data/agentdojo_curated/workspace_calendar_events.jsonl
    head -n 2 data/agentdojo_curated/workspace_cloud_drive_files.jsonl
"""

import csv
import json
import re
from pathlib import Path

import yaml
import agentdojo

# ── Locate package ────────────────────────────────────────────────────────────

AGENTDOJO_ROOT  = Path(agentdojo.__file__).resolve().parent
WORKSPACE_ROOT  = AGENTDOJO_ROOT / "data" / "suites" / "workspace"
INCLUDE_ROOT    = WORKSPACE_ROOT / "include"

print(f"AgentDojo root: {AGENTDOJO_ROOT}")
print(f"Workspace root: {WORKSPACE_ROOT}")

for path in [WORKSPACE_ROOT, INCLUDE_ROOT]:
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found. Check agentdojo installation.")

# ── Output directory ──────────────────────────────────────────────────────────

OUT_DIR = Path("data/agentdojo_curated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"  Written: {path} ({len(records)} records)")


def write_csv(path, records):
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        for r in records:
            # Flatten any list fields to semicolon-separated strings for CSV
            flat = {}
            for k, v in r.items():
                flat[k] = "; ".join(v) if isinstance(v, list) else v
            writer.writerow(flat)
    print(f"  Written: {path} ({len(records)} records)")

# ── 1. Inbox ──────────────────────────────────────────────────────────────────

inbox_path = INCLUDE_ROOT / "inbox.yaml"
with open(inbox_path, encoding="utf-8") as f:
    inbox_data = yaml.safe_load(f)

print(f"\nInbox YAML top-level keys: {list(inbox_data.keys()) if isinstance(inbox_data, dict) else type(inbox_data)}")
if isinstance(inbox_data, dict) and not inbox_data.get("initial_emails"):
    print(f"  WARNING: 'initial_emails' key missing or empty. Available keys: {list(inbox_data.keys())}")
emails = inbox_data.get("initial_emails", []) if isinstance(inbox_data, dict) else []
print(f"Total emails in inbox: {len(emails)}")

clean_emails = []
placeholder_emails = []

for e in emails:
    body = e.get("body", "")
    record = {
        "id":        e.get("id_", ""),
        "sender":    e.get("sender", ""),
        "recipients":e.get("recipients", []),
        "subject":   e.get("subject", ""),
        "body":      body,
        "status":    e.get("status", ""),
        "read":      e.get("read", True),
        "timestamp": str(e.get("timestamp", "")),
        "has_placeholder": "{" in body,
        "placeholder_key": "",
        "source":        "AgentDojo",
        "source_suite":  "workspace",
        "source_type":   "email",
    }

    if "{" in body:
        match = re.search(r"\{(\w+)\}", body)
        if match:
            record["placeholder_key"] = match.group(1)
        placeholder_emails.append(record)
    else:
        clean_emails.append(record)

print(f"  Clean emails (no injection placeholder): {len(clean_emails)}")
print(f"  Placeholder emails (injection sites):    {len(placeholder_emails)}")

write_jsonl(OUT_DIR / "workspace_clean_emails.jsonl", clean_emails)
write_csv(OUT_DIR / "workspace_clean_emails.csv", clean_emails)
write_jsonl(OUT_DIR / "workspace_placeholder_emails.jsonl", placeholder_emails)
write_csv(OUT_DIR / "workspace_placeholder_emails.csv", placeholder_emails)

# Print clean email summary
print("\nClean emails available for contexts:")
for e in clean_emails:
    print(f"  id={e['id']} | {e['sender'][:35]:<35} | {e['subject'][:50]}")

print("\nPlaceholder emails (injection sites):")
for e in placeholder_emails:
    print(f"  id={e['id']} | key={e['placeholder_key']:<35} | {e['subject'][:50]}")

# ── 2. Calendar ───────────────────────────────────────────────────────────────

calendar_path = INCLUDE_ROOT / "calendar.yaml"
with open(calendar_path, encoding="utf-8") as f:
    calendar_data = yaml.safe_load(f)

print(f"\nCalendar YAML top-level keys: {list(calendar_data.keys()) if isinstance(calendar_data, dict) else type(calendar_data)}")
if isinstance(calendar_data, dict) and not calendar_data.get("initial_events"):
    print(f"  WARNING: 'initial_events' key missing or empty. Available keys: {list(calendar_data.keys())}")
events = calendar_data.get("initial_events", []) if isinstance(calendar_data, dict) else []
print(f"Total calendar events: {len(events)}")

calendar_records = []
for ev in events:
    calendar_records.append({
        "title":        ev.get("title", ""),
        "description":  ev.get("description", ""),
        "start_time":   str(ev.get("start_time", "")),
        "end_time":     str(ev.get("end_time", "")),
        "location":     ev.get("location", ""),
        "participants": ev.get("participants", []),
        "all_day":      ev.get("all_day", False),
        "source":       "AgentDojo",
        "source_suite": "workspace",
        "source_type":  "calendar_event",
    })
    print(f"  {ev.get('title','')[:50]} | {str(ev.get('start_time',''))[:16]}")

write_jsonl(OUT_DIR / "workspace_calendar_events.jsonl", calendar_records)
write_csv(OUT_DIR / "workspace_calendar_events.csv", calendar_records)

# ── 3. Cloud drive ────────────────────────────────────────────────────────────

drive_path = INCLUDE_ROOT / "cloud_drive.yaml"
with open(drive_path, encoding="utf-8") as f:
    drive_data = yaml.safe_load(f)

print(f"\nCloud drive YAML top-level keys: {list(drive_data.keys()) if isinstance(drive_data, dict) else type(drive_data)}")
if isinstance(drive_data, dict) and not drive_data.get("initial_files"):
    print(f"  WARNING: 'initial_files' key missing or empty. Available keys: {list(drive_data.keys())}")
files = drive_data.get("initial_files", {}) if isinstance(drive_data, dict) else {}
print(f"Total cloud drive files: {len(files)}")

drive_records = []
if isinstance(files, dict):
    file_items = list(files.items())
elif isinstance(files, list):
    file_items = [
        (f.get("id_", f.get("id", f"file_{i+1:03d}")), f)
        for i, f in enumerate(files)
    ]
else:
    file_items = []

for file_id, file_data in file_items:
    record = {
        "file_id":        file_id,
        "filename":       file_data.get("filename", ""),
        "content":        file_data.get("content", ""),
        "content_preview":file_data.get("content", "")[:500],
        "owner":          file_data.get("owner", ""),
        "shared_with":    file_data.get("shared_with", []),
        "size":           file_data.get("size", 0),
        "source":         "AgentDojo",
        "source_suite":   "workspace",
        "source_type":    "cloud_drive_file",
    }
    drive_records.append(record)
    print(f"  id={file_id} | {file_data.get('filename','')[:50]}")

write_jsonl(OUT_DIR / "workspace_cloud_drive_files.jsonl", drive_records)
write_csv(OUT_DIR / "workspace_cloud_drive_files.csv", drive_records)

# ── 4. Injection vectors ──────────────────────────────────────────────────────

vectors_path = WORKSPACE_ROOT / "injection_vectors.yaml"
with open(vectors_path, encoding="utf-8") as f:
    vectors_data = yaml.safe_load(f)

print(f"\nInjection vectors:")
vector_records = []
if not isinstance(vectors_data, dict):
    raise SystemExit(f"ERROR: Unexpected injection_vectors.yaml structure: {type(vectors_data)}")

for key, value in vectors_data.items():
    description = value.get("description", "") if isinstance(value, dict) else ""
    default     = value.get("default", "")     if isinstance(value, dict) else str(value)
    record = {
        "vector_key":   key,
        "description":  description,
        "default":      default,
        "source":       "AgentDojo",
        "source_suite": "workspace",
        "source_type":  "injection_vector",
    }
    vector_records.append(record)
    print(f"  {key}")
    print(f"    default: {str(default)[:100]}")

write_jsonl(OUT_DIR / "workspace_injection_vectors.jsonl", vector_records)
write_csv(OUT_DIR / "workspace_injection_vectors.csv", vector_records)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Extraction complete.")
print(f"  Clean emails:          {len(clean_emails)}")
print(f"  Placeholder emails:    {len(placeholder_emails)}")
print(f"  Calendar events:       {len(calendar_records)}")
print(f"  Cloud drive files:     {len(drive_records)}")
print(f"  Injection vectors:     {len(vector_records)}")
print(f"\nAll outputs written to: {OUT_DIR}")
print("\nNext step: review outputs and select contexts for")
print("20 tool misuse + 25 benign tool-use examples.")
