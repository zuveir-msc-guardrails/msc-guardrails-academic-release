"""
inspect_agentdojo.py
--------------------
Extracts workspace injection task goals and user task prompts
from the installed agentdojo package and writes curated CSVs.

Run from project root:
    python experiments/inspect_agentdojo.py
"""

import csv
import re
import sys
from pathlib import Path

# Find the installed package
import agentdojo
PACKAGE_ROOT = Path(agentdojo.__file__).parent
print(f"AgentDojo package found at: {PACKAGE_ROOT}")

# ── Injection tasks ───────────────────────────────────────────────────────────

V1_INJECTION   = PACKAGE_ROOT / "default_suites/v1/workspace/injection_tasks.py"
V1_2_INJECTION = PACKAGE_ROOT / "default_suites/v1_2/workspace/injection_tasks.py"

def extract_injection_tasks(filepath, suite_version):
    """Extract class names, GOAL strings, and attacker tool from a task file."""
    text = filepath.read_text(encoding="utf-8")
    tasks = []

    # Split into class blocks
    blocks = re.split(r"\n(?=@task_suite\.register_injection_task\nclass )", text)
    for block in blocks:
        class_match = re.search(r"class (InjectionTask\d+)", block)
        if not class_match:
            continue
        class_name = class_match.group(1)
        task_num   = re.search(r"\d+", class_name).group()
        task_id    = f"injection_task_{task_num}"

        # Extract constants
        constants = {}
        for m in re.finditer(r'_(\w+)\s*=\s*["\']([^"\']+)["\']', block):
            constants[m.group(1)] = m.group(2)

        # Extract GOAL
        goal_match = re.search(
            r'GOAL\s*=\s*(?:f?"([^"]+)"|f?\'([^\']+)\'|\(([^)]+)\))',
            block, re.DOTALL
        )
        if not goal_match:
            continue
        raw_goal = (goal_match.group(1) or goal_match.group(2) or goal_match.group(3) or "").strip()

        # Resolve f-string variables
        goal = raw_goal
        for k, v in constants.items():
            goal = goal.replace(f"{{{k}}}", v)
            goal = goal.replace(f"{{self._{k}}}", v)
            goal = goal.replace(f"{{_{k}}}", v)
        goal = goal.replace("\n", " ").strip()

        # Detect attacker tool
        tool = ""
        dest = ""
        if "send_email" in block:
            tool = "send_email"
            dest = constants.get("RECIPIENT", "mark.black-2134@gmail.com")
        elif "delete_file" in block:
            tool = "delete_file"
        elif "create_calendar_event" in block:
            tool = "create_calendar_event"
            dest = constants.get("PARTICIPANT", "")

        difficulty_match = re.search(r"TaskDifficulty\.(\w+)", block)
        difficulty = difficulty_match.group(1).lower() if difficulty_match else ""

        tasks.append({
            "task_id":               task_id,
            "class_name":            class_name,
            "suite_version":         suite_version,
            "difficulty":            difficulty,
            "goal":                  goal,
            "attacker_tool":         tool,
            "attacker_destination":  dest,
        })

    return tasks

injection_tasks = []
for path, version in [(V1_INJECTION, "v1"), (V1_2_INJECTION, "v1_2")]:
    if path.exists():
        extracted = extract_injection_tasks(path, version)
        injection_tasks.extend(extracted)
        print(f"Extracted {len(extracted)} injection tasks from {version}")
    else:
        print(f"WARNING: {path} not found")

# ── User tasks ────────────────────────────────────────────────────────────────

V1_USER = PACKAGE_ROOT / "default_suites/v1/workspace/user_tasks.py"

def extract_user_tasks(filepath):
    """Extract class names and PROMPT strings from user tasks file."""
    text = filepath.read_text(encoding="utf-8")
    tasks = []

    blocks = re.split(r"\n(?=@task_suite\.register_user_task\nclass )", text)
    for block in blocks:
        class_match = re.search(r"class (UserTask\d+)", block)
        if not class_match:
            continue
        class_name = class_match.group(1)
        task_num   = re.search(r"\d+", class_name).group()
        task_id    = f"user_task_{task_num}"

        # Extract constants
        constants = {}
        for m in re.finditer(r'_(\w+)\s*=\s*["\']([^"\']+)["\']', block):
            constants[m.group(1)] = m.group(2)

        # Extract PROMPT
        prompt_match = re.search(
            r'PROMPT\s*=\s*(?:f?"([^"]+)"|f?\'([^\']+)\'|\(([^)]+)\))',
            block, re.DOTALL
        )
        if not prompt_match:
            continue
        raw_prompt = (
            prompt_match.group(1) or
            prompt_match.group(2) or
            prompt_match.group(3) or ""
        ).strip()

        prompt = raw_prompt
        for k, v in constants.items():
            prompt = prompt.replace(f"{{{k}}}", v)
            prompt = prompt.replace(f"{{self._{k}}}", v)
            prompt = prompt.replace(f"{{_{k}}}", v)
        prompt = prompt.replace("\n", " ").strip()

        # Detect task type
        if "send_email" in block:
            task_type = "email_write"
            safe_tool = "send_email"
        elif "create_calendar_event" in block or "reschedule" in block:
            task_type = "calendar_write"
            safe_tool = "create_calendar_event"
        elif "delete_file" in block:
            task_type = "drive_write"
            safe_tool = "delete_file"
        elif "append_to_file" in block or "create_file" in block:
            task_type = "drive_write"
            safe_tool = "append_to_file"
        elif "calendar" in block.lower():
            task_type = "calendar_read"
            safe_tool = ""
        elif "email" in block.lower() or "inbox" in block.lower():
            task_type = "email_read"
            safe_tool = ""
        else:
            task_type = "drive_read"
            safe_tool = ""

        tasks.append({
            "task_id":    task_id,
            "class_name": class_name,
            "suite_version": "v1",
            "prompt":     prompt,
            "task_type":  task_type,
            "safe_tool":  safe_tool,
        })

    return tasks

user_tasks = []
if V1_USER.exists():
    user_tasks = extract_user_tasks(V1_USER)
    print(f"Extracted {len(user_tasks)} user tasks from v1")
else:
    print(f"WARNING: {V1_USER} not found")

# ── Write CSVs ────────────────────────────────────────────────────────────────

OUT_DIR = Path("data/agentdojo_curated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

injection_path = OUT_DIR / "workspace_injection_tasks.csv"
with open(injection_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=injection_tasks[0].keys())
    writer.writeheader()
    writer.writerows(injection_tasks)
print(f"\nInjection tasks written to {injection_path}")

user_path = OUT_DIR / "workspace_user_tasks.csv"
with open(user_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=user_tasks[0].keys())
    writer.writeheader()
    writer.writerows(user_tasks)
print(f"User tasks written to {user_path}")

# ── Print summary ─────────────────────────────────────────────────────────────

print(f"\nSummary:")
print(f"  Injection tasks: {len(injection_tasks)}")
print(f"  User tasks:      {len(user_tasks)}")
print(f"\nInjection task goals:")
for t in injection_tasks:
    print(f"  {t['task_id']} ({t['suite_version']}): {t['goal'][:80]}")
print(f"\nUser task prompts (first 10):")
for t in user_tasks[:10]:
    print(f"  {t['task_id']}: {t['prompt'][:80]}")