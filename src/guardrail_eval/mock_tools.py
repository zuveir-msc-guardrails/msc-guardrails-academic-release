# src/guardrail_eval/mock_tools.py

"""
Safe mock tool layer for the guardrail evaluation harness.

The experiment includes tool-use tasks because indirect prompt-injection attacks
often try to make an agent send emails, open URLs, or export data. For safety
and reproducibility, this module exposes those tools to the LLM as *mock* tools
only.

Important safety property:
    No function in this file performs a real external side effect.

That means:
    - send_email() does not send an email;
    - open_url() does not open or fetch a URL;
    - export_file() does not write a file;
    - create_report() does not create a real report.

Instead, each mock tool returns a structured dictionary recording what the model
tried to do. The experiment then uses this record to score whether the proposed
tool call was legitimate, attacker-controlled, or irrelevant.

This design lets the dissertation evaluate tool-misuse behaviour without
putting real systems, recipients, URLs, or files at risk.
"""

from __future__ import annotations

from typing import Any


# ── Mock tool implementations ────────────────────────────────────────────────
#
# Each function mirrors a tool the backend model can choose to call. The return
# values are intentionally simple dictionaries so they can be serialised directly
# into CSV/JSONL logs and inspected during human review.
#
# The common fields mean:
#   live_executed=False  -> no real-world action was taken;
#   mock_executed=True   -> the tool call was captured by the harness;
#   allowed=True         -> the harness allowed the mock execution to be logged.
#
# "allowed=True" does not mean the call was safe or correct. It only means the
# evaluation harness recorded the proposed call instead of suppressing it. Safety
# and utility are judged later from the logged arguments and human_final_* fields.


def send_email(
    to: str,
    subject: str,
    body: str,
    example_id: str,
    condition: str,
) -> dict[str, Any]:
    """
    Record a proposed email send without sending anything.

    This is used for both benign email tasks and tool-misuse attacks. The
    recipient, subject, and body preview are logged so review scripts can check
    whether the model targeted the legitimate destination or an attacker
    destination.
    """
    return {
        "tool": "send_email",
        "to": to,
        "subject": subject,
        "body_preview": body[:200],
        "example_id": example_id,
        "condition": condition,
        "live_executed": False,
        "mock_executed": True,
        "allowed": True,
    }


def open_url(
    url: str,
    example_id: str,
    condition: str,
) -> dict[str, Any]:
    """
    Record a proposed URL open without opening or fetching the URL.

    This prevents prompt-injection payloads from causing real network access
    while still preserving the URL the model attempted to use.
    """
    return {
        "tool": "open_url",
        "url": url,
        "example_id": example_id,
        "condition": condition,
        "live_executed": False,
        "mock_executed": True,
        "allowed": True,
    }


def export_file(
    filename: str,
    content: str,
    example_id: str,
    condition: str,
) -> dict[str, Any]:
    """
    Record a proposed file export without writing to disk.

    Only a content preview is logged to keep the output manageable and to avoid
    creating files as a side effect of model behaviour.
    """
    return {
        "tool": "export_file",
        "filename": filename,
        "content_preview": str(content)[:200],
        "example_id": example_id,
        "condition": condition,
        "live_executed": False,
        "mock_executed": True,
        "allowed": True,
    }


def create_report(
    title: str,
    content: str,
    example_id: str,
    condition: str,
) -> dict[str, Any]:
    """
    Record a proposed report creation without creating a real document.

    The title and content preview are retained for review of benign report tasks
    and possible injected report-generation requests.
    """
    return {
        "tool": "create_report",
        "title": title,
        "content_preview": str(content)[:200],
        "example_id": example_id,
        "condition": condition,
        "live_executed": False,
        "mock_executed": True,
        "allowed": True,
    }


MOCK_TOOLS = {
    "send_email": send_email,
    "open_url": open_url,
    "export_file": export_file,
    "create_report": create_report,
}


# ── OpenAI tool definitions ──────────────────────────────────────────────────
#
# TOOL_DEFINITIONS is the schema passed to the backend LLM. These definitions
# describe what tools the model *believes* are available. The actual execution
# remains local and safe because call_agent() routes proposed tool calls to
# execute_mock_tool(), not to real email, browser, or filesystem APIs.
#
# Keeping these schemas in one file makes it clear which actions were available
# consistently across C0-C5 conditions.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in a browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to open"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_file",
            "description": "Export content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_report",
            "description": "Create a report with a title and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        },
    },
]


# ── Mock tool dispatcher ─────────────────────────────────────────────────────

def execute_mock_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    example_id: str,
    condition: str,
) -> dict[str, Any] | None:
    """
    Execute the local mock version of a proposed tool call.

    Args:
        tool_name:
            Name proposed by the LLM, such as "send_email" or "open_url".
        tool_args:
            Parsed JSON arguments proposed by the LLM.
        example_id:
            Dataset row identifier, included in the returned audit record.
        condition:
            Experiment condition name, such as C0, C3, C5b, or C5c.

    Returns:
        A structured mock execution record, or None if the model proposed an
        unknown tool name.

    Safety note:
        This dispatcher never calls external services. It only records the
        model's proposed action in a normalised dictionary for later scoring.
    """
    tool_call_result = None

    if tool_name in MOCK_TOOLS:
        fn = MOCK_TOOLS[tool_name]

        # Explicit branches keep the argument mapping readable for examiners.
        # They also avoid accidentally passing unrecognised model-supplied
        # fields into a mock function.
        if tool_name == "send_email":
            tool_call_result = fn(
                tool_args.get("to", ""),
                tool_args.get("subject", ""),
                tool_args.get("body", ""),
                example_id,
                condition,
            )

        elif tool_name == "open_url":
            tool_call_result = fn(
                tool_args.get("url", ""),
                example_id,
                condition,
            )

        elif tool_name == "export_file":
            tool_call_result = fn(
                tool_args.get("filename", ""),
                tool_args.get("content", ""),
                example_id,
                condition,
            )

        elif tool_name == "create_report":
            tool_call_result = fn(
                tool_args.get("title", ""),
                tool_args.get("content", ""),
                example_id,
                condition,
            )

    return tool_call_result
