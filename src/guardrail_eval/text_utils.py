# src/guardrail_eval/text_utils.py

"""
Small text-normalisation helpers shared by deterministic review flags.

These functions are intentionally simple and deterministic. They are not used
to decide final dissertation labels; they support the ``auto_*`` helper fields
that make manual review easier.
"""

import json


def normalise_text(value):
    """
    Convert a value into a lowercase string for substring-style helper checks.

    Dictionaries and lists are serialised with ``json.dumps`` so tool-call
    argument structures can be searched in the same way as plain text outputs.

    This function deliberately does not do stemming, fuzzy matching, or semantic
    comparison. The helper flags should remain transparent and easy to audit.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False).lower().strip()
    return str(value).lower().strip()


def output_channels_text(response_text, proposed_tool_call=None, tool_call_result=None):
    """
    Combine all output channels that could contain a leaked string or destination.

    Some attacks succeed through normal text, while tool-misuse attacks may show
    up only in proposed tool arguments or mock tool results. Helper checks such
    as canary leakage and attacker URL reproduction therefore inspect a combined
    text view of all channels.

    Args:
        response_text:
            Assistant-visible output text or tool-call placeholder.
        proposed_tool_call:
            Parsed tool call proposed by the model, if any.
        tool_call_result:
            Local mock tool execution result, if any.

    Returns:
        A newline-separated string containing the response, proposed tool call,
        and mock tool result.
    """
    return "\n".join(
        [
            str(response_text or ""),
            json.dumps(proposed_tool_call or {}, ensure_ascii=False),
            json.dumps(tool_call_result or {}, ensure_ascii=False),
        ]
    )
