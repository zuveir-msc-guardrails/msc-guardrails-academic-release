# src/guardrail_eval/schemas.py

"""
Schema definitions for experiment CSV/JSONL outputs.

Each condition runner returns dictionaries whose keys are ordered by the
field-name lists in this module. Keeping schemas centralised has two purposes:

1. It makes the refactored runners comparable with the original experiment
   outputs because column order and field names are stable.
2. It separates *experiment behaviour* from *logging contract*. Condition files
   decide what happened; this file defines how that result is recorded.

Important scoring note:
    ``auto_*`` fields are deterministic review aids.
    ``llm_judge_*`` fields are reserved and currently blank.
    ``human_final_*`` fields are the authoritative final review/scoring fields
    used for dissertation metrics.

C5-specific note:
    For C5a/C5b/C5c, ``guardrail_decision`` records the document-level
    classifier decision. The final end-to-end action is recorded separately in
    ``pipeline_action`` and ``backend_context_type``.
"""

# ── C0/C1 base schema ────────────────────────────────────────────────────────
# C0 and C1 share the same output contract: one backend call result plus helper
# flags, human-review placeholders, timing/token/cost fields, and error status.

C0_FIELDNAMES = [
    "example_id",
    "stratum",
    "attack_type",
    "label",
    "condition",
    "prompt_version",
    "guardrail_decision",
    "agent_output",
    "full_output",
    "proposed_tool_call",
    "tool_call_allowed",
    "auto_canary_leaked",
    "auto_tool_call_proposed",
    "auto_tool_name",
    "auto_tool_arguments",
    "auto_attacker_destination_matched",
    "auto_attacker_signal_present",
    "auto_expected_answer_present",
    "auto_expected_safe_tool_matched",
    "auto_expected_safe_destination_matched",
    "auto_expected_safe_tool_arguments_matched",
    "auto_benign_tool_task_success_signal",
    "auto_markdown_marker_success",
    "auto_attacker_url_reproduced",
    "auto_attack_success_signal",
    "auto_task_success_signal",
    "llm_judge_attack_success",
    "llm_judge_task_success",
    "llm_judge_ua",
    "llm_judge_reason",
    "human_final_attack_success",
    "human_final_task_success",
    "human_final_ua",
    "human_final_reason",
    "needs_human_review",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "approx_cost_usd",
    "model_name",
    "timestamp",
    "error",
]

# C1 is prompt-only hardening, so it reuses the C0 row structure exactly.
C1_FIELDNAMES = C0_FIELDNAMES

# C2 adds deterministic heuristic detector metadata before the backend fields.
C2_FIELDNAMES = (
    C0_FIELDNAMES[:7]
    + [
        "guardrail_reason",
        "heuristic_rule_matches",
    ]
    + C0_FIELDNAMES[7:]
)

# C3 adds LLM document-classifier metadata: decision explanation, model usage,
# latency, and cost. These fields support detector-level analysis separately
# from backend task/attack outcomes.
C3_GUARDRAIL_FIELDNAMES = [
    "guardrail_reason",
    "guardrail_attack_type",
    "guardrail_confidence",
    "guardrail_raw_response",
    "guardrail_error",
    "guardrail_model_name",
    "guardrail_latency_seconds",
    "guardrail_prompt_tokens",
    "guardrail_completion_tokens",
    "guardrail_total_tokens",
    "guardrail_approx_cost_usd",
]

# Pipeline totals combine guardrail and backend model usage for conditions with
# more than one model-facing stage.
C3_PIPELINE_FIELDNAMES = [
    "total_pipeline_latency_seconds",
    "total_pipeline_tokens",
    "total_pipeline_approx_cost_usd",
]

# C3 schema = C0 prefix + classifier fields + agent/scoring fields + pipeline totals.
C3_FIELDNAMES = (
    C0_FIELDNAMES[:7]
    + C3_GUARDRAIL_FIELDNAMES
    + C0_FIELDNAMES[7:C0_FIELDNAMES.index("model_name")]
    + C3_PIPELINE_FIELDNAMES
    + C0_FIELDNAMES[C0_FIELDNAMES.index("model_name"):]
)

# ── C5 shared schema fragments ───────────────────────────────────────────────
# C5 conditions share a common prefix because they all have:
#   1. document-level classifier output;
#   2. final pipeline-action fields;
#   3. sanitisation outcome fields.
#
# ``pipeline_action`` is the key field for review:
#   - allow_full_context: backend saw the original context;
#   - sanitised_context: backend saw cleaned context;
#   - blocked_fallback: backend was not called.
C5_BASE_PREFIX_FIELDNAMES = [
    "example_id",
    "stratum",
    "attack_type",
    "label",
    "condition",
    "prompt_version",
    "guardrail_prompt_version",
    "guardrail_decision",
    "guardrail_reason",
    "guardrail_attack_type",
    "guardrail_confidence",
    "guardrail_raw_response",
    "guardrail_error",
    "guardrail_model_name",
    "guardrail_latency_seconds",
    "guardrail_prompt_tokens",
    "guardrail_completion_tokens",
    "guardrail_total_tokens",
    "guardrail_approx_cost_usd",
    "pipeline_action",
    "backend_called",
    "backend_context_type",
    "sanitisation_attempted",
    "sanitisation_succeeded",
    "sanitisation_fallback_to_block",
    "sanitisation_strategy",
    "sanitisation_segments_removed",
    "best_match_score",
    "sanitised_context",
    "sanitisation_removed_preview",
]

# Additional model-usage and audit fields for C5a/C5c sanitisation stages.
# C5b does not use these because its fuzzy removal is deterministic and does
# not make a second sanitiser LLM call.
C5_SENTENCE_GUARDRAIL_FIELDNAMES = [
    "sentence_units_total",
    "sentence_units_blocked",
    "sentence_guardrail_prompt_tokens",
    "sentence_guardrail_completion_tokens",
    "sentence_guardrail_total_tokens",
    "sentence_guardrail_latency_seconds",
    "sentence_guardrail_approx_cost_usd",
    "sentence_guardrail_errors",
    "sentence_guardrail_decisions_json",
]

# Backend output, helper flags, review placeholders, backend usage, pipeline
# totals, and run metadata shared by all C5 conditions.
C5_AGENT_AND_SCORING_FIELDNAMES = [
    "agent_output",
    "full_output",
    "proposed_tool_call",
    "tool_call_allowed",
    "auto_canary_leaked",
    "auto_tool_call_proposed",
    "auto_tool_name",
    "auto_tool_arguments",
    "auto_attacker_destination_matched",
    "auto_attacker_signal_present",
    "auto_expected_answer_present",
    "auto_expected_safe_tool_matched",
    "auto_expected_safe_destination_matched",
    "auto_expected_safe_tool_arguments_matched",
    "auto_benign_tool_task_success_signal",
    "auto_markdown_marker_success",
    "auto_attacker_url_reproduced",
    "auto_attack_success_signal",
    "auto_task_success_signal",
    "llm_judge_attack_success",
    "llm_judge_task_success",
    "llm_judge_ua",
    "llm_judge_reason",
    "human_final_attack_success",
    "human_final_task_success",
    "human_final_ua",
    "human_final_reason",
    "needs_human_review",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "approx_cost_usd",
    "total_pipeline_latency_seconds",
    "total_pipeline_tokens",
    "total_pipeline_approx_cost_usd",
    "model_name",
    "timestamp",
    "error",
]

# C5a = document classifier + sentence/unit LLM sanitiser + backend.
C5A_FIELDNAMES = (
    C5_BASE_PREFIX_FIELDNAMES
    + C5_SENTENCE_GUARDRAIL_FIELDNAMES
    + C5_AGENT_AND_SCORING_FIELDNAMES
)

# C5b = document classifier + deterministic fuzzy/payload sanitiser + backend.
# No sentence_guardrail fields are included because no second LLM sanitiser call
# is made in C5b.
C5B_FIELDNAMES = (
    C5_BASE_PREFIX_FIELDNAMES
    + C5_AGENT_AND_SCORING_FIELDNAMES
)

# C5c asks the document classifier for extra localisation metadata used by the
# context-aware unit-removal sanitiser.
C5C_EXTRA_GUARDRAIL_FIELDNAMES = [
    "guardrail_blocking_evidence",
    "guardrail_suspect_instruction_type",
]

# C5c audit fields make the unitisation and removal decision inspectable without
# rerunning the sanitiser.
C5C_AUDIT_FIELDNAMES = [
    "context_units_json",
    "context_units_text",
    "removed_unit_ids_json",
    "removed_units_json",
    "kept_units_json",
    "cleaning_trace_json",
]

# C5c = document classifier with localisation metadata + one context-aware
# unit-removal LLM call + backend.
C5C_FIELDNAMES = (
    [
        "example_id",
        "stratum",
        "attack_type",
        "label",
        "condition",
        "prompt_version",
        "guardrail_prompt_version",
        "guardrail_decision",
        "guardrail_reason",
        "guardrail_attack_type",
        "guardrail_confidence",
    ]
    + C5C_EXTRA_GUARDRAIL_FIELDNAMES
    + [
        "guardrail_raw_response",
        "guardrail_error",
        "guardrail_model_name",
        "guardrail_latency_seconds",
        "guardrail_prompt_tokens",
        "guardrail_completion_tokens",
        "guardrail_total_tokens",
        "guardrail_approx_cost_usd",
    ]
    + C5_BASE_PREFIX_FIELDNAMES[C5_BASE_PREFIX_FIELDNAMES.index("pipeline_action"):]
    + C5_SENTENCE_GUARDRAIL_FIELDNAMES
    + C5C_AUDIT_FIELDNAMES
    + C5_AGENT_AND_SCORING_FIELDNAMES
)