# src/guardrail_eval/costs.py

def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Approximate GPT-4o-mini cost.

    Input:  $0.150 per 1M tokens
    Output: $0.600 per 1M tokens

    This preserves the original dissertation runner cost formula.
    """
    return round(
        prompt_tokens * 0.00000015
        + completion_tokens * 0.00000060,
        6,
    )