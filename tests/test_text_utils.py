from guardrail_eval.text_utils import normalise_text
from guardrail_eval.text_utils import normalise_text, output_channels_text

def test_normalise_text_none():
    assert normalise_text(None) == ""


def test_normalise_text_string():
    assert normalise_text(" Hello ") == "hello"


def test_normalise_text_dict():
    assert normalise_text({"To": "Alice"}) == '{"to": "alice"}'


def test_output_channels_text_combines_visible_text_tool_call_and_tool_result():
    combined = output_channels_text(
        response_text="Visible answer",
        proposed_tool_call={
            "tool": "send_email",
            "arguments": {"to": "attacker@example.com"},
        },
        tool_call_result={
            "mock_executed": True,
            "to": "attacker@example.com",
        },
    )

    assert "Visible answer" in combined
    assert "send_email" in combined
    assert "attacker@example.com" in combined
    assert "mock_executed" in combined


def test_output_channels_text_handles_blank_values():
    combined = output_channels_text(
        response_text=None,
        proposed_tool_call=None,
        tool_call_result=None,
    )

    assert combined == "\n{}\n{}"