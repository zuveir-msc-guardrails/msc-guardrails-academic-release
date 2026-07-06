from guardrail_eval.mock_tools import execute_mock_tool, TOOL_DEFINITIONS


def test_send_email_mock_does_not_execute_live():
    result = execute_mock_tool(
        tool_name="send_email",
        tool_args={
            "to": "alice@example.com",
            "subject": "Hello",
            "body": "Test",
        },
        example_id="T-001",
        condition="C0",
    )

    assert result["tool"] == "send_email"
    assert result["to"] == "alice@example.com"
    assert result["live_executed"] is False
    assert result["mock_executed"] is True


def test_unknown_tool_returns_none():
    result = execute_mock_tool(
        tool_name="unknown_tool",
        tool_args={},
        example_id="T-001",
        condition="C0",
    )

    assert result is None


def test_tool_definitions_include_expected_tools():
    names = {
        item["function"]["name"]
        for item in TOOL_DEFINITIONS
    }

    assert names == {
        "send_email",
        "open_url",
        "export_file",
        "create_report",
    }