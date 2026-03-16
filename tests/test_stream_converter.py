"""
Tests for Gemini NDJSON stream → Claude SSE conversion.

Covers:
- jsonPath parsing
- Nested value set/append
- partialArgs accumulation (AskUserQuestion, TodoWrite, MCP tools)
- Empty functionCall with willContinue handling
- No-args tool calls
- Multi-question streaming
"""

import json
import pytest
import asyncio
from typing import List, AsyncIterator

from gemini_claude_bridge.stream_converter import (
    GeminiStreamConverter,
    parse_json_path,
    set_nested_value,
    append_nested_value,
)


class TestParseJsonPath:

    def test_simple_field(self):
        assert parse_json_path("$.field") == ["field"]

    def test_nested_field(self):
        assert parse_json_path("$.field.subfield") == ["field", "subfield"]

    def test_array_index(self):
        assert parse_json_path("$.array[0]") == ["array", 0]

    def test_array_with_field(self):
        assert parse_json_path("$.array[0].field") == ["array", 0, "field"]

    def test_nested_array(self):
        assert parse_json_path("$.questions[0].question") == ["questions", 0, "question"]
        assert parse_json_path("$.questions[0].options[0].description") == ["questions", 0, "options", 0, "description"]

    def test_todos_path(self):
        assert parse_json_path("$.todos[0].content") == ["todos", 0, "content"]
        assert parse_json_path("$.todos[0].status") == ["todos", 0, "status"]

    def test_empty(self):
        assert parse_json_path("") == []
        assert parse_json_path("$") == []


class TestSetNestedValue:

    def test_simple_field(self):
        obj = {}
        set_nested_value(obj, ["field"], "value")
        assert obj == {"field": "value"}

    def test_nested_field(self):
        obj = {}
        set_nested_value(obj, ["a", "b"], "value")
        assert obj == {"a": {"b": "value"}}

    def test_array_index(self):
        obj = {}
        set_nested_value(obj, ["array", 0], "value")
        assert obj == {"array": ["value"]}

    def test_array_with_field(self):
        obj = {}
        set_nested_value(obj, ["questions", 0, "question"], "What?")
        assert obj == {"questions": [{"question": "What?"}]}

    def test_complex_nested(self):
        obj = {}
        set_nested_value(obj, ["questions", 0, "options", 0, "description"], "desc")
        assert obj == {"questions": [{"options": [{"description": "desc"}]}]}


class TestAppendNestedValue:

    def test_append_new(self):
        obj = {}
        append_nested_value(obj, ["field"], "hello")
        assert obj == {"field": "hello"}

    def test_append_existing(self):
        obj = {"field": "hello"}
        append_nested_value(obj, ["field"], " world")
        assert obj == {"field": "hello world"}

    def test_append_nested_array(self):
        obj = {}
        append_nested_value(obj, ["questions", 0, "question"], "What is ")
        append_nested_value(obj, ["questions", 0, "question"], "your name?")
        assert obj == {"questions": [{"question": "What is your name?"}]}


# --- Stream converter integration tests ---

async def collect_events(converter: GeminiStreamConverter, ndjson_lines: List[str]) -> List[str]:
    events = []

    async def mock_stream() -> AsyncIterator[str]:
        for line in ndjson_lines:
            yield line

    async for event in converter.convert_stream(mock_stream()):
        events.append(event)
    return events


def extract_tool_input(events: List[str]) -> dict:
    accumulated = ""
    for event in events:
        if "input_json_delta" in event:
            for line in event.split('\n'):
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if data.get("delta", {}).get("type") == "input_json_delta":
                        accumulated += data["delta"]["partial_json"]
    if accumulated:
        return json.loads(accumulated)
    return {}


class TestStreamConverter:

    def test_ask_user_question_all_fields(self):
        """Verify question field is not lost across empty functionCall boundaries."""
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"AskUserQuestion","willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].header","stringValue":"Info","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].header","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"boolValue":false,"jsonPath":"$.questions[0].multiSelect"}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].options[0].description","stringValue":"Provide ERP","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].options[0].description","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].options[0].label","stringValue":"Enter ERP","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].options[0].label","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            # Empty functionCall with willContinue:true — must NOT trigger early close
            '{"candidates":[{"content":{"parts":[{"functionCall":{"willContinue":true}}],"role":"model"}}]}',
            # question field comes AFTER the empty functionCall
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":"Please provide your ERP account","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":50}}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-1", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert "questions" in tool_input
        q = tool_input["questions"][0]
        assert q["question"] == "Please provide your ERP account"
        assert q["header"] == "Info"
        assert q["multiSelect"] is False
        assert q["options"][0]["description"] == "Provide ERP"
        assert q["options"][0]["label"] == "Enter ERP"

    def test_empty_fc_will_continue_true_does_not_close(self):
        """Empty functionCall with willContinue:true must not end the stream."""
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"AskUserQuestion","willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].header","stringValue":"Test","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":"Test question","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP"}]}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-wc", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert tool_input["questions"][0]["header"] == "Test"
        assert tool_input["questions"][0]["question"] == "Test question"

    def test_todo_write_nested(self):
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"TodoWrite","willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.todos[0].content","stringValue":"Task content","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.todos[0].status","stringValue":"in_progress","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.todos[0].activeForm","stringValue":"Running","willContinue":false}],"willContinue":false}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP"}]}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-todo", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert tool_input["todos"][0]["content"] == "Task content"
        assert tool_input["todos"][0]["status"] == "in_progress"
        assert tool_input["todos"][0]["activeForm"] == "Running"

    def test_string_accumulation(self):
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"AskUserQuestion","willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":"Please ","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":"provide ","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":"info","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":""}],"willContinue":false}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP"}]}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-accum", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert tool_input["questions"][0]["question"] == "Please provide info"

    def test_multiple_questions(self):
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"AskUserQuestion","willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].header","stringValue":"Q1","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"boolValue":false,"jsonPath":"$.questions[0].multiSelect"}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].question","stringValue":"First?","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].options[0].label","stringValue":"A","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[0].options[0].description","stringValue":"D1","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[1].header","stringValue":"Q2","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"boolValue":false,"jsonPath":"$.questions[1].multiSelect"}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[1].question","stringValue":"Second?","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[1].options[0].label","stringValue":"B","willContinue":false}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.questions[1].options[0].description","stringValue":"D2","willContinue":false}],"willContinue":false}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP"}]}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-multi", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert len(tool_input["questions"]) == 2
        assert tool_input["questions"][0]["header"] == "Q1"
        assert tool_input["questions"][0]["question"] == "First?"
        assert tool_input["questions"][1]["header"] == "Q2"
        assert tool_input["questions"][1]["question"] == "Second?"

    def test_no_args_tool_call(self):
        """No-args tool call (e.g. list_tables) must not be silently dropped."""
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"mcp__joytable__list_tables"},"thoughtSignature":"ClABjz1rX7Uhvxj6"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":16}}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-noargs", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))

        block_starts = [e for e in events if "content_block_start" in e]
        assert len(block_starts) == 1
        assert "mcp__joytable__list_tables" in block_starts[0]

        tool_input = extract_tool_input(events)
        assert tool_input == {}

        block_stops = [e for e in events if "content_block_stop" in e]
        assert len(block_stops) == 1

    def test_mcp_tool_mixed_simple_and_nested(self):
        """MCP tool with mixed simple ($.agentId) and nested ($.formFields[0].label) paths."""
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"mcp__save_form","willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.agentId","stringValue":"12345","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.agentId","stringValue":"67890","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.agentId","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.formFields[0].label","stringValue":"Name","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.formFields[0].label","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"boolValue":true,"jsonPath":"$.formFields[0].required"}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.formFields[0].type","stringValue":"TextField","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.formFields[0].type","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.pageCode","stringValue":"pg001","willContinue":true}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.pageCode","stringValue":""}],"willContinue":true}}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"functionCall":{}}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":100}}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-mcp", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert tool_input["agentId"] == "1234567890"
        assert tool_input["formFields"][0]["label"] == "Name"
        assert tool_input["formFields"][0]["required"] is True
        assert tool_input["formFields"][0]["type"] == "TextField"
        assert tool_input["pageCode"] == "pg001"

    def test_text_streaming(self):
        """Plain text streaming with cumulative delta."""
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"text":"Hello world"}],"role":"model"}}]}',
            '{"candidates":[{"content":{"parts":[{"text":"Hello world!"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3}}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-text", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))

        # Collect text deltas
        text = ""
        for event in events:
            if "text_delta" in event:
                for line in event.split('\n'):
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data.get("delta", {}).get("type") == "text_delta":
                            text += data["delta"]["text"]

        assert text == "Hello world!"

    def test_non_streaming_function_call(self):
        """Non-streaming function call (complete args at once)."""
        ndjson_lines = [
            '{"candidates":[{"content":{"parts":[{"functionCall":{"name":"Read","args":{"file_path":"/tmp/test.py"}},"thoughtSignature":"sig123"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":10}}',
        ]

        converter = GeminiStreamConverter(
            request_id="test-nonstream-fc", model="gemini-2.5-flash", original_request_model="claude-sonnet-4-20250514",
        )
        events = asyncio.run(collect_events(converter, ndjson_lines))
        tool_input = extract_tool_input(events)

        assert tool_input == {"file_path": "/tmp/test.py"}

        # Verify stop_reason is tool_use
        for event in events:
            if "message_delta" in event:
                for line in event.split('\n'):
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data.get("type") == "message_delta":
                            assert data["delta"]["stop_reason"] == "tool_use"
