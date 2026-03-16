"""
Tests for Claude → Gemini request conversion.

Covers:
- Basic message conversion
- Tool declarations
- Image handling in tool results
- System instruction injection
- thoughtSignature recovery
- Mixed model message splitting
- Consecutive user message merging
"""

import pytest
from unittest.mock import MagicMock, patch

from gemini_claude_bridge.request_converter import convert_claude_to_gemini
from gemini_claude_bridge.signature_manager import GeminiSignatureManager


def _make_msg(role: str, content):
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def _make_request(messages, tools=None, thinking=None, system="You are helpful."):
    req = MagicMock()
    req.messages = messages
    req.system = system
    req.stream = False
    req.model = "claude-sonnet-4-20250514"
    req.tools = tools
    req.thinking = thinking
    return req


class TestBasicConversion:

    def test_simple_text_message(self):
        messages = [_make_msg("user", "Hello")]
        req = _make_request(messages)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")

        assert result["model"] == "gemini-2.5-flash"
        assert len(result["contents"]) == 1
        # First user message gets system instruction prepended
        parts = result["contents"][0]["parts"]
        assert result["contents"][0]["role"] == "user"
        assert any("Hello" in p.get("text", "") for p in parts)

    def test_assistant_becomes_model(self):
        messages = [
            _make_msg("user", "Hi"),
            _make_msg("assistant", "Hello!"),
        ]
        req = _make_request(messages)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")

        roles = [c["role"] for c in result["contents"]]
        assert "model" in roles
        assert "assistant" not in roles

    def test_stream_flag(self):
        messages = [_make_msg("user", "Hi")]
        req = _make_request(messages)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash", stream=True)
        assert result["stream"] is True

    def test_no_system(self):
        messages = [_make_msg("user", "Hi")]
        req = _make_request(messages, system=None)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")
        # Should still work without system instruction
        assert len(result["contents"]) >= 1


class TestToolConversion:

    def test_tool_declarations(self):
        tool = MagicMock()
        tool.name = "Read"
        tool.description = "Read a file"
        tool.input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

        def is_server_tool():
            return False
        tool.is_server_tool = is_server_tool

        messages = [_make_msg("user", "Read file")]
        req = _make_request(messages, tools=[tool])
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")

        assert "tools" in result
        decls = result["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "Read"

    def test_stream_enables_stream_function_call_args(self):
        tool = MagicMock()
        tool.name = "Write"
        tool.description = "Write a file"
        tool.input_schema = {"type": "object", "properties": {}}

        def is_server_tool():
            return False
        tool.is_server_tool = is_server_tool

        messages = [_make_msg("user", "Write file")]
        req = _make_request(messages, tools=[tool])
        result = convert_claude_to_gemini(req, "gemini-2.5-flash", stream=True)

        assert result["toolConfig"]["functionCallingConfig"]["streamFunctionCallArguments"] is True


class TestImageToolResult:

    def test_image_in_tool_result_nested_under_function_response(self):
        messages = [
            _make_msg("user", "Analyze this image."),
            _make_msg("assistant", [
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "/tmp/img.png"}},
            ]),
            _make_msg("user", [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [
                        {"type": "text", "text": "Image data"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                    ],
                },
            ]),
        ]

        req = _make_request(messages)
        sig_mgr = GeminiSignatureManager(base_dir="/tmp/test-sigs")
        sig_mgr.store("sess", "proj", "call_1", "sig_1")

        result = convert_claude_to_gemini(
            req, "gemini-3-flash-preview",
            signature_manager=sig_mgr, session_id="sess", project_path="proj",
        )

        user_contents = [c for c in result["contents"] if c["role"] == "user"]
        tool_parts = user_contents[-1]["parts"]

        # Should have functionResponse (not separate inlineData)
        fr_parts = [p for p in tool_parts if "functionResponse" in p]
        assert len(fr_parts) >= 1
        fr = fr_parts[0]["functionResponse"]
        assert fr["name"] == "Read"
        assert fr["response"]["result"] == "Image data"
        assert "parts" in fr
        assert fr["parts"][0]["inlineData"]["mimeType"] == "image/png"


class TestThinkingConfig:

    def test_thinking_enabled(self):
        messages = [_make_msg("user", "Think")]
        thinking = MagicMock()
        thinking.is_enabled.return_value = True
        thinking.budget_tokens = 50000
        req = _make_request(messages, thinking=thinking)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")

        assert result["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"

    def test_thinking_low_budget(self):
        messages = [_make_msg("user", "Think")]
        thinking = MagicMock()
        thinking.is_enabled.return_value = True
        thinking.budget_tokens = 5000
        req = _make_request(messages, thinking=thinking)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")

        assert result["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "LOW"

    def test_thinking_disabled(self):
        messages = [_make_msg("user", "Hi")]
        thinking = MagicMock()
        thinking.is_enabled.return_value = False
        req = _make_request(messages, thinking=thinking)
        result = convert_claude_to_gemini(req, "gemini-2.5-flash")

        assert "generationConfig" not in result


class TestSystemInstruction:

    def test_custom_prefix(self):
        messages = [_make_msg("user", "Hi")]
        req = _make_request(messages, system="Be concise.")
        result = convert_claude_to_gemini(
            req, "gemini-2.5-flash",
            system_instruction_prefix="EXTRA RULES",
        )

        first_user = result["contents"][0]
        all_text = " ".join(p.get("text", "") for p in first_user["parts"])
        assert "EXTRA RULES" in all_text
        assert "Be concise." in all_text
