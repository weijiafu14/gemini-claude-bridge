"""
Tests for multimodal (image) tool result handling.

Verifies that images in tool_result are nested inside functionResponse.parts
(not as top-level sibling parts), which is required by Gemini API.
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


def _make_request(messages):
    req = MagicMock()
    req.messages = messages
    req.system = "You are helpful."
    req.stream = False
    req.model = "claude-sonnet-4-20250514"
    req.tools = None
    req.thinking = None
    return req


def _make_base64_image_block(data: str = "QUJDRA=="):
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": data,
        },
    }


class TestMultimodalToolResult:

    def test_single_image_nested_under_function_response(self):
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
                        {"type": "text", "text": "Image data for img.png"},
                        _make_base64_image_block(),
                    ],
                },
            ]),
        ]

        req = _make_request(messages)
        sig_mgr = GeminiSignatureManager(base_dir="/tmp/test-sigs-mm")
        sig_mgr.store("sess", "/proj", "call_1", "sig_1")

        result = convert_claude_to_gemini(
            req, "gemini-2.5-flash",
            signature_manager=sig_mgr, session_id="sess", project_path="/proj",
        )

        user_contents = [c for c in result["contents"] if c["role"] == "user"]
        tool_parts = user_contents[-1]["parts"]

        # Should be exactly 1 functionResponse part (no top-level inlineData)
        fr_parts = [p for p in tool_parts if "functionResponse" in p]
        assert len(fr_parts) == 1

        fr = fr_parts[0]["functionResponse"]
        assert fr["name"] == "Read"
        assert fr["response"]["result"] == "Image data for img.png"
        assert "parts" in fr
        assert len(fr["parts"]) == 1
        assert fr["parts"][0]["inlineData"]["mimeType"] == "image/png"

    def test_multiple_image_tool_results(self):
        messages = [
            _make_msg("user", "Analyze these images."),
            _make_msg("assistant", [
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "/tmp/a.png"}},
            ]),
            _make_msg("assistant", [
                {"type": "tool_use", "id": "call_2", "name": "Read", "input": {"file_path": "/tmp/b.png"}},
            ]),
            _make_msg("user", [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [
                        {"type": "text", "text": "Image A"},
                        _make_base64_image_block("QUJDRA=="),
                    ],
                },
            ]),
            _make_msg("user", [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": [
                        {"type": "text", "text": "Image B"},
                        _make_base64_image_block("RUZHSA=="),
                    ],
                },
            ]),
        ]

        req = _make_request(messages)
        sig_mgr = GeminiSignatureManager(base_dir="/tmp/test-sigs-mm2")
        sig_mgr.store("sess", "/proj", "call_1", "sig_1")
        sig_mgr.store("sess", "/proj", "call_2", "sig_2")

        result = convert_claude_to_gemini(
            req, "gemini-2.5-flash",
            signature_manager=sig_mgr, session_id="sess", project_path="/proj",
        )

        user_contents = [c for c in result["contents"] if c["role"] == "user"]
        merged_parts = user_contents[-1]["parts"]

        # Both should be functionResponse (merged consecutive user messages)
        fr_parts = [p for p in merged_parts if "functionResponse" in p]
        assert len(fr_parts) == 2
        assert all("parts" in p["functionResponse"] for p in fr_parts)
        assert all(len(p["functionResponse"]["parts"]) == 1 for p in fr_parts)

        # No top-level inlineData
        assert not any("inlineData" in p for p in merged_parts)
