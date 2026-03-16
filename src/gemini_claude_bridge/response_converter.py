"""
Gemini → Claude non-streaming response converter.

Converts a single Gemini API response into Anthropic Claude Messages API format.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional

from .models import ClaudeMessagesRequest
from .constants import Constants
from .signature_manager import GeminiSignatureManager

logger = logging.getLogger(__name__)


def convert_gemini_to_claude(
    gemini_response: Dict[str, Any],
    original_request: ClaudeMessagesRequest,
    signature_manager: Optional[GeminiSignatureManager] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a Gemini API response to Claude Messages API format.

    Args:
        gemini_response: Raw Gemini API response.
        original_request: The original Claude request (used for model name).
        signature_manager: Optional manager to persist thoughtSignatures.
        session_id: Session ID for signature storage.
        project_path: Project path for signature storage.

    Returns:
        Claude Messages API response dict.
    """
    response_id = gemini_response.get("responseId", f"resp_{uuid.uuid4().hex[:24]}")
    candidates = gemini_response.get("candidates", [])
    usage_metadata = gemini_response.get("usageMetadata", {})

    content: List[Dict[str, Any]] = []
    finish_reason = "STOP"
    has_function_call = False

    if candidates:
        candidate = candidates[0]
        candidate_content = candidate.get("content", {})
        parts = candidate_content.get("parts", [])
        finish_reason = candidate.get("finishReason", "STOP")
        finish_message = candidate.get("finishMessage", "")

        # Handle MALFORMED_FUNCTION_CALL
        if finish_reason == "MALFORMED_FUNCTION_CALL":
            error_text = f"[Gemini Error: {finish_reason}] {finish_message}"
            content.append({
                "type": Constants.CONTENT_TEXT,
                "text": error_text,
            })
            logger.warning(f"Gemini returned MALFORMED_FUNCTION_CALL: {finish_message}")

        for part in parts:
            if "text" in part:
                text = part.get("text", "")
                if text:
                    content.append({
                        "type": Constants.CONTENT_TEXT,
                        "text": text,
                    })

            elif "thought" in part:
                thought = part.get("thought", "")
                if thought:
                    content.append({
                        "type": Constants.CONTENT_THINKING,
                        "thinking": thought,
                    })

            elif "functionCall" in part:
                function_call = part.get("functionCall", {})
                func_name = function_call.get("name", "")
                func_args = function_call.get("args", {})

                if func_name:
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    content.append({
                        "type": Constants.CONTENT_TOOL_USE,
                        "id": tool_use_id,
                        "name": func_name,
                        "input": func_args if isinstance(func_args, dict) else {},
                    })
                    has_function_call = True

                    # Persist thoughtSignature
                    thought_signature = part.get("thoughtSignature")
                    if thought_signature and signature_manager:
                        effective_session = session_id or "default"
                        effective_path = project_path or "default"
                        signature_manager.store(
                            effective_session, effective_path,
                            tool_use_id, thought_signature,
                        )

            elif "inlineData" in part:
                inline_data = part.get("inlineData", {})
                mime_type = inline_data.get("mimeType", "image/png")
                content.append({
                    "type": Constants.CONTENT_TEXT,
                    "text": f"[Image generated: {mime_type}]",
                })

    if not content:
        content.append({"type": Constants.CONTENT_TEXT, "text": ""})

    if has_function_call:
        stop_reason = Constants.STOP_TOOL_USE
    else:
        stop_reason = _convert_finish_reason(finish_reason)

    if response_id.startswith("resp_"):
        message_id = f"msg_{response_id[5:]}"
    else:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

    return {
        "id": message_id,
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_metadata.get("promptTokenCount", 0),
            "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _convert_finish_reason(finish_reason: str) -> str:
    """Map Gemini finishReason to Claude stop_reason."""
    mapping = {
        "STOP": Constants.STOP_END_TURN,
        "MAX_TOKENS": Constants.STOP_MAX_TOKENS,
        "SAFETY": Constants.STOP_END_TURN,
        "RECITATION": Constants.STOP_END_TURN,
        "OTHER": Constants.STOP_END_TURN,
    }
    return mapping.get(finish_reason, Constants.STOP_END_TURN)
