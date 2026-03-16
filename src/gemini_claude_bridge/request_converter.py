"""
Claude → Gemini request converter.

Converts Anthropic Claude Messages API requests into Google Gemini API format.

Key conversions:
- messages → contents (with parts)
- assistant role → model role
- tool_use → functionCall
- tool_result → functionResponse
- thinking config → thinkingConfig
- tools → functionDeclarations
- thoughtSignature recovery for multi-turn tool calling
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Set, Tuple

from .models import ClaudeMessagesRequest, ClaudeMessage
from .constants import Constants
from .signature_manager import GeminiSignatureManager
from .schema_cleaner import clean_schema_for_gemini
from .image_utils import (
    convert_image_source,
    parse_tool_result_content_with_images,
)

logger = logging.getLogger(__name__)

DUMMY_THOUGHT_SIGNATURE = "context_engineering_is_the_way_to_go"


def convert_claude_to_gemini(
    request: ClaudeMessagesRequest,
    model: str,
    signature_manager: Optional[GeminiSignatureManager] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    stream: bool = False,
    system_instruction_prefix: Optional[str] = None,
    multimodal_tool_results: bool = True,
) -> Dict[str, Any]:
    """
    Convert a Claude Messages API request to Gemini API format.

    Args:
        request: Claude format request.
        model: Target Gemini model name (e.g. "gemini-2.5-pro").
        signature_manager: Optional signature manager for thoughtSignature recovery.
        session_id: Session ID for signature lookup.
        project_path: Project path for signature lookup.
        stream: Whether to enable streaming.
        system_instruction_prefix: Optional text to prepend to system instruction.
        multimodal_tool_results: If True, embed images in functionResponse.parts.
            Set to False for models that don't support multimodal function responses
            (e.g. gemini-2.5-flash). Images will be described as text instead.

    Returns:
        Gemini API format request dict.
    """
    logger.info(f"Converting Claude request to Gemini format: model={model}, stream={stream}")

    # 1. Build tool_id -> tool_name mapping (needed for tool_result conversion)
    tool_id_to_name: Dict[str, str] = _build_tool_id_to_name_map(request.messages)

    # 2. Classify tool IDs with missing signatures
    skipped_tool_ids: Set[str] = set()
    dummy_signature_tool_ids: Set[str] = set()
    if signature_manager:
        skipped_tool_ids, dummy_signature_tool_ids = _classify_missing_signature_tool_ids(
            request.messages, signature_manager, session_id, project_path
        )
        if skipped_tool_ids:
            logger.warning(f"Skipping {len(skipped_tool_ids)} tool calls with missing signature")
        if dummy_signature_tool_ids:
            logger.info(f"Replaying {len(dummy_signature_tool_ids)} tool calls with dummy signature")

    # Auto-detect multimodal support if not explicitly set
    # gemini-3-* supports multimodal function responses, gemini-2.5-* does not
    if multimodal_tool_results:
        model_lower = model.lower() if model else ""
        if "2.5" in model_lower or "2.0" in model_lower:
            multimodal_tool_results = False
            logger.info(f"Auto-disabled multimodal_tool_results for model: {model}")

    # 3. Convert messages to contents
    contents: List[Dict[str, Any]] = []
    for msg in request.messages:
        converted = _convert_message(
            msg, tool_id_to_name, signature_manager,
            session_id, project_path,
            skipped_tool_ids, dummy_signature_tool_ids,
            multimodal_tool_results=multimodal_tool_results,
        )
        if converted:
            contents.append(converted)

    # 4. Post-processing: split mixed model messages, merge consecutive user messages
    contents = _split_mixed_model_messages(contents)
    contents = _merge_consecutive_user_messages(contents)

    # 5. Inject system instruction into first user message
    system_text = _extract_system_text(request.system) if request.system else ""
    if system_instruction_prefix:
        system_text = (system_instruction_prefix + "\n\n" + system_text) if system_text else system_instruction_prefix

    if system_text.strip() and contents:
        for content in contents:
            if content.get("role") == "user":
                parts = content.get("parts", [])
                parts.insert(0, {"text": f"[System Instruction]\n{system_text}\n\n"})
                break

    # 6. Build request
    gemini_request: Dict[str, Any] = {
        "model": model,
        "contents": contents,
        "stream": stream,
    }

    # 7. Generation config (thinking)
    generation_config: Dict[str, Any] = {}
    if request.thinking and request.thinking.is_enabled():
        thinking_level = _determine_thinking_level(request.thinking.budget_tokens)
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        logger.info(f"Thinking enabled: level={thinking_level}")

    if generation_config:
        gemini_request["generationConfig"] = generation_config

    # 8. Tools (functionDeclarations)
    if request.tools:
        gemini_tools = _convert_tools_to_function_declarations(request.tools)
        if gemini_tools:
            gemini_request["tools"] = gemini_tools
            logger.info(f"Added {len(gemini_tools[0].get('functionDeclarations', []))} function declarations")

            # Enable streamFunctionCallArguments for Gemini models
            model_lower = model.lower() if model else ""
            is_gemini = "pro" in model_lower or "flash" in model_lower or "gemini" in model_lower
            if stream and is_gemini:
                gemini_request["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "AUTO",
                        "streamFunctionCallArguments": True,
                    }
                }
                logger.info("Enabled streamFunctionCallArguments")

    return gemini_request


def _convert_tools_to_function_declarations(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert Claude tools to Gemini functionDeclarations format."""
    if not tools:
        return []

    function_declarations: List[Dict[str, Any]] = []

    for tool in tools:
        if hasattr(tool, "name"):
            name = tool.name
            description = getattr(tool, "description", "") or ""
            input_schema = getattr(tool, "input_schema", None)
        elif isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("input_schema")
        else:
            continue

        if not name:
            continue

        # Skip server tools (web_search, etc.) - they don't have input_schema
        if hasattr(tool, "is_server_tool") and tool.is_server_tool():
            continue

        function_decl: Dict[str, Any] = {
            "name": name,
            "description": description,
        }

        if input_schema:
            if hasattr(input_schema, "model_dump"):
                function_decl["parameters"] = input_schema.model_dump()
            elif isinstance(input_schema, dict):
                function_decl["parameters"] = input_schema.copy()
            else:
                function_decl["parameters"] = {"type": "object", "properties": {}}
        else:
            function_decl["parameters"] = {"type": "object", "properties": {}}

        clean_schema_for_gemini(function_decl["parameters"])
        function_declarations.append(function_decl)

    if not function_declarations:
        return []

    return [{"functionDeclarations": function_declarations}]


def _build_tool_id_to_name_map(messages: List[Any]) -> Dict[str, str]:
    """Build tool_use_id -> tool_name mapping from message history."""
    mapping: Dict[str, str] = {}

    for msg in messages:
        content = getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else None)
        if not content or not isinstance(content, list):
            continue

        for block in content:
            block_type = getattr(block, 'type', None) or (block.get('type') if isinstance(block, dict) else None)
            if block_type == Constants.CONTENT_TOOL_USE:
                tool_id = getattr(block, 'id', None) or (block.get('id') if isinstance(block, dict) else '')
                tool_name = getattr(block, 'name', None) or (block.get('name') if isinstance(block, dict) else '')
                if tool_id and tool_name:
                    mapping[tool_id] = tool_name

    return mapping


def _classify_missing_signature_tool_ids(
    messages: List[Any],
    signature_manager: GeminiSignatureManager,
    session_id: Optional[str],
    project_path: Optional[str],
) -> Tuple[Set[str], Set[str]]:
    """
    Classify tool IDs with missing thoughtSignatures.

    When a request is aborted mid-stream, the proxy may not have stored the signature,
    but the CLI may have recorded the tool call. On the next request these tool calls
    have no signature and Gemini would reject them.

    Returns:
        (skipped_ids, dummy_signature_ids):
        - skipped_ids: tool calls without a matching tool_result (incomplete, skip entirely)
        - dummy_signature_ids: tool calls with a tool_result (complete, use dummy signature)
    """
    skipped_ids: Set[str] = set()
    dummy_ids: Set[str] = set()
    tool_result_ids: Set[str] = set()

    # Collect all tool_result IDs
    for msg in messages:
        role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
        if role != Constants.ROLE_USER:
            continue
        content = getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else None)
        if not content or not isinstance(content, list):
            continue
        for block in content:
            block_type = getattr(block, 'type', None) or (block.get('type') if isinstance(block, dict) else None)
            if block_type == Constants.CONTENT_TOOL_RESULT:
                tool_use_id = getattr(block, 'tool_use_id', None) or (block.get('tool_use_id') if isinstance(block, dict) else '')
                if tool_use_id:
                    tool_result_ids.add(tool_use_id)

    effective_session = session_id or "default"
    effective_path = project_path or "default"

    # Check assistant tool_use blocks for missing signatures
    for msg in messages:
        role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
        if role != Constants.ROLE_ASSISTANT:
            continue
        content = getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else None)
        if not content or not isinstance(content, list):
            continue
        for block in content:
            block_type = getattr(block, 'type', None) or (block.get('type') if isinstance(block, dict) else None)
            if block_type == Constants.CONTENT_TOOL_USE:
                tool_id = getattr(block, 'id', None) or (block.get('id') if isinstance(block, dict) else '')
                if tool_id:
                    sig = signature_manager.get(effective_session, effective_path, tool_id)
                    if not sig:
                        if tool_id in tool_result_ids:
                            dummy_ids.add(tool_id)
                        else:
                            skipped_ids.add(tool_id)

    return skipped_ids, dummy_ids


def _extract_system_text(system: Union[str, List[Any]]) -> str:
    """Extract plain text from system field."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        text_parts = []
        for block in system:
            if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                text_parts.append(block.get("text", ""))
        return "\n\n".join(text_parts)
    return ""


def _determine_thinking_level(budget_tokens: Optional[int]) -> str:
    """Map budget_tokens to Gemini thinking level."""
    if budget_tokens is None:
        return "HIGH"
    return "LOW" if budget_tokens < 10000 else "HIGH"


def _convert_message(
    msg: ClaudeMessage,
    tool_id_to_name: Optional[Dict[str, str]] = None,
    signature_manager: Optional[GeminiSignatureManager] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    skipped_tool_ids: Optional[Set[str]] = None,
    dummy_signature_tool_ids: Optional[Set[str]] = None,
    multimodal_tool_results: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convert a single Claude message to Gemini format."""
    role = "model" if msg.role == Constants.ROLE_ASSISTANT else msg.role

    parts = _convert_content_to_parts(
        msg.content, tool_id_to_name, signature_manager,
        session_id, project_path,
        skipped_tool_ids, dummy_signature_tool_ids,
        multimodal_tool_results=multimodal_tool_results,
    )

    if not parts:
        return None

    return {"role": role, "parts": parts}


def _convert_content_to_parts(
    content: Any,
    tool_id_to_name: Optional[Dict[str, str]] = None,
    signature_manager: Optional[GeminiSignatureManager] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    skipped_tool_ids: Optional[Set[str]] = None,
    dummy_signature_tool_ids: Optional[Set[str]] = None,
    multimodal_tool_results: bool = True,
) -> List[Dict[str, Any]]:
    """Convert Claude content blocks to Gemini parts."""
    parts: List[Dict[str, Any]] = []
    tool_id_to_name = tool_id_to_name or {}
    skipped_tool_ids = skipped_tool_ids or set()
    dummy_signature_tool_ids = dummy_signature_tool_ids or set()

    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return parts

    for block in content:
        block_type = getattr(block, 'type', None) or (block.get('type') if isinstance(block, dict) else None)

        if block_type == Constants.CONTENT_TEXT:
            text = getattr(block, 'text', None) or (block.get('text') if isinstance(block, dict) else '')
            if text:
                parts.append({"text": text})

        elif block_type == Constants.CONTENT_THINKING:
            # Thinking blocks are not sent back to Gemini
            pass

        elif block_type == Constants.CONTENT_IMAGE:
            source = getattr(block, 'source', None) or (block.get('source') if isinstance(block, dict) else None)
            if source:
                image_part = convert_image_source(source)
                if image_part:
                    parts.append(image_part)

        elif block_type == Constants.CONTENT_TOOL_USE:
            tool_name = getattr(block, 'name', None) or (block.get('name') if isinstance(block, dict) else '')
            tool_id = getattr(block, 'id', None) or (block.get('id') if isinstance(block, dict) else '')
            tool_input = getattr(block, 'input', None) or (block.get('input') if isinstance(block, dict) else {})

            if tool_id in skipped_tool_ids:
                continue

            if tool_name:
                fc_part: Dict[str, Any] = {
                    "functionCall": {
                        "name": tool_name,
                        "args": tool_input if isinstance(tool_input, dict) else {},
                    }
                }

                # Recover thoughtSignature
                thought_signature = None
                if tool_id and signature_manager and session_id and project_path:
                    thought_signature = signature_manager.get(
                        session_id or "default",
                        project_path or "default",
                        tool_id,
                    )

                if not thought_signature and tool_id in dummy_signature_tool_ids:
                    thought_signature = DUMMY_THOUGHT_SIGNATURE

                if thought_signature:
                    fc_part["thoughtSignature"] = thought_signature

                parts.append(fc_part)

        elif block_type == Constants.CONTENT_TOOL_RESULT:
            tool_use_id = getattr(block, 'tool_use_id', None) or (block.get('tool_use_id') if isinstance(block, dict) else '')
            tool_content = getattr(block, 'content', None) or (block.get('content') if isinstance(block, dict) else '')

            if tool_use_id in skipped_tool_ids:
                continue

            tool_name = tool_id_to_name.get(tool_use_id, tool_use_id)

            if multimodal_tool_results:
                text_result, image_parts_from_result = parse_tool_result_content_with_images(tool_content)
            else:
                # For models that don't support multimodal function responses,
                # parse as text only (images become "[Image: ...]" placeholders)
                from .image_utils import parse_tool_result_content
                text_result = parse_tool_result_content(tool_content)
                image_parts_from_result = []

            function_response: Dict[str, Any] = {
                "name": tool_name,
                "response": {"result": text_result},
            }

            if image_parts_from_result:
                function_response["parts"] = image_parts_from_result

            parts.append({"functionResponse": function_response})

    return parts


def _split_mixed_model_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split model messages that mix text and functionCall parts.

    Gemini requires that parts with thoughtSignature and parts without
    are not mixed in the same message.
    """
    try:
        expanded: List[Dict[str, Any]] = []
        for content in contents:
            if content.get("role") != "model":
                expanded.append(content)
                continue

            parts = content.get("parts", [])
            if not parts:
                expanded.append(content)
                continue

            text_parts = [p for p in parts if "text" in p and "functionCall" not in p]
            fc_parts = [p for p in parts if "functionCall" in p]
            other_parts = [p for p in parts if "text" not in p and "functionCall" not in p]

            if text_parts and fc_parts:
                if text_parts or other_parts:
                    expanded.append({"role": "model", "parts": text_parts + other_parts})
                expanded.append({"role": "model", "parts": fc_parts})
            else:
                expanded.append(content)

        return expanded
    except Exception:
        return contents


def _merge_consecutive_user_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive user messages.

    Gemini requires functionCall and functionResponse counts to match.
    When Claude sends multiple user messages (each with one tool_result),
    they must be merged into a single user message.
    """
    if not contents:
        return contents

    try:
        merged: List[Dict[str, Any]] = []
        i = 0
        while i < len(contents):
            current = contents[i]
            if current.get("role") == "user":
                merged_parts = list(current.get("parts", []))
                j = i + 1
                while j < len(contents) and contents[j].get("role") == "user":
                    merged_parts.extend(contents[j].get("parts", []))
                    j += 1
                if j > i + 1:
                    merged.append({"role": "user", "parts": merged_parts})
                    i = j
                else:
                    merged.append(current)
                    i += 1
            else:
                merged.append(current)
                i += 1
        return merged
    except Exception:
        return contents
