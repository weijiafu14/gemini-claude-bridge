"""
Gemini → Claude streaming response converter.

Converts Gemini NDJSON streaming responses into Claude SSE (Server-Sent Events) format.

Handles:
- Text streaming with delta accumulation
- Tool calling with complete args (non-streaming)
- Tool calling with partialArgs (streaming) + jsonPath accumulation
- thoughtSignature persistence
- Incremental JSON delta for large-content tools (Write, Edit)
"""

import json
import re
import uuid
import copy
import logging
from typing import Dict, Any, Optional, AsyncIterator, List, Union, Set

from .constants import Constants
from .signature_manager import GeminiSignatureManager

logger = logging.getLogger(__name__)


def parse_json_path(json_path: str) -> List[Union[str, int]]:
    """
    Parse a jsonPath string into path segments.

    Examples:
        "$.field" -> ["field"]
        "$.array[0].field" -> ["array", 0, "field"]
        "$.todos[0].content" -> ["todos", 0, "content"]
    """
    if not json_path or json_path == "$":
        return []

    path = json_path.lstrip("$").lstrip(".")
    if not path:
        return []

    segments: List[Union[str, int]] = []
    pattern = r'([^.\[\]]+)|\[(\d+)\]'
    for match in re.finditer(pattern, path):
        if match.group(1):
            segments.append(match.group(1))
        elif match.group(2):
            segments.append(int(match.group(2)))

    return segments


def set_nested_value(obj: Dict[str, Any], path: List[Union[str, int]], value: Any) -> None:
    """Set a value at a nested path, auto-creating intermediate structures."""
    if not path:
        return

    current = obj
    for i, segment in enumerate(path[:-1]):
        next_segment = path[i + 1]
        if isinstance(segment, int):
            while len(current) <= segment:
                current.append({} if isinstance(next_segment, str) else [])
            if current[segment] is None:
                current[segment] = {} if isinstance(next_segment, str) else []
            current = current[segment]
        else:
            if segment not in current:
                current[segment] = [] if isinstance(next_segment, int) else {}
            elif current[segment] is None:
                current[segment] = [] if isinstance(next_segment, int) else {}
            current = current[segment]

    final = path[-1]
    if isinstance(final, int):
        while len(current) <= final:
            current.append(None)
        current[final] = value
    else:
        current[final] = value


def append_nested_value(obj: Dict[str, Any], path: List[Union[str, int]], value: str) -> None:
    """Append a string value at a nested path (for streaming accumulation)."""
    if not path:
        return

    current = obj
    for i, segment in enumerate(path[:-1]):
        next_segment = path[i + 1]
        if isinstance(segment, int):
            while len(current) <= segment:
                current.append({} if isinstance(next_segment, str) else [])
            if current[segment] is None:
                current[segment] = {} if isinstance(next_segment, str) else []
            current = current[segment]
        else:
            if segment not in current:
                current[segment] = [] if isinstance(next_segment, int) else {}
            elif current[segment] is None:
                current[segment] = [] if isinstance(next_segment, int) else {}
            current = current[segment]

    final = path[-1]
    if isinstance(final, int):
        while len(current) <= final:
            current.append("")
        if current[final] is None:
            current[final] = ""
        current[final] += value
    else:
        if final not in current:
            current[final] = ""
        elif current[final] is None:
            current[final] = ""
        current[final] += value


class GeminiStreamConverter:
    """
    Converts Gemini NDJSON stream into Claude SSE events.

    Args:
        request_id: Request tracking ID.
        model: Gemini model name.
        original_request_model: Original Claude model name (returned in SSE).
        signature_manager: Optional signature manager for thoughtSignature persistence.
        session_id: Session ID for signature storage.
        project_path: Project path for signature storage.
        incremental_tools: Set of tool names that should use incremental JSON delta
            mode (for large content like Write/Edit). Defaults to common edit tools.
    """

    # Default tools that benefit from incremental streaming
    DEFAULT_INCREMENTAL_TOOLS = {
        'write', 'edit', 'multiedit', 'notebookedit',
        'write_file', 'writefile',
    }

    def __init__(
        self,
        request_id: str,
        model: str,
        original_request_model: str,
        signature_manager: Optional[GeminiSignatureManager] = None,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        incremental_tools: Optional[Set[str]] = None,
    ):
        self.request_id = request_id
        self.model = model
        self.original_request_model = original_request_model
        self.signature_manager = signature_manager
        self.session_id = session_id
        self.project_path = project_path
        self.incremental_tools = incremental_tools or self.DEFAULT_INCREMENTAL_TOOLS

        # State tracking
        self.message_id: str = f"msg_{uuid.uuid4().hex[:24]}"
        self.content_index: int = 0
        self.has_sent_message_start: bool = False
        self.has_sent_text_block_start: bool = False
        self.has_function_call: bool = False
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        # Text accumulation (Gemini sends cumulative text)
        self.accumulated_text: str = ""

        # Streaming function_call state
        self.current_fc_id: Optional[str] = None
        self.current_fc_name: Optional[str] = None
        self.current_fc_started: bool = False
        self.current_fc_args_buffer: str = ""

        # partialArgs accumulation
        self.partial_args_obj: Dict[str, Any] = {}
        self.last_sent_json: str = ""
        self.last_sent_str: str = ""
        self.last_sent_obj: Dict[str, Any] = {}
        self.cumulative_sent: str = ""

        # Incremental mode (set once per tool call, not toggled mid-stream)
        self.use_incremental_mode: bool = False

        # Field streaming state tracking
        self.field_streaming_state: Dict[str, bool] = {}
        self.completed_fields: Set[str] = set()

    def _build_message_start(self) -> str:
        return json.dumps({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.original_request_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.usage.get("input_tokens", 0),
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        })

    def _build_content_block_start(self, block_type: str, extra: Optional[Dict] = None) -> str:
        block: Dict[str, Any] = {"type": block_type}
        if block_type == "thinking":
            block["thinking"] = ""
        elif block_type == "text":
            block["text"] = ""
        elif block_type == "tool_use":
            block["id"] = extra.get("id", "") if extra else ""
            block["name"] = extra.get("name", "") if extra else ""
            block["input"] = {}
        return json.dumps({
            "type": "content_block_start",
            "index": self.content_index,
            "content_block": block,
        })

    def _build_content_block_delta(self, delta_type: str, content: Any) -> str:
        delta: Dict[str, Any] = {"type": delta_type}
        if delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content
        return json.dumps({
            "type": "content_block_delta",
            "index": self.content_index,
            "delta": delta,
        })

    def _build_content_block_stop(self) -> str:
        return json.dumps({
            "type": "content_block_stop",
            "index": self.content_index,
        })

    def _build_message_delta(self, stop_reason: str) -> str:
        return json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": self.usage.get("input_tokens", 0),
                "output_tokens": self.usage.get("output_tokens", 0),
            },
        })

    def _build_message_stop(self) -> str:
        return json.dumps({"type": "message_stop"})

    async def convert_stream(
        self,
        gemini_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """
        Convert a Gemini NDJSON stream into Claude SSE events.

        Args:
            gemini_stream: Async iterator yielding NDJSON lines.

        Yields:
            Claude SSE event strings (e.g. "event: message_start\\ndata: {...}\\n\\n").
        """
        line_count = 0

        async for line in gemini_stream:
            line = line.strip()
            if not line:
                continue
            line_count += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[{self.request_id}] Failed to parse NDJSON line #{line_count}: {e}")
                continue

            async for event in self._process_ndjson_line(data, line_count):
                yield event

        # Stream ended - send closing events
        if self.has_sent_message_start:
            if self.has_sent_text_block_start:
                yield f"event: content_block_stop\ndata: {self._build_content_block_stop()}\n\n"
                self.content_index += 1
                self.has_sent_text_block_start = False

            if self.current_fc_started:
                yield f"event: content_block_stop\ndata: {self._build_content_block_stop()}\n\n"
                self.content_index += 1
                self.current_fc_started = False

            stop_reason = Constants.STOP_TOOL_USE if self.has_function_call else Constants.STOP_END_TURN
            yield f"event: message_delta\ndata: {self._build_message_delta(stop_reason)}\n\n"
            yield f"event: message_stop\ndata: {self._build_message_stop()}\n\n"

            logger.info(f"[{self.request_id}] Stream completed: {line_count} lines, {self.content_index} blocks")

    async def _process_ndjson_line(self, data: Dict[str, Any], line_num: int) -> AsyncIterator[str]:
        """Process a single NDJSON line."""
        if not self.has_sent_message_start:
            self.has_sent_message_start = True
            yield f"event: message_start\ndata: {self._build_message_start()}\n\n"

        candidates = data.get("candidates", [])
        if not candidates:
            return

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        finish_reason = candidate.get("finishReason")

        for part in parts:
            # Text
            if "text" in part:
                text = part["text"]
                if text.startswith(self.accumulated_text):
                    delta_text = text[len(self.accumulated_text):]
                    self.accumulated_text = text
                else:
                    delta_text = text
                    self.accumulated_text = text

                if delta_text:
                    if not self.has_sent_text_block_start:
                        self.has_sent_text_block_start = True
                        yield f"event: content_block_start\ndata: {self._build_content_block_start('text')}\n\n"
                    yield f"event: content_block_delta\ndata: {self._build_content_block_delta('text_delta', delta_text)}\n\n"

                # Save thoughtSignature from text part (final line)
                thought_signature = part.get("thoughtSignature")
                if thought_signature and finish_reason == "STOP":
                    if self.signature_manager and self.session_id and self.project_path:
                        self.signature_manager.store(
                            self.session_id, self.project_path,
                            self.message_id, thought_signature,
                        )

            # Function call
            if "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name", "")
                args = fc.get("args")
                partial_args = fc.get("partialArgs")
                will_continue = fc.get("willContinue", False)

                # Determine call_id
                if fc.get("id"):
                    call_id = fc.get("id")
                elif name and not self.current_fc_started:
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    self.current_fc_id = call_id
                    self.current_fc_name = name
                elif self.current_fc_id:
                    call_id = self.current_fc_id
                else:
                    call_id = f"call_{uuid.uuid4().hex[:8]}"

                self.has_function_call = True

                # Close text block if open
                if self.has_sent_text_block_start:
                    yield f"event: content_block_stop\ndata: {self._build_content_block_stop()}\n\n"
                    self.content_index += 1
                    self.has_sent_text_block_start = False

                # Save thoughtSignature from functionCall part
                thought_signature = part.get("thoughtSignature")
                if thought_signature and self.signature_manager and self.session_id and self.project_path:
                    self.signature_manager.store(
                        self.session_id, self.project_path,
                        call_id, thought_signature,
                    )

                # Case 1: Streaming partialArgs
                if partial_args is not None:
                    if not self.current_fc_started and name:
                        self.current_fc_id = call_id
                        self.current_fc_name = name
                        self.current_fc_started = True
                        self.partial_args_obj = {}
                        self.last_sent_json = ""
                        self.last_sent_str = ""
                        self.cumulative_sent = ""
                        self.last_sent_obj = {}
                        self.field_streaming_state = {}
                        self.completed_fields = set()

                        name_lower = name.lower()
                        self.use_incremental_mode = name_lower in self.incremental_tools

                        yield f"event: content_block_start\ndata: {self._build_content_block_start('tool_use', {'id': call_id, 'name': name})}\n\n"

                    # Accumulate partialArgs
                    for pa in partial_args:
                        json_path = pa.get("jsonPath", "")
                        field_will_continue = pa.get("willContinue", False)
                        string_value = pa.get("stringValue")
                        bool_value = pa.get("boolValue")
                        int_value = pa.get("intValue")
                        double_value = pa.get("doubleValue")

                        if not json_path:
                            continue
                        path_segments = parse_json_path(json_path)
                        if not path_segments:
                            continue

                        field_name = path_segments[-1] if path_segments else ""

                        if string_value is not None:
                            if string_value:
                                append_nested_value(self.partial_args_obj, path_segments, string_value)
                            if field_will_continue:
                                self.field_streaming_state[field_name] = True
                            else:
                                self.field_streaming_state[field_name] = False
                                self.completed_fields.add(field_name)
                        elif bool_value is not None:
                            set_nested_value(self.partial_args_obj, path_segments, bool_value)
                            self.completed_fields.add(field_name)
                        elif int_value is not None:
                            set_nested_value(self.partial_args_obj, path_segments, int_value)
                            self.completed_fields.add(field_name)
                        elif double_value is not None:
                            set_nested_value(self.partial_args_obj, path_segments, double_value)
                            self.completed_fields.add(field_name)

                    # Send incremental JSON delta for large-content tools
                    if self.use_incremental_mode and self.partial_args_obj:
                        json_parts = ["{"]
                        sorted_keys = sorted(self.partial_args_obj.keys())
                        for ki, key in enumerate(sorted_keys):
                            value = self.partial_args_obj[key]
                            if ki > 0:
                                json_parts.append(", ")
                            json_parts.append(f'"{key}": ')

                            if isinstance(value, str):
                                escaped = json.dumps(value, ensure_ascii=False)[1:-1]
                                is_streaming = self.field_streaming_state.get(key, False)
                                if is_streaming:
                                    json_parts.append(f'"{escaped}')
                                else:
                                    json_parts.append(f'"{escaped}"')
                            elif isinstance(value, bool):
                                json_parts.append("true" if value else "false")
                            elif isinstance(value, (int, float)):
                                json_parts.append(str(value))
                            else:
                                json_parts.append(json.dumps(value, ensure_ascii=False))

                        current_partial = "".join(json_parts)

                        if not self.cumulative_sent:
                            delta = current_partial
                            self.cumulative_sent = current_partial
                        elif len(current_partial) > len(self.cumulative_sent):
                            delta = current_partial[len(self.cumulative_sent):]
                            self.cumulative_sent = current_partial
                        else:
                            delta = ""

                        if delta:
                            yield f"event: content_block_delta\ndata: {self._build_content_block_delta('input_json_delta', delta)}\n\n"
                        self.last_sent_obj = copy.deepcopy(self.partial_args_obj)

                    # partialArgs stream ended
                    if not will_continue and self.current_fc_started:
                        if self.use_incremental_mode and self.cumulative_sent:
                            yield f"event: content_block_delta\ndata: {self._build_content_block_delta('input_json_delta', '}')}\n\n"
                        elif not self.use_incremental_mode and self.partial_args_obj:
                            final_json = json.dumps(self.partial_args_obj, ensure_ascii=False, sort_keys=True)
                            yield f"event: content_block_delta\ndata: {self._build_content_block_delta('input_json_delta', final_json)}\n\n"

                        yield f"event: content_block_stop\ndata: {self._build_content_block_stop()}\n\n"
                        self.content_index += 1
                        self._reset_fc_state()

                # Case 1.5: Empty functionCall = stream end (when no willContinue)
                elif not partial_args and not args and not name and self.current_fc_started and not will_continue:
                    if self.use_incremental_mode and self.cumulative_sent:
                        yield f"event: content_block_delta\ndata: {self._build_content_block_delta('input_json_delta', '}')}\n\n"
                    elif not self.use_incremental_mode and self.partial_args_obj:
                        final_json = json.dumps(self.partial_args_obj, ensure_ascii=False, sort_keys=True)
                        yield f"event: content_block_delta\ndata: {self._build_content_block_delta('input_json_delta', final_json)}\n\n"

                    yield f"event: content_block_stop\ndata: {self._build_content_block_stop()}\n\n"
                    self.content_index += 1
                    self._reset_fc_state()

                # Case 2: Non-streaming (complete args at once)
                elif args is not None:
                    async for event in self._emit_function_call({
                        "call_id": call_id,
                        "name": name,
                        "arguments": args,
                    }):
                        yield event

                # Case 3: Stream start (name + willContinue, waiting for partialArgs)
                elif name and will_continue and not self.current_fc_started:
                    self.current_fc_id = call_id
                    self.current_fc_name = name
                    self.current_fc_started = True
                    self.partial_args_obj = {}
                    self.last_sent_json = ""
                    self.last_sent_str = ""
                    self.cumulative_sent = ""
                    self.last_sent_obj = {}
                    self.field_streaming_state = {}
                    self.completed_fields = set()

                    name_lower = name.lower()
                    self.use_incremental_mode = name_lower in self.incremental_tools

                    yield f"event: content_block_start\ndata: {self._build_content_block_start('tool_use', {'id': call_id, 'name': name})}\n\n"

                # Case 4: No-args tool call (name only, no args, no willContinue)
                elif name and not args and not will_continue and not self.current_fc_started:
                    async for event in self._emit_function_call({
                        "call_id": call_id,
                        "name": name,
                        "arguments": {},
                    }):
                        yield event

        # Usage metadata (usually in the last line)
        usage_metadata = data.get("usageMetadata", {})
        if usage_metadata:
            self.usage = {
                "input_tokens": usage_metadata.get("promptTokenCount", 0),
                "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
            }

    def _reset_fc_state(self):
        """Reset function call streaming state."""
        self.current_fc_started = False
        self.current_fc_id = None
        self.current_fc_name = None
        self.partial_args_obj = {}
        self.last_sent_json = ""
        self.last_sent_str = ""
        self.cumulative_sent = ""
        self.last_sent_obj = {}
        self.use_incremental_mode = False
        self.field_streaming_state = {}
        self.completed_fields = set()

    async def _emit_function_call(self, fc: Dict[str, Any]) -> AsyncIterator[str]:
        """Emit SSE events for a complete function call."""
        call_id = fc.get("call_id")
        name = fc.get("name")
        arguments = fc.get("arguments", {})

        if isinstance(arguments, str):
            try:
                input_dict = json.loads(arguments)
            except json.JSONDecodeError:
                input_dict = {"raw": arguments}
        else:
            input_dict = arguments

        yield f"event: content_block_start\ndata: {self._build_content_block_start('tool_use', {'id': call_id, 'name': name})}\n\n"

        input_json = json.dumps(input_dict, ensure_ascii=False)
        yield f"event: content_block_delta\ndata: {self._build_content_block_delta('input_json_delta', input_json)}\n\n"

        yield f"event: content_block_stop\ndata: {self._build_content_block_stop()}\n\n"
        self.content_index += 1


async def convert_gemini_stream_to_claude(
    gemini_stream: AsyncIterator[str],
    request_id: str,
    model: str,
    original_request_model: str,
    signature_manager: Optional[GeminiSignatureManager] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    incremental_tools: Optional[Set[str]] = None,
) -> AsyncIterator[str]:
    """
    Convenience function: convert Gemini NDJSON stream to Claude SSE stream.

    Args:
        gemini_stream: Async iterator yielding NDJSON lines.
        request_id: Request tracking ID.
        model: Gemini model name.
        original_request_model: Original Claude model name.
        signature_manager: Optional signature manager.
        session_id: Session ID.
        project_path: Project path.
        incremental_tools: Tool names for incremental streaming mode.

    Yields:
        Claude SSE event strings.
    """
    converter = GeminiStreamConverter(
        request_id=request_id,
        model=model,
        original_request_model=original_request_model,
        signature_manager=signature_manager,
        session_id=session_id,
        project_path=project_path,
        incremental_tools=incremental_tools,
    )
    async for event in converter.convert_stream(gemini_stream):
        yield event
