"""
Pydantic data models for the Anthropic Claude Messages API.

These models define the request/response format that clients (e.g. Claude Code CLI)
send and receive. The bridge converts between these and the Gemini API format.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional, Union, Literal


class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]


class ClaudeContentBlockThinking(BaseModel):
    """Thinking/reasoning content block.

    In multi-turn conversations, thinking blocks should be passed back to the API:
    - Normal conversation: recommended (API auto-filters)
    - After tool calls: required (maintains reasoning continuity)
    """
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None


class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[
        ClaudeContentBlockText,
        ClaudeContentBlockImage,
        ClaudeContentBlockToolUse,
        ClaudeContentBlockToolResult,
        ClaudeContentBlockThinking,
    ]]]


class ClaudeTool(BaseModel):
    """Claude tool definition.

    Supports two types:
    1. Regular tools: have name + input_schema (e.g. Bash, Read, Edit)
    2. Server tools: have type field (e.g. web_search), no input_schema
    """
    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    max_uses: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    user_location: Optional[Dict[str, Any]] = None

    def is_server_tool(self) -> bool:
        return self.type is not None and self.input_schema is None

    model_config = ConfigDict(extra="allow")


class ClaudeThinkingConfig(BaseModel):
    """Thinking/reasoning configuration.

    Supports:
    1. Claude native: {"type": "enabled", "budget_tokens": 10000}
    2. Simplified: {"enabled": true}
    """
    type: Optional[Literal["enabled", "disabled", "adaptive"]] = None
    budget_tokens: Optional[int] = None
    enabled: Optional[bool] = None

    def is_enabled(self) -> bool:
        if self.type == "enabled":
            return True
        if self.type == "disabled":
            return False
        return self.enabled is True


class ClaudeMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None


class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None


class ClaudeUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ClaudeMessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Dict[str, Any]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: ClaudeUsage
