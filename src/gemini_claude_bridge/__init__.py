"""
gemini-claude-bridge: Production-grade Gemini ↔ Anthropic protocol converter.

Converts between Google Gemini API format and Anthropic Claude Messages API format,
handling streaming, tool calling, thinking signatures, and multimodal content.
"""

from .constants import Constants
from .models import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeMessage,
    ClaudeTool,
    ClaudeThinkingConfig,
    ClaudeUsage,
)
from .request_converter import convert_claude_to_gemini
from .response_converter import convert_gemini_to_claude
from .stream_converter import (
    GeminiStreamConverter,
    convert_gemini_stream_to_claude,
)
from .signature_manager import GeminiSignatureManager

__version__ = "0.1.0"

__all__ = [
    "Constants",
    "ClaudeMessagesRequest",
    "ClaudeMessagesResponse",
    "ClaudeMessage",
    "ClaudeTool",
    "ClaudeThinkingConfig",
    "ClaudeUsage",
    "convert_claude_to_gemini",
    "convert_gemini_to_claude",
    "GeminiStreamConverter",
    "convert_gemini_stream_to_claude",
    "GeminiSignatureManager",
]
