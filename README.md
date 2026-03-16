# gemini-claude-bridge

Production-grade protocol converter between **Google Gemini API** and **Anthropic Claude Messages API**.

Use any Gemini model (2.5 Pro, 2.5 Flash, etc.) as a drop-in replacement for Claude — including with **Claude Code CLI**.

## Why this exists

LiteLLM has 3 open bugs in this area that break real-world usage:

| Problem | LiteLLM Issue | This library |
|---------|--------------|-------------|
| Multi-turn tool calling + thinking crashes | [#17949](https://github.com/BerriAI/litellm/issues/17949) (open 3+ months) | thoughtSignature file persistence + missing-signature recovery |
| tool_result images silently dropped | [#23712](https://github.com/BerriAI/litellm/issues/23712) | Image separation + functionResponse.parts nesting |
| No streaming tool arguments | [#22206](https://github.com/BerriAI/litellm/issues/22206) | partialArgs state machine + incremental JSON delta |

## Quick start

```bash
pip install gemini-claude-bridge
```

### As a proxy server

```bash
pip install "gemini-claude-bridge[proxy]"

GEMINI_API_KEY=your-key python -m examples.proxy_server
# or
GEMINI_API_KEY=your-key GEMINI_MODEL=gemini-2.5-pro python examples/proxy_server.py
```

Then point Claude Code at it:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

### As a library

```python
from gemini_claude_bridge import (
    convert_claude_to_gemini,
    convert_gemini_to_claude,
    GeminiStreamConverter,
    GeminiSignatureManager,
    ClaudeMessagesRequest,
)

# Parse incoming Claude request
claude_request = ClaudeMessagesRequest(**request_body)

# Convert to Gemini format
gemini_request = convert_claude_to_gemini(
    request=claude_request,
    model="gemini-2.5-flash-preview-04-17",
    signature_manager=GeminiSignatureManager(),
    session_id="session-123",
    project_path="/my/project",
    stream=True,
)

# Send to Gemini API, then convert streaming response back:
converter = GeminiStreamConverter(
    request_id="req-1",
    model="gemini-2.5-flash-preview-04-17",
    original_request_model=claude_request.model,
    signature_manager=GeminiSignatureManager(),
    session_id="session-123",
    project_path="/my/project",
)

async for sse_event in converter.convert_stream(gemini_ndjson_stream):
    yield sse_event  # Claude SSE format
```

## Features

### Core protocol conversion
- **Request**: Claude Messages → Gemini contents/parts (messages, system, tools, thinking)
- **Response**: Gemini → Claude (non-streaming, single JSON)
- **Streaming**: Gemini NDJSON → Claude SSE (text deltas, tool_use blocks)

### thoughtSignature handling
Gemini's thinking mode requires `thoughtSignature` to be echoed back on multi-turn tool calls. This library:
- Persists signatures to disk (survives process restarts)
- Recovers from aborted requests (dummy signature for completed tool calls, skip for incomplete)
- Configurable storage directory

### Streaming tool arguments (partialArgs)
Gemini streams tool arguments via `partialArgs` with `jsonPath` addressing. This library:
- Accumulates nested JSON structures from jsonPath fragments
- Supports incremental delta mode for large-content tools (Write, Edit)
- Handles `willContinue` state transitions correctly

### Multimodal tool results
When tool results contain images (e.g. Read tool reading a PNG), this library:
- Extracts images from tool_result content
- Nests them inside `functionResponse.parts` (not as sibling parts)
- Converts between Claude base64 and Gemini inlineData formats

### JSON Schema cleaning
Gemini only supports a subset of JSON Schema. This library strips unsupported fields
(`additionalProperties`, `format`, `oneOf`, etc.) recursively.

## Configuration

### Signature storage

```python
# Default: ~/.gemini-claude-bridge/signatures/
manager = GeminiSignatureManager()

# Custom directory:
manager = GeminiSignatureManager(base_dir="/path/to/signatures")
```

### Incremental streaming tools

By default, `Write`, `Edit`, `MultiEdit`, `NotebookEdit` use incremental JSON delta mode.
Customize:

```python
converter = GeminiStreamConverter(
    ...,
    incremental_tools={"write", "edit", "my_custom_tool"},
)
```

### System instruction prefix

Inject extra instructions into the system prompt:

```python
convert_claude_to_gemini(
    ...,
    system_instruction_prefix="Always respond in English.",
)
```

## Development

```bash
git clone https://github.com/weijiafu14/gemini-claude-bridge
cd gemini-claude-bridge
pip install -e ".[dev]"
pytest
```

## License

MIT
