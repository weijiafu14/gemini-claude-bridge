# gemini-claude-bridge

Production-grade **Gemini ↔ Anthropic Claude** protocol converter.

Two ways to use it:

1. **Claude Code Proxy** — drop-in replacement, `ANTHROPIC_BASE_URL=http://localhost:8082 claude`
2. **Python Library** — `pip install gemini-claude-bridge`, convert requests/responses in your own app

## Why this exists

LiteLLM has 3 open bugs in Gemini↔Claude conversion that break real-world tool calling:

| Problem | LiteLLM Issue | This library |
|---------|--------------|-------------|
| Multi-turn tool calling + thinking crashes | [#17949](https://github.com/BerriAI/litellm/issues/17949) (3+ months) | thoughtSignature file persistence + missing-signature recovery |
| tool_result images silently dropped | [#23712](https://github.com/BerriAI/litellm/issues/23712) | Image separation + functionResponse.parts nesting |
| No streaming tool arguments | [#22206](https://github.com/BerriAI/litellm/issues/22206) | partialArgs state machine + incremental JSON delta |

---

## Usage 1: Claude Code Proxy

### Quick start

```bash
pip install "gemini-claude-bridge[proxy]"

# Start proxy (default port 8082)
GEMINI_API_KEY=your-key python -m examples.proxy_server

# In another terminal — use Claude Code with Gemini backend
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=dummy claude
```

### Docker

```bash
# .env file
GEMINI_API_KEY=your-key
BIG_MODEL=gemini-2.5-pro
SMALL_MODEL=gemini-2.5-flash

docker compose up -d

ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

### Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google AI Studio API key |
| `GEMINI_API_BASE_URL` | `https://generativelanguage.googleapis.com` | Gemini API base URL (for custom endpoints / Vertex) |
| `BIG_MODEL` | `gemini-2.5-flash` | Model for opus/sonnet requests |
| `SMALL_MODEL` | `gemini-2.5-flash` | Model for haiku requests |
| `ANTHROPIC_API_KEY` | (optional) | If set, proxy validates incoming requests against this key |
| `PORT` | `8082` | Server port |

### Model mapping

Claude Code sends model names like `claude-sonnet-4-20250514`. The proxy maps them:

| Claude model contains | Maps to |
|----------------------|---------|
| `haiku` | `SMALL_MODEL` |
| `sonnet`, `opus`, anything else | `BIG_MODEL` |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API (streaming + non-streaming) |
| `/v1/count_tokens` | POST | Token count estimate |
| `/health` | GET | Health check |

---

## Usage 2: Python Library

```bash
pip install gemini-claude-bridge
```

### Convert requests

```python
from gemini_claude_bridge import (
    ClaudeMessagesRequest,
    convert_claude_to_gemini,
    convert_gemini_to_claude,
    GeminiStreamConverter,
    GeminiSignatureManager,
)

# Parse Claude request
claude_req = ClaudeMessagesRequest(**request_body)

# Convert to Gemini format
gemini_req = convert_claude_to_gemini(
    request=claude_req,
    model="gemini-2.5-flash",
    signature_manager=GeminiSignatureManager(),  # optional
    session_id="session-123",                    # optional
    project_path="/my/project",                  # optional
    stream=True,
)
# → Send gemini_req to Gemini API
```

### Convert streaming responses

```python
converter = GeminiStreamConverter(
    request_id="req-1",
    model="gemini-2.5-flash",
    original_request_model=claude_req.model,
    signature_manager=GeminiSignatureManager(),
)

async for sse_event in converter.convert_stream(gemini_ndjson_stream):
    yield sse_event  # Claude SSE format, ready for Anthropic SDK clients
```

### Convert non-streaming responses

```python
claude_response = convert_gemini_to_claude(
    gemini_response=gemini_json,
    original_request=claude_req,
)
# → Return claude_response as JSON to client
```

---

## Features

| Feature | Details |
|---------|---------|
| **thoughtSignature persistence** | Survives restarts, recovers from aborted requests |
| **Streaming tool args (partialArgs)** | jsonPath accumulation, incremental JSON delta for large tools |
| **Multimodal tool results** | Images nested in functionResponse.parts, not as siblings |
| **JSON Schema cleaning** | Strips `additionalProperties`, `format`, `oneOf` etc. |
| **Thinking mode** | Maps `budget_tokens` → Gemini `thinkingLevel` (LOW/HIGH) |
| **System instruction** | Configurable prefix via `system_instruction_prefix` param |

## Development

```bash
git clone https://github.com/weijiafu14/gemini-claude-bridge
cd gemini-claude-bridge
pip install -e ".[dev,proxy]"
pytest  # 45 tests
```

## License

MIT

Sources:
- [1rgs/claude-code-proxy](https://github.com/1rgs/claude-code-proxy)
- [fuergaosi233/claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy)
- [nielspeter/claude-code-proxy](https://github.com/nielspeter/claude-code-proxy)
