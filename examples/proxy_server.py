"""
Claude Code compatible proxy server — use Gemini as Claude backend.

Usage:
    # Minimal
    GEMINI_API_KEY=your-key python examples/proxy_server.py

    # Then in another terminal:
    ANTHROPIC_BASE_URL=http://localhost:8082 claude

    # With model mapping
    GEMINI_API_KEY=your-key BIG_MODEL=gemini-2.5-pro SMALL_MODEL=gemini-2.5-flash python examples/proxy_server.py

Environment variables:
    GEMINI_API_KEY        - Google Gemini API key (required)
    GEMINI_API_BASE_URL   - API base (default: https://generativelanguage.googleapis.com)
    BIG_MODEL             - Model for opus/sonnet requests (default: gemini-2.5-flash)
    SMALL_MODEL           - Model for haiku requests (default: gemini-2.5-flash)
    ANTHROPIC_API_KEY     - If set, proxy validates incoming requests against this key
    PORT                  - Server port (default: 8082)
"""

import os
import sys
import json
import uuid
import logging
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gemini_claude_bridge import (
    ClaudeMessagesRequest,
    convert_claude_to_gemini,
    convert_gemini_to_claude,
    GeminiStreamConverter,
    GeminiSignatureManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("proxy")

app = FastAPI(title="gemini-claude-bridge proxy")

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_BASE_URL = os.environ.get(
    "GEMINI_API_BASE_URL",
    "https://generativelanguage.googleapis.com",
)
BIG_MODEL = os.environ.get("BIG_MODEL", "gemini-2.5-flash")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-2.5-flash")

# Optional: if set, clients must send this exact key as x-api-key / Authorization
PROXY_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Signature persistence
signature_manager = GeminiSignatureManager()


# --- Model mapping ---

def _resolve_model(claude_model: str) -> str:
    """
    Map Claude model name to Gemini model name.

    Claude Code sends model names like:
      claude-sonnet-4-20250514, claude-opus-4-20250514, claude-haiku-3.5-20241022
    We map by family: opus/sonnet → BIG_MODEL, haiku → SMALL_MODEL.
    """
    m = claude_model.lower()
    if "haiku" in m:
        return SMALL_MODEL
    # opus, sonnet, or anything else → big model
    return BIG_MODEL


# --- Auth ---

def _check_auth(request: Request) -> Optional[JSONResponse]:
    """Validate API key if ANTHROPIC_API_KEY is configured."""
    if not PROXY_API_KEY:
        return None  # no auth required

    # Claude Code sends: x-api-key header or Authorization: Bearer xxx
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            api_key = auth[7:]

    if api_key != PROXY_API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid API key"},
            },
        )
    return None


# --- Gemini HTTP ---

def _gemini_url(model: str, stream: bool) -> str:
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{GEMINI_API_BASE_URL}/v1beta/models/{model}:{action}"
    if stream:
        url += "?alt=sse"
    return url


# --- Endpoints ---

@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API compatible endpoint."""
    # Auth
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "api_error", "message": "GEMINI_API_KEY not configured"}},
        )

    body = await request.json()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        claude_request = ClaudeMessagesRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
        )

    gemini_model = _resolve_model(claude_request.model)
    is_stream = claude_request.stream or False
    session_id = request.headers.get("x-session-id", "default")
    project_path = request.headers.get("x-project-path", "default")

    gemini_request = convert_claude_to_gemini(
        request=claude_request,
        model=gemini_model,
        signature_manager=signature_manager,
        session_id=session_id,
        project_path=project_path,
        stream=is_stream,
    )

    logger.info(
        f"[{request_id}] {claude_request.model} → {gemini_model} "
        f"{'stream' if is_stream else 'sync'}"
    )

    if is_stream:
        return await _handle_stream(request_id, gemini_model, gemini_request, claude_request, session_id, project_path)
    else:
        return await _handle_sync(request_id, gemini_model, gemini_request, claude_request, session_id, project_path)


@app.post("/v1/count_tokens")
async def count_tokens(request: Request):
    """Token counting endpoint (stub — returns estimate)."""
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err
    body = await request.json()
    # Rough estimate: 4 chars per token
    text = json.dumps(body.get("messages", []))
    return JSONResponse(content={"input_tokens": max(1, len(text) // 4)})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
    }


# --- Handlers ---

async def _handle_sync(
    request_id: str,
    gemini_model: str,
    gemini_request: dict,
    claude_request: ClaudeMessagesRequest,
    session_id: str,
    project_path: str,
) -> JSONResponse:
    url = _gemini_url(gemini_model, stream=False)
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {k: v for k, v in gemini_request.items() if k != "stream"}

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        logger.error(f"[{request_id}] Gemini {resp.status_code}: {resp.text[:1000]}")
        logger.error(f"[{request_id}] Sent payload keys: {list(payload.keys())}, contents count: {len(payload.get('contents',[]))}")
        import json as _j
        for i, c in enumerate(payload.get('contents', [])):
            parts_summary = []
            for p in c.get('parts', []):
                pk = list(p.keys())
                parts_summary.append(pk)
            logger.error(f"[{request_id}]   contents[{i}] role={c.get('role')} parts={parts_summary}")
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"Gemini returned {resp.status_code}"}},
        )

    claude_response = convert_gemini_to_claude(
        gemini_response=resp.json(),
        original_request=claude_request,
        signature_manager=signature_manager,
        session_id=session_id,
        project_path=project_path,
    )

    logger.info(f"[{request_id}] → {len(claude_response.get('content', []))} blocks")
    return JSONResponse(content=claude_response)


async def _handle_stream(
    request_id: str,
    gemini_model: str,
    gemini_request: dict,
    claude_request: ClaudeMessagesRequest,
    session_id: str,
    project_path: str,
) -> StreamingResponse:
    url = _gemini_url(gemini_model, stream=True)
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {k: v for k, v in gemini_request.items() if k != "stream"}

    async def sse_generator():
        converter = GeminiStreamConverter(
            request_id=request_id,
            model=gemini_model,
            original_request_model=claude_request.model,
            signature_manager=signature_manager,
            session_id=session_id,
            project_path=project_path,
        )

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(f"[{request_id}] Gemini stream {resp.status_code}: {error_body[:500]}")
                    yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message':error_body.decode()[:500]}})}\n\n"
                    return

                async def ndjson_lines():
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                line = line[6:]
                            if line.startswith("{"):
                                yield line

                async for event in converter.convert_stream(ndjson_lines()):
                    yield event

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Request-Id": request_id},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8082"))
    if not GEMINI_API_KEY:
        logger.error("Set GEMINI_API_KEY environment variable")
        sys.exit(1)
    logger.info(f"Starting on :{port}  big={BIG_MODEL}  small={SMALL_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=port)
