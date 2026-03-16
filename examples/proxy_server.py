"""
Minimal Gemini-Claude bridge proxy server.

Receives Anthropic Claude Messages API requests and proxies them to Google Gemini,
converting protocols in both directions.

Usage:
    GEMINI_API_KEY=your-key python examples/proxy_server.py

    # Then point Claude Code at the proxy:
    ANTHROPIC_BASE_URL=http://localhost:8082 claude

Environment variables:
    GEMINI_API_KEY      - Google Gemini API key (required)
    GEMINI_MODEL        - Model name (default: gemini-2.5-flash-preview-04-17)
    GEMINI_API_BASE_URL - API base URL (default: https://generativelanguage.googleapis.com)
    PORT                - Server port (default: 8082)
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# Add parent directory to path so we can import the library
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

app = FastAPI(title="Gemini-Claude Bridge Proxy")

# Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
GEMINI_API_BASE_URL = os.environ.get("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com")

# Global signature manager (persists thoughtSignatures across requests)
signature_manager = GeminiSignatureManager()


def _get_gemini_url(model: str, stream: bool) -> str:
    """Build Gemini API URL."""
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{GEMINI_API_BASE_URL}/v1beta/models/{model}:{action}"
    if stream:
        url += "?alt=sse"
    return url


@app.post("/v1/messages")
async def messages_endpoint(request: Request):
    """Claude Messages API compatible endpoint."""
    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY not set"},
        )

    body = await request.json()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        claude_request = ClaudeMessagesRequest(**body)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    is_stream = claude_request.stream or False
    session_id = request.headers.get("x-session-id", "default")
    project_path = request.headers.get("x-project-path", "default")

    # Convert Claude request to Gemini format
    gemini_request = convert_claude_to_gemini(
        request=claude_request,
        model=GEMINI_MODEL,
        signature_manager=signature_manager,
        session_id=session_id,
        project_path=project_path,
        stream=is_stream,
    )

    logger.info(f"[{request_id}] {'Stream' if is_stream else 'Non-stream'} request: model={GEMINI_MODEL}")

    if is_stream:
        return await _handle_stream(request_id, gemini_request, claude_request, session_id, project_path)
    else:
        return await _handle_non_stream(request_id, gemini_request, claude_request, session_id, project_path)


async def _handle_non_stream(
    request_id: str,
    gemini_request: dict,
    claude_request: ClaudeMessagesRequest,
    session_id: str,
    project_path: str,
) -> JSONResponse:
    """Handle non-streaming request."""
    url = _get_gemini_url(GEMINI_MODEL, stream=False)
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    # Remove stream field from request body (Gemini doesn't use it)
    payload = {k: v for k, v in gemini_request.items() if k != "stream"}

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        logger.error(f"[{request_id}] Gemini error {resp.status_code}: {resp.text[:500]}")
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

    gemini_response = resp.json()
    claude_response = convert_gemini_to_claude(
        gemini_response=gemini_response,
        original_request=claude_request,
        signature_manager=signature_manager,
        session_id=session_id,
        project_path=project_path,
    )

    logger.info(f"[{request_id}] Response: {len(claude_response.get('content', []))} blocks")
    return JSONResponse(content=claude_response)


async def _handle_stream(
    request_id: str,
    gemini_request: dict,
    claude_request: ClaudeMessagesRequest,
    session_id: str,
    project_path: str,
) -> StreamingResponse:
    """Handle streaming request."""
    url = _get_gemini_url(GEMINI_MODEL, stream=True)
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    payload = {k: v for k, v in gemini_request.items() if k != "stream"}

    async def sse_generator():
        converter = GeminiStreamConverter(
            request_id=request_id,
            model=GEMINI_MODEL,
            original_request_model=claude_request.model,
            signature_manager=signature_manager,
            session_id=session_id,
            project_path=project_path,
        )

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(f"[{request_id}] Gemini stream error {resp.status_code}: {error_body[:500]}")
                    error_event = json.dumps({
                        "type": "error",
                        "error": {"type": "api_error", "message": error_body.decode()[:500]},
                    })
                    yield f"event: error\ndata: {error_event}\n\n"
                    return

                async def ndjson_lines():
                    """Parse SSE or NDJSON lines from Gemini response."""
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            # Handle SSE format (data: {...})
                            if line.startswith("data: "):
                                line = line[6:]
                            if line.startswith("{"):
                                yield line

                async for event in converter.convert_stream(ndjson_lines()):
                    yield event

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-Id": request_id,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8082"))
    logger.info(f"Starting proxy: port={port}, model={GEMINI_MODEL}")
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable is required!")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=port)
