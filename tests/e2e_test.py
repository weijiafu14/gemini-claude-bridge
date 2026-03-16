"""
End-to-end tests against a running proxy server.

Simulates what Claude Code CLI sends and validates the full request→Gemini→response pipeline.

Usage:
    # Start proxy first:
    GEMINI_API_KEY=your-key BIG_MODEL=gemini-3-flash-preview PORT=8089 python examples/proxy_server.py

    # Run tests:
    PROXY_URL=http://localhost:8089 python tests/e2e_test.py
"""

import os
import sys
import json
import time
import httpx
import base64

PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8089")
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds
PASS = 0
FAIL = 0


def post(body: dict, stream: bool = False):
    """Send request to proxy with rate-limit retry."""
    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{PROXY_URL}/v1/messages",
                json=body,
                headers={"Content-Type": "application/json"},
            )
            data = resp.json()
            # Retry on 429/502
            if resp.status_code in (429, 502) or "429" in str(data.get("error", "")):
                if attempt < MAX_RETRIES - 1:
                    print(f"    ⏳ Rate limited, retrying in {RETRY_DELAY}s... ({attempt+1}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                    continue
            return data
    return data


def post_stream(body: dict) -> list:
    """Send streaming request with rate-limit retry, collect all SSE events."""
    for attempt in range(MAX_RETRIES):
        events = []
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST",
                f"{PROXY_URL}/v1/messages",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                buffer = ""
                for chunk in resp.iter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        events.append(event_str)
        # Check for error event with 429
        has_429 = any("429" in e for e in events)
        if has_429 and attempt < MAX_RETRIES - 1:
            print(f"    ⏳ Rate limited (stream), retrying in {RETRY_DELAY}s... ({attempt+1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            continue
        return events
    return events


def extract_text(response: dict) -> str:
    """Extract text from non-streaming response."""
    for block in response.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def extract_tool_calls(response: dict) -> list:
    """Extract tool_use blocks from response."""
    return [b for b in response.get("content", []) if b.get("type") == "tool_use"]


def extract_stream_text(events: list) -> str:
    """Extract accumulated text from SSE events."""
    text = ""
    for event in events:
        for line in event.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("delta", {}).get("type") == "text_delta":
                        text += data["delta"]["text"]
                except json.JSONDecodeError:
                    pass
    return text


def extract_stream_tool_input(events: list) -> dict:
    """Extract tool_use input from SSE events."""
    accumulated = ""
    for event in events:
        for line in event.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("delta", {}).get("type") == "input_json_delta":
                        accumulated += data["delta"]["partial_json"]
                except json.JSONDecodeError:
                    pass
    if accumulated:
        return json.loads(accumulated)
    return {}


def extract_stream_stop_reason(events: list) -> str:
    """Extract stop_reason from SSE message_delta event."""
    for event in events:
        for line in event.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "message_delta":
                        return data.get("delta", {}).get("stop_reason", "")
                except json.JSONDecodeError:
                    pass
    return ""


def extract_stream_tool_name(events: list) -> str:
    """Extract tool name from SSE content_block_start."""
    for event in events:
        for line in event.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "content_block_start":
                        cb = data.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            return cb.get("name", "")
                except json.JSONDecodeError:
                    pass
    return ""


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


# ============================================================
# Test 1: Basic non-streaming text
# ============================================================
def test_1_basic_text():
    print("\n🧪 Test 1: Basic non-streaming text")
    resp = post({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
    })

    check("has id", "id" in resp, str(resp))
    check("type is message", resp.get("type") == "message", resp.get("type"))
    check("role is assistant", resp.get("role") == "assistant")
    check("model preserved", resp.get("model") == "claude-sonnet-4-20250514")
    check("has content", len(resp.get("content", [])) > 0)
    check("stop_reason is end_turn", resp.get("stop_reason") == "end_turn")

    text = extract_text(resp)
    check("answer contains Paris", "paris" in text.lower(), text)
    check("has usage", "usage" in resp)


# ============================================================
# Test 2: Streaming text
# ============================================================
def test_2_streaming_text():
    print("\n🧪 Test 2: Streaming text")
    events = post_stream({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "user", "content": "Count from 1 to 5, each number on a new line. Nothing else."}],
    })

    check("got events", len(events) > 0, f"got {len(events)}")

    # Check SSE event sequence
    event_types = []
    for event in events:
        for line in event.split("\n"):
            if line.startswith("event: "):
                event_types.append(line[7:])

    check("starts with message_start", event_types[0] == "message_start" if event_types else False)
    check("has content_block_start", "content_block_start" in event_types)
    check("has content_block_delta", "content_block_delta" in event_types)
    check("has content_block_stop", "content_block_stop" in event_types)
    check("has message_delta", "message_delta" in event_types)
    check("ends with message_stop", event_types[-1] == "message_stop" if event_types else False)

    text = extract_stream_text(events)
    check("text has numbers", "1" in text and "5" in text, text[:100])

    stop = extract_stream_stop_reason(events)
    check("stop_reason is end_turn", stop == "end_turn", stop)


# ============================================================
# Test 3: Single tool call (streaming)
# ============================================================
def test_3_single_tool_call():
    print("\n🧪 Test 3: Streaming tool call")
    events = post_stream({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "stream": True,
        "tools": [{
            "name": "get_weather",
            "description": "Get weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        }],
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    })

    tool_name = extract_stream_tool_name(events)
    check("tool_name is get_weather", tool_name == "get_weather", tool_name)

    tool_input = extract_stream_tool_input(events)
    check("tool_input has city", "city" in tool_input, str(tool_input))
    check("city contains Tokyo", "tokyo" in tool_input.get("city", "").lower(), str(tool_input))

    stop = extract_stream_stop_reason(events)
    check("stop_reason is tool_use", stop == "tool_use", stop)


# ============================================================
# Test 4: Multi-turn tool calling (2 rounds)
# ============================================================
def test_4_multi_turn_tool_call():
    print("\n🧪 Test 4: Multi-turn tool calling")

    tools = [{
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    }]

    # Round 1: User asks → model calls Read
    r1 = post({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "tools": tools,
        "messages": [
            {"role": "user", "content": "Read the file /tmp/test.py and tell me what it does."},
        ],
    })

    tc1 = extract_tool_calls(r1)
    check("round 1 has tool_use", len(tc1) > 0, str(r1.get("content", [])))
    if not tc1:
        return

    check("round 1 tool is Read", tc1[0]["name"] == "Read", tc1[0].get("name"))
    tool_use_id_1 = tc1[0]["id"]

    # Round 2: Provide tool_result → model calls Read again or responds
    r2 = post({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "tools": tools,
        "messages": [
            {"role": "user", "content": "Read /tmp/test.py and /tmp/utils.py, then summarize both."},
            {"role": "assistant", "content": [tc1[0]]},
            {"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id_1,
                    "content": "def hello():\n    print('Hello world')\n\nhello()",
                },
            ]},
        ],
    })

    check("round 2 succeeded", r2.get("type") == "message", str(r2)[:200])
    # Model should either call Read again for utils.py, or respond with summary
    has_text = any(b.get("type") == "text" for b in r2.get("content", []))
    has_tool = any(b.get("type") == "tool_use" for b in r2.get("content", []))
    check("round 2 has text or tool_use", has_text or has_tool, str(r2.get("content", []))[:200])


# ============================================================
# Test 5: Image in tool_result
# ============================================================
def test_5_image_in_tool_result():
    print("\n🧪 Test 5: Image in tool_result")

    # Create a tiny 1x1 red PNG
    import struct, zlib
    def make_tiny_png():
        # Minimal valid PNG: 1x1 pixel, red
        sig = b'\x89PNG\r\n\x1a\n'
        # IHDR
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        # IDAT
        raw = zlib.compress(b'\x00\xff\x00\x00')  # filter byte + RGB
        idat_crc = zlib.crc32(b'IDAT' + raw) & 0xffffffff
        idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
        # IEND
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend

    png_b64 = base64.b64encode(make_tiny_png()).decode()

    tools = [{
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    }]

    resp = post({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "tools": tools,
        "messages": [
            {"role": "user", "content": "What does the image at /tmp/icon.png show?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_img_1", "name": "Read", "input": {"file_path": "/tmp/icon.png"}},
            ]},
            {"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_img_1",
                    "content": [
                        {"type": "text", "text": "Image file: icon.png (1x1 pixel)"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png_b64,
                            },
                        },
                    ],
                },
            ]},
        ],
    })

    check("response succeeded", resp.get("type") == "message", str(resp)[:300])
    text = extract_text(resp)
    # Model should acknowledge it saw an image, not say "I can't see images"
    check("model processed image (no error)", resp.get("stop_reason") in ("end_turn", "tool_use"), resp.get("stop_reason"))
    check("has text response", len(text) > 0, "empty response")


# ============================================================
# Test 6: Thinking mode (non-streaming)
# ============================================================
def test_6_thinking_mode():
    print("\n🧪 Test 6: Thinking mode")
    resp = post({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "messages": [{"role": "user", "content": "What is 17 * 23? Show your work."}],
    })

    check("response succeeded", resp.get("type") == "message", str(resp)[:300])
    text = extract_text(resp)
    check("answer contains 391", "391" in text, text[:200])


# ============================================================
# Test 7: Streaming tool call with thinking
# ============================================================
def test_7_streaming_tool_with_thinking():
    print("\n🧪 Test 7: Streaming tool call + thinking")
    events = post_stream({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "tools": [{
            "name": "Bash",
            "description": "Run a bash command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }],
        "messages": [{"role": "user", "content": "List all Python files in /tmp using the Bash tool."}],
    })

    tool_name = extract_stream_tool_name(events)
    check("tool called", tool_name == "Bash", tool_name)

    tool_input = extract_stream_tool_input(events)
    check("command has content", len(tool_input.get("command", "")) > 0, str(tool_input))

    stop = extract_stream_stop_reason(events)
    check("stop_reason is tool_use", stop == "tool_use", stop)


# ============================================================
# Test 8: Multiple tools in one response
# ============================================================
def test_8_multiple_tools():
    print("\n🧪 Test 8: Non-streaming tool call")
    resp = post({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ],
        "messages": [{"role": "user", "content": "What's the weather in London?"}],
    })

    tc = extract_tool_calls(resp)
    check("has tool_use", len(tc) > 0, str(resp.get("content", [])))
    if tc:
        check("tool name correct", tc[0]["name"] == "get_weather")
        check("tool has id", len(tc[0].get("id", "")) > 0)
        check("input has city", "city" in tc[0].get("input", {}))
    check("stop_reason is tool_use", resp.get("stop_reason") == "tool_use")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print(f"🚀 E2E Testing against {PROXY_URL}")
    print("=" * 60)

    tests = [
        test_1_basic_text,
        test_2_streaming_text,
        test_3_single_tool_call,
        test_4_multi_turn_tool_call,
        test_5_image_in_tool_result,
        test_6_thinking_mode,
        test_7_streaming_tool_with_thinking,
        test_8_multiple_tools,
    ]

    for i, test_fn in enumerate(tests):
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            print(f"  💥 CRASH: {e}")
        # Brief pause between tests to avoid rate limiting
        if i < len(tests) - 1:
            time.sleep(3)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    sys.exit(1 if FAIL > 0 else 0)
