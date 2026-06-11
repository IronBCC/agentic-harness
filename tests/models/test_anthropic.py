from __future__ import annotations

import json

import httpx
import pytest
from harness.types import LLMRequest, Message, ModelBinding
from respx import MockRouter


def _request() -> LLMRequest:
    return LLMRequest(
        binding=ModelBinding(
            name="claude",
            provider="anthropic",
            model="claude-opus-4-8",
        ),
        messages=[
            Message(role="system", content="Return concise output."),
            Message(role="user", content="say hi"),
        ],
    )


def _sse_response() -> str:
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-4-8",
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hel"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "lo"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q"'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ':"pizza"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "future_event", "ignored": True},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 15},
        },
        {"type": "message_stop"},
    ]
    return "\n\n".join(f"data: {json.dumps(chunk)}" for chunk in chunks)


@pytest.mark.asyncio
async def test_anthropic_streams_tokens_tool_calls_usage_and_done(
    respx_mock: MockRouter,
) -> None:
    from harness.models.adapters.anthropic import AnthropicAdapter

    route = respx_mock.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_response(),
        )
    )
    adapter = AnthropicAdapter(
        base_url="https://api.anthropic.test/v1",
        api_key="test-key",
    )

    events = [event async for event in adapter.complete(_request())]

    assert [(event.type, event.data) for event in events] == [
        ("token", {"text": "Hel"}),
        ("token", {"text": "lo"}),
        (
            "tool_call",
            {
                "id": "toolu_1",
                "type": "tool_use",
                "name": "search",
                "arguments": '{"q":"pizza"}',
            },
        ),
        ("usage", {"input_tokens": 12, "output_tokens": 15, "total_tokens": 27}),
        ("done", {}),
    ]
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["accept"] == "text/event-stream"
    body = request.read().decode()
    assert '"model":"claude-opus-4-8"' in body
    assert '"max_tokens":1024' in body
    assert '"stream":true' in body
    assert '"system":"Return concise output."' in body
    assert '"messages":[{"role":"user","content":"say hi"}]' in body


@pytest.mark.asyncio
async def test_anthropic_accepts_custom_version_and_max_tokens(
    respx_mock: MockRouter,
) -> None:
    from harness.models.adapters.anthropic import AnthropicAdapter

    route = respx_mock.post("https://api.anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"type":"message_stop"}\n\n',
        )
    )
    adapter = AnthropicAdapter(
        base_url="https://api.anthropic.test/v1",
        api_key="test-key",
        anthropic_version="2025-01-01",
    )
    req = _request().model_copy(
        update={
            "binding": _request().binding.model_copy(
                update={"params": {"max_tokens": 2048}},
            )
        }
    )

    _ = [event async for event in adapter.complete(req)]

    request = route.calls.last.request
    assert request.headers["anthropic-version"] == "2025-01-01"
    assert '"max_tokens":2048' in request.read().decode()
