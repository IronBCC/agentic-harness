from __future__ import annotations

import json

import httpx
import pytest
from harness.types import LLMRequest, Message, ModelBinding
from respx import MockRouter


def _request(*, guided: bool = False, schema: dict[str, object] | None = None) -> LLMRequest:
    return LLMRequest(
        binding=ModelBinding(
            name="local-gemma",
            provider="openai_compat",
            model="gemma4-nano",
            params={"guided_decoding": guided},
        ),
        messages=[Message(role="user", content="say hi")],
        output_schema=schema,
    )


def _sse_response() -> str:
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q"'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ':"pizza"}'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
        },
    ]
    return "\n\n".join(
        [*(f"data: {json.dumps(chunk)}" for chunk in chunks), "data: [DONE]"]
    )


@pytest.mark.asyncio
async def test_openai_compat_streams_tokens_tool_calls_usage_and_done(
    respx_mock: MockRouter,
) -> None:
    from harness.models.adapters.openai_compat import OpenAICompatAdapter

    route = respx_mock.post("https://llm.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse_response(),
        )
    )
    adapter = OpenAICompatAdapter(base_url="https://llm.test/v1", api_key="test-key")

    events = [event async for event in adapter.complete(_request())]

    assert [(event.type, event.data) for event in events] == [
        ("token", {"text": "Hel"}),
        ("token", {"text": "lo"}),
        (
            "usage",
            {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13},
        ),
        (
            "tool_call",
            {
                "id": "call_1",
                "type": "function",
                "name": "search",
                "arguments": '{"q":"pizza"}',
            },
        ),
        ("done", {}),
    ]
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["accept"] == "text/event-stream"
    body = request.read().decode()
    assert '"model":"gemma4-nano"' in body
    assert '"stream":true' in body
    assert '"include_usage":true' in body


@pytest.mark.asyncio
async def test_openai_compat_passes_guided_json_only_when_capable(
    respx_mock: MockRouter,
) -> None:
    from harness.models.adapters.openai_compat import OpenAICompatAdapter

    route = respx_mock.post("https://llm.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="data: [DONE]\n\n")
    )
    adapter = OpenAICompatAdapter(base_url="https://llm.test/v1", api_key="test-key")
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}

    _ = [event async for event in adapter.complete(_request(guided=True, schema=schema))]
    guided_body = route.calls.last.request.read().decode()

    _ = [event async for event in adapter.complete(_request(guided=False, schema=schema))]
    unguided_body = route.calls.last.request.read().decode()

    assert '"guided_json"' in guided_body
    assert '"guided_json"' not in unguided_body
