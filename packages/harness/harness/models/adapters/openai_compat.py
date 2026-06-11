"""OpenAI-compatible streaming chat adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from harness.types import LLMEvent, LLMRequest


@dataclass
class _ToolCallBuilder:
    id: str | None = None
    type: str = "function"
    name: str | None = None
    arguments: list[str] = field(default_factory=list)

    def update(self, delta: dict[str, Any]) -> None:
        if "id" in delta:
            self.id = str(delta["id"])
        if "type" in delta:
            self.type = str(delta["type"])
        function = delta.get("function")
        if isinstance(function, dict):
            if "name" in function:
                self.name = str(function["name"])
            if "arguments" in function:
                self.arguments.append(str(function["arguments"]))

    def event(self) -> LLMEvent:
        return LLMEvent(
            type="tool_call",
            data={
                "id": self.id or "",
                "type": self.type,
                "name": self.name or "",
                "arguments": "".join(self.arguments),
            },
        )


class OpenAICompatAdapter:
    """Stream Chat Completions from OpenAI-compatible servers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s

    async def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        body = self._request_body(req)
        tool_calls: dict[int, _ToolCallBuilder] = {}
        async with httpx.AsyncClient(http2=True, timeout=self._timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "authorization": f"Bearer {self._api_key}",
                    "accept": "text/event-stream",
                },
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is None:
                        continue
                    if event == "[DONE]":
                        break
                    parsed = json.loads(event)
                    for output in _events_from_chunk(parsed, tool_calls):
                        yield output

        for call in tool_calls.values():
            yield call.event()
        yield LLMEvent(type="done")

    def _request_body(self, req: LLMRequest) -> dict[str, object]:
        body: dict[str, object] = {
            "model": req.binding.model,
            "messages": [message.model_dump(mode="json") for message in req.messages],
            "stream": req.stream,
            "stream_options": {"include_usage": True},
        }
        if req.output_schema is not None and _guided_decoding_enabled(req):
            body["guided_json"] = req.output_schema
        return body


def _events_from_chunk(
    parsed: dict[str, Any],
    tool_calls: dict[int, _ToolCallBuilder],
) -> list[LLMEvent]:
    events: list[LLMEvent] = []
    choices = parsed.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                events.append(LLMEvent(type="token", data={"text": content}))
            _assemble_tool_calls(delta, tool_calls)

    usage = parsed.get("usage")
    if isinstance(usage, dict):
        events.append(
            LLMEvent(
                type="usage",
                data={
                    "input_tokens": _int_value(usage, "prompt_tokens"),
                    "output_tokens": _int_value(usage, "completion_tokens"),
                    "total_tokens": _int_value(usage, "total_tokens"),
                },
            )
        )
    return events


def _assemble_tool_calls(
    delta: dict[str, Any],
    tool_calls: dict[int, _ToolCallBuilder],
) -> None:
    raw_calls = delta.get("tool_calls")
    if not isinstance(raw_calls, list):
        return
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        index = raw_call.get("index", 0)
        if not isinstance(index, int):
            index = 0
        builder = tool_calls.setdefault(index, _ToolCallBuilder())
        builder.update(raw_call)


def _parse_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped.removeprefix("data:").strip()


def _guided_decoding_enabled(req: LLMRequest) -> bool:
    value = req.binding.params.get("guided_decoding", False)
    return value if isinstance(value, bool) else False


def _int_value(values: dict[str, Any], key: str) -> int:
    value = values.get(key, 0)
    return value if isinstance(value, int) else 0
