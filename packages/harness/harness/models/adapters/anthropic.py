"""Anthropic Messages API streaming adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from harness.types import LLMEvent, LLMRequest

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


@dataclass
class _ToolUseBuilder:
    id: str
    name: str
    partial_json: list[str] = field(default_factory=list)

    def update(self, delta: dict[str, Any]) -> None:
        if delta.get("type") == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                self.partial_json.append(partial)

    def event(self) -> LLMEvent:
        return LLMEvent(
            type="tool_call",
            data={
                "id": self.id,
                "type": "tool_use",
                "name": self.name,
                "arguments": "".join(self.partial_json),
            },
        )


class AnthropicAdapter:
    """Stream normalized events from Anthropic's Messages API."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._anthropic_version = anthropic_version
        self._timeout_s = timeout_s

    async def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        body = _request_body(req)
        state = _StreamState()
        async with httpx.AsyncClient(http2=True, timeout=self._timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self._anthropic_version,
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    raw = _parse_sse_line(line)
                    if raw is None:
                        continue
                    for event in state.update(json.loads(raw)):
                        yield event

        for event in state.finish():
            yield event


@dataclass
class _StreamState:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_builders: dict[int, _ToolUseBuilder] = field(default_factory=dict)
    emitted_tools: set[int] = field(default_factory=set)
    stopped: bool = False

    def update(self, payload: dict[str, Any]) -> list[LLMEvent]:
        event_type = payload.get("type")
        if event_type == "message_start":
            self._record_message_start(payload)
            return []
        if event_type == "content_block_start":
            self._start_content_block(payload)
            return []
        if event_type == "content_block_delta":
            return self._handle_delta(payload)
        if event_type == "content_block_stop":
            return self._stop_content_block(payload)
        if event_type == "message_delta":
            self._record_message_delta(payload)
            return []
        if event_type == "message_stop":
            self.stopped = True
            return []
        if event_type == "error":
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message", "Anthropic stream error")
                raise RuntimeError(str(message))
            raise RuntimeError("Anthropic stream error")
        return []

    def finish(self) -> list[LLMEvent]:
        events = [
            builder.event()
            for index, builder in sorted(self.tool_builders.items())
            if index not in self.emitted_tools
        ]
        if self.input_tokens or self.output_tokens:
            events.append(
                LLMEvent(
                    type="usage",
                    data={
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens,
                        "total_tokens": self.input_tokens + self.output_tokens,
                    },
                )
            )
        events.append(LLMEvent(type="done"))
        return events

    def _record_message_start(self, payload: dict[str, Any]) -> None:
        message = payload.get("message")
        if not isinstance(message, dict):
            return
        usage = message.get("usage")
        if isinstance(usage, dict):
            self.input_tokens = _int_value(usage, "input_tokens")
            self.output_tokens = _int_value(usage, "output_tokens")

    def _record_message_delta(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self.output_tokens = _int_value(usage, "output_tokens")

    def _start_content_block(self, payload: dict[str, Any]) -> None:
        index = _index(payload)
        block = payload.get("content_block")
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return
        self.tool_builders[index] = _ToolUseBuilder(
            id=str(block.get("id", "")),
            name=str(block.get("name", "")),
        )

    def _handle_delta(self, payload: dict[str, Any]) -> list[LLMEvent]:
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return []
        if delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                return [LLMEvent(type="token", data={"text": text})]
        builder = self.tool_builders.get(_index(payload))
        if builder is not None:
            builder.update(delta)
        return []

    def _stop_content_block(self, payload: dict[str, Any]) -> list[LLMEvent]:
        index = _index(payload)
        builder = self.tool_builders.get(index)
        if builder is None or index in self.emitted_tools:
            return []
        self.emitted_tools.add(index)
        return [builder.event()]


def _request_body(req: LLMRequest) -> dict[str, object]:
    body: dict[str, object] = {
        "model": req.binding.model,
        "max_tokens": _max_tokens(req),
        "messages": [
            message.model_dump(mode="json")
            for message in req.messages
            if message.role != "system"
        ],
        "stream": req.stream,
    }
    system = "\n\n".join(message.content for message in req.messages if message.role == "system")
    if system:
        body["system"] = system
    return body


def _max_tokens(req: LLMRequest) -> int:
    value = req.binding.params.get("max_tokens", DEFAULT_MAX_TOKENS)
    return value if isinstance(value, int) else DEFAULT_MAX_TOKENS


def _parse_sse_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped.removeprefix("data:").strip()


def _index(payload: dict[str, Any]) -> int:
    value = payload.get("index", 0)
    return value if isinstance(value, int) else 0


def _int_value(values: dict[str, Any], key: str) -> int:
    value = values.get(key, 0)
    return value if isinstance(value, int) else 0
