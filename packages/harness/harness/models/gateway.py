"""Model gateway core and capability registry."""

from __future__ import annotations

import tomllib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

from harness.errors import ValidationFailed
from harness.types import LLMEvent, LLMRequest, ModelBinding


class ModelAdapter(Protocol):
    def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...


@dataclass(frozen=True)
class ModelCaps:
    tool_calls: bool
    guided_decoding: bool
    max_context: int
    prompt_cache: bool
    streaming: bool
    price_input_per_million: float = 0.0
    price_output_per_million: float = 0.0


class ModelGateway:
    """Dispatch model calls to provider adapters and normalize capability metadata."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ModelAdapter],
        caps: Mapping[str, ModelCaps] | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._caps = dict(caps or _load_static_caps())

    def capabilities(self, binding: ModelBinding) -> ModelCaps:
        """Return provider capabilities with binding-level param overrides."""
        if binding.provider not in self._adapters:
            raise ValidationFailed(f"unknown model provider: {binding.provider}")
        base = self._caps.get(binding.provider)
        if base is None:
            raise ValidationFailed(f"missing model capabilities: {binding.provider}")
        return _merge_overrides(base, binding.params)

    async def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Stream adapter events and emit normalized usage metering."""
        adapter = self._adapters.get(req.binding.provider)
        if adapter is None:
            raise ValidationFailed(f"unknown model provider: {req.binding.provider}")

        async for event in adapter.complete(req):
            yield event
            if event.type == "usage":
                yield LLMEvent(
                    type="usage",
                    data={
                        "provider": req.binding.provider,
                        "model": req.binding.model,
                        "input_tokens": _int_data(event.data, "input_tokens"),
                        "output_tokens": _int_data(event.data, "output_tokens"),
                        "usd": _float_data(event.data, "usd"),
                    },
                )


def _load_static_caps() -> dict[str, ModelCaps]:
    payload = files("harness.models").joinpath("caps.toml").read_bytes()
    raw = tomllib.loads(payload.decode())
    return {
        provider: ModelCaps(
            tool_calls=bool(values["tool_calls"]),
            guided_decoding=bool(values["guided_decoding"]),
            max_context=int(values["max_context"]),
            prompt_cache=bool(values["prompt_cache"]),
            streaming=bool(values["streaming"]),
            price_input_per_million=float(values.get("price_input_per_million", 0.0)),
            price_output_per_million=float(values.get("price_output_per_million", 0.0)),
        )
        for provider, values in raw.items()
    }


def _merge_overrides(base: ModelCaps, params: dict[str, object]) -> ModelCaps:
    return ModelCaps(
        tool_calls=_bool_param(params, "tool_calls", base.tool_calls),
        guided_decoding=_bool_param(params, "guided_decoding", base.guided_decoding),
        max_context=_int_param(params, "max_context", base.max_context),
        prompt_cache=_bool_param(params, "prompt_cache", base.prompt_cache),
        streaming=_bool_param(params, "streaming", base.streaming),
        price_input_per_million=_float_param(
            params,
            "price_input_per_million",
            base.price_input_per_million,
        ),
        price_output_per_million=_float_param(
            params,
            "price_output_per_million",
            base.price_output_per_million,
        ),
    )


def _bool_param(params: dict[str, object], key: str, default: bool) -> bool:
    value = params.get(key, default)
    return value if isinstance(value, bool) else default


def _int_param(params: dict[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    return value if isinstance(value, int) else default


def _float_param(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    return default


def _int_data(data: dict[str, object], key: str) -> int:
    value = data.get(key, 0)
    return value if isinstance(value, int) else 0


def _float_data(data: dict[str, object], key: str) -> float:
    value = data.get(key, 0.0)
    if isinstance(value, int | float):
        return float(value)
    return 0.0
