from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from harness.errors import ValidationFailed
from harness.types import LLMEvent, LLMRequest, Message, ModelBinding


class DummyAdapter:
    async def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        yield LLMEvent(type="token", data={"text": f"echo:{req.messages[0].content}"})
        yield LLMEvent(
            type="usage",
            data={"input_tokens": 10, "output_tokens": 5, "usd": 0.001},
        )
        yield LLMEvent(type="done")


def _binding(**params: object) -> ModelBinding:
    return ModelBinding(
        name="gemma-small",
        provider="dummy",
        model="gemma4-nano",
        params=params,
    )


def _request(binding: ModelBinding) -> LLMRequest:
    return LLMRequest(
        binding=binding,
        messages=[Message(role="user", content="hello")],
    )


def test_capabilities_come_from_static_registry_and_binding_overrides() -> None:
    from harness.models.gateway import ModelGateway

    gateway = ModelGateway(adapters={"dummy": DummyAdapter()})

    caps = gateway.capabilities(
        _binding(max_context=4096, guided_decoding=False, price_input_per_million=0.05)
    )

    assert caps.tool_calls is True
    assert caps.guided_decoding is False
    assert caps.max_context == 4096
    assert caps.prompt_cache is False
    assert caps.streaming is True
    assert caps.price_input_per_million == 0.05
    assert caps.price_output_per_million == 0.2


@pytest.mark.asyncio
async def test_gateway_dispatches_adapter_and_emits_metering_event() -> None:
    from harness.models.gateway import ModelGateway

    gateway = ModelGateway(adapters={"dummy": DummyAdapter()})

    events = [event async for event in gateway.complete(_request(_binding()))]

    assert [(event.type, event.data) for event in events] == [
        ("token", {"text": "echo:hello"}),
        ("usage", {"input_tokens": 10, "output_tokens": 5, "usd": 0.001}),
        (
            "usage",
            {
                "provider": "dummy",
                "model": "gemma4-nano",
                "input_tokens": 10,
                "output_tokens": 5,
                "usd": 0.001,
            },
        ),
        ("done", {}),
    ]


def test_unknown_provider_raises_validation_failed() -> None:
    from harness.models.gateway import ModelGateway

    gateway = ModelGateway(adapters={})

    with pytest.raises(ValidationFailed, match="unknown model provider"):
        gateway.capabilities(_binding())
