from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from harness.errors import SchemaViolation
from harness.models.gateway import ModelCaps, ModelGateway
from harness.types import LLMEvent, LLMRequest, Message, ModelBinding


class TextAdapter:
    def __init__(self, text: str) -> None:
        self.seen: list[LLMRequest] = []
        self._text = text

    def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        return self._complete(req)

    async def _complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.seen.append(req)
        yield LLMEvent(type="token", data={"text": self._text})
        yield LLMEvent(type="done")


def _request(guided: bool) -> LLMRequest:
    return LLMRequest(
        binding=ModelBinding(
            name="fake",
            provider="fake",
            model="fake-model",
            params={"guided_decoding": guided},
        ),
        messages=[Message(role="user", content="return structured")],
    )


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
    }


@pytest.mark.asyncio
async def test_complete_structured_passes_schema_for_guided_capable_model() -> None:
    from harness.models.structured import complete_structured

    adapter = TextAdapter('{"status":"ok"}')
    gateway = ModelGateway(
        adapters={"fake": adapter},
        caps={
            "fake": ModelCaps(
                tool_calls=False,
                guided_decoding=True,
                max_context=1024,
                prompt_cache=False,
                streaming=True,
            )
        },
    )

    result = await complete_structured(gateway, _request(guided=True), _schema())

    assert result == {"status": "ok"}
    assert adapter.seen[0].output_schema == _schema()
    assert adapter.seen[0].binding.params["guided_decoding"] is True


@pytest.mark.asyncio
async def test_complete_structured_fallback_affixes_schema_prompt_and_parses_json() -> None:
    from harness.models.structured import complete_structured

    adapter = TextAdapter('```json\n{"status":"fallback"}\n```')
    gateway = ModelGateway(
        adapters={"fake": adapter},
        caps={
            "fake": ModelCaps(
                tool_calls=False,
                guided_decoding=False,
                max_context=1024,
                prompt_cache=False,
                streaming=True,
            )
        },
    )

    result = await complete_structured(gateway, _request(guided=False), _schema())

    assert result == {"status": "fallback"}
    assert adapter.seen[0].output_schema is None
    assert "Return only JSON matching this schema" in adapter.seen[0].messages[-1].content


@pytest.mark.asyncio
async def test_complete_structured_invalid_json_raises_schema_violation() -> None:
    from harness.models.structured import complete_structured

    gateway = ModelGateway(
        adapters={"fake": TextAdapter("not-json")},
        caps={
            "fake": ModelCaps(
                tool_calls=False,
                guided_decoding=False,
                max_context=1024,
                prompt_cache=False,
                streaming=True,
            )
        },
    )

    with pytest.raises(SchemaViolation):
        await complete_structured(gateway, _request(guided=False), _schema())
