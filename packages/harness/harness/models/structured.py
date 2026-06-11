"""Structured output enforcement over the model gateway."""

from __future__ import annotations

import json

from harness.errors import SchemaViolation
from harness.models.gateway import ModelGateway
from harness.types import LLMRequest, Message


async def complete_structured(
    gateway: ModelGateway,
    req: LLMRequest,
    schema: dict[str, object],
) -> dict[str, object]:
    """Complete a request and parse one JSON object matching the caller schema."""
    caps = gateway.capabilities(req.binding)
    structured_req = req
    if caps.guided_decoding and _guided_requested(req):
        structured_req = req.model_copy(update={"output_schema": schema})
    else:
        structured_req = _with_schema_prompt(req, schema)

    text = ""
    async for event in gateway.complete(structured_req):
        if event.type == "token":
            text += str(event.data.get("text", ""))

    return _parse_json_object(text)


def _guided_requested(req: LLMRequest) -> bool:
    value = req.binding.params.get("guided_decoding", False)
    return value if isinstance(value, bool) else False


def _with_schema_prompt(req: LLMRequest, schema: dict[str, object]) -> LLMRequest:
    messages = [
        *req.messages,
        Message(
            role="system",
            content=(
                "Return only JSON matching this schema. No markdown. "
                f"Schema: {json.dumps(schema, sort_keys=True)}"
            ),
        ),
    ]
    return req.model_copy(update={"messages": messages, "output_schema": None})


def _parse_json_object(text: str) -> dict[str, object]:
    cleaned = _clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SchemaViolation("structured model output was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise SchemaViolation("structured model output must be a JSON object")
    return parsed


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned.removeprefix("json").strip()
    return cleaned
