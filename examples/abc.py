from __future__ import annotations

import asyncio
import json
import os

from harness.models.adapters.openai_compat import OpenAICompatAdapter
from harness.types import LLMRequest, Message, ModelBinding

DEFAULT_BASE_URL = "https://ironbccllm.tail0cc1d4.ts.net:8002/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_MODEL = "gemma4"


def build_request(model: str) -> LLMRequest:
    return LLMRequest(
        binding=ModelBinding(
            name="ironbcc-gemma",
            provider="openai_compat",
            model=model,
        ),
        messages=[
            Message(role="system", content="Return only JSON. No markdown."),
            Message(
                role="user",
                content=(
                    'Return JSON with keys "goal", "steps", and "final_answer". '
                    '"steps" must be an array of 3 objects with keys "id", "action", "risk". '
                    "Topic: failed restaurant reservation tool call."
                ),
            ),
        ],
    )


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned.removeprefix("json").strip()
    return cleaned


async def main() -> None:
    adapter = OpenAICompatAdapter(
        base_url=os.environ.get("GEMMA_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("GEMMA_API_KEY", DEFAULT_API_KEY),
    )
    request = build_request(os.environ.get("GEMMA_MODEL", DEFAULT_MODEL))
    text = ""

    async for event in adapter.complete(request):
        payload = event.model_dump(mode="json")
        print("EVENT:", payload)
        if event.type == "token":
            text += str(event.data.get("text", ""))

    print("\nRAW_TEXT:")
    print(repr(text))

    if not text.strip():
        print("\nNo token text was streamed. Check the EVENT lines above for the server shape.")
        return

    print("\nPARSED:")
    print(json.dumps(json.loads(clean_json_text(text)), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
