from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
from harness.errors import MissingCassette
from harness.types import LLMEvent, ToolResult


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


async def _live_llm() -> AsyncIterator[LLMEvent]:
    yield LLMEvent(type="token", data={"text": "hi"})
    yield LLMEvent(type="usage", data={"input_tokens": 1, "output_tokens": 1, "usd": 0.0})
    yield LLMEvent(type="done")


async def _live_tool() -> ToolResult:
    return ToolResult(output={"ok": True}, artifact_hint="artifact:1")


@pytest.mark.asyncio
async def test_record_then_replay_produces_byte_identical_events_and_tool_result(
    pg: str,
    tmp_path: Path,
) -> None:
    from harness.testing.cassettes import CassetteSession, CassetteStore

    await _reset_and_migrate(pg)
    store = CassetteStore(pg)
    recorder = CassetteSession(store, mode="record")
    llm_recorded = [event async for event in recorder.llm({"case": "a"}, _live_llm)]
    tool_recorded = await recorder.tool({"case": "tool"}, _live_tool)

    replayer = CassetteSession(store, mode="replay")
    llm_replayed = [event async for event in replayer.llm({"case": "a"}, _live_llm)]
    tool_replayed = await replayer.tool({"case": "tool"}, _live_tool)

    assert [event.model_dump(mode="json") for event in llm_replayed] == [
        event.model_dump(mode="json") for event in llm_recorded
    ]
    assert tool_replayed == tool_recorded

    exported = tmp_path / "cassettes.jsonl"
    await store.export_jsonl(exported)
    await _reset_and_migrate(pg)
    await store.import_jsonl(exported)
    assert [event async for event in replayer.llm({"case": "a"}, _live_llm)] == llm_recorded


@pytest.mark.asyncio
async def test_replay_miss_raises_missing_cassette(pg: str) -> None:
    from harness.testing.cassettes import CassetteSession, CassetteStore

    await _reset_and_migrate(pg)
    session = CassetteSession(CassetteStore(pg), mode="replay")

    with pytest.raises(MissingCassette):
        _ = [event async for event in session.llm({"missing": True}, _live_llm)]


@pytest.mark.asyncio
async def test_hybrid_records_miss_then_replays(pg: str) -> None:
    from harness.testing.cassettes import CassetteSession, CassetteStore

    await _reset_and_migrate(pg)
    store = CassetteStore(pg)
    session = CassetteSession(store, mode="hybrid")

    first = [event async for event in session.llm({"case": "hybrid"}, _live_llm)]
    second = [event async for event in session.llm({"case": "hybrid"}, _live_llm)]

    assert second == first
