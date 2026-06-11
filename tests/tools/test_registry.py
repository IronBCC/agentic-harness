from __future__ import annotations

import asyncio
from uuid import UUID

import asyncpg
import pytest
from harness.errors import ToolOutcomeUnknown
from harness.types import CallCtx, Principal, ToolCall, ToolRecord, ToolResult


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


def _ctx() -> CallCtx:
    return CallCtx(
        tenant_id="tenant-a",
        principal=Principal(user_id="user-a"),
        run_id=UUID(int=1300),
        node_id="node-a",
        root_run_id=UUID(int=1300),
    )


def _record(
    tool_id: str = "weather.lookup",
    *,
    side_effect: str = "read",
    idempotency: str = "none",
    freshness: str = "session",
) -> ToolRecord:
    return ToolRecord(
        tenant_id="tenant-a",
        tool_id=tool_id,
        name=tool_id,
        description="Look up weather.",
        input_schema={"type": "object"},
        source="fixture",
        side_effect=side_effect,
        idempotency=idempotency,
        freshness=freshness,
        auth_mode="service",
        index_card="weather.lookup - Look up weather.",
    )


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.headers: list[dict[str, str]] = []

    async def __call__(self, call: ToolCall, ctx: CallCtx, headers: dict[str, str]) -> ToolResult:
        self.calls += 1
        self.headers.append(headers)
        await asyncio.sleep(0)
        return ToolResult(
            output={"calls": self.calls, "args": call.args, "user": ctx.principal.user_id}
        )


@pytest.mark.asyncio
async def test_registry_memo_hit_skips_execution(pg: str) -> None:
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    executor = CountingExecutor()
    registry = ToolRegistry(pg, executors={"fixture": executor})
    await registry.upsert_tool(_record())
    call = ToolCall(tool_id="weather.lookup", args={"city": "Paris"})

    first = await registry.invoke(call, _ctx())
    second = await registry.invoke(call, _ctx())

    assert first.output == {"calls": 1, "args": {"city": "Paris"}, "user": "user-a"}
    assert second.output == first.output
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_registry_singleflight_coalesces_concurrent_identical_reads(pg: str) -> None:
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    executor = CountingExecutor()
    registry = ToolRegistry(pg, executors={"fixture": executor})
    await registry.upsert_tool(_record())
    call = ToolCall(tool_id="weather.lookup", args={"city": "Paris"})

    results = await asyncio.gather(*(registry.invoke(call, _ctx()) for _ in range(50)))

    assert {result.output["calls"] for result in results} == {1}
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_registry_volatile_tool_never_uses_cache(pg: str) -> None:
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    executor = CountingExecutor()
    registry = ToolRegistry(pg, executors={"fixture": executor})
    await registry.upsert_tool(_record(freshness="volatile"))
    call = ToolCall(tool_id="weather.lookup", args={"city": "Paris"})

    first = await registry.invoke(call, _ctx())
    second = await registry.invoke(call, _ctx())

    assert first.output["calls"] == 1
    assert second.output["calls"] == 2


@pytest.mark.asyncio
async def test_keyed_write_sends_idempotency_header(pg: str) -> None:
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    executor = CountingExecutor()
    registry = ToolRegistry(pg, executors={"fixture": executor})
    await registry.upsert_tool(_record("crm.update", side_effect="write", idempotency="keyed"))

    await registry.invoke(ToolCall(tool_id="crm.update", args={"id": "123"}), _ctx())

    assert executor.headers == [{"idempotency-key": executor.headers[0]["idempotency-key"]}]
    assert len(executor.headers[0]["idempotency-key"]) == 64


@pytest.mark.asyncio
async def test_none_write_ambiguous_error_raises_outcome_unknown(pg: str) -> None:
    from harness.tools.registry import AmbiguousToolError, ToolRegistry

    async def failing(_call: ToolCall, _ctx: CallCtx, _headers: dict[str, str]) -> ToolResult:
        raise AmbiguousToolError("connection dropped")

    await _reset_and_migrate(pg)
    registry = ToolRegistry(pg, executors={"fixture": failing})
    await registry.upsert_tool(_record("crm.create", side_effect="write", idempotency="none"))

    with pytest.raises(ToolOutcomeUnknown):
        await registry.invoke(ToolCall(tool_id="crm.create", args={"name": "A"}), _ctx())
