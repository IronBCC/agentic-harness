from __future__ import annotations

import asyncpg
import pytest
from harness.types import EventKind


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


@pytest.mark.asyncio
async def test_reference_graph_completes_with_one_write_effect(pg: str) -> None:
    from harness.durability.replay import replay

    from tests.m0.runner import run_reference_graph

    await _reset_and_migrate(pg)

    result = await run_reference_graph(pg)
    state_a = replay(result.events)
    state_b = replay(result.events)

    assert result.probe.effects_for("reference-write") == 1
    assert len([event for event in result.events if event.kind == EventKind.node_started]) == 20
    assert len([event for event in result.events if event.kind == EventKind.run_finished]) == 1
    assert state_a == state_b
    assert state_a.finished is True

