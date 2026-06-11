from __future__ import annotations

import json
from uuid import UUID

import asyncpg
import pytest


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
async def test_scenario_a_replay_matches_expected_trace_and_facts(pg: str) -> None:
    from examples.scenario_a.runner import run_replay

    await _reset_and_migrate(pg)

    loaded, facts = await run_replay(pg, run_id=UUID(int=1800))

    expected_events = json.loads(
        open("examples/scenario_a/expected_events.json", encoding="utf-8").read()
    )
    expected_facts = json.loads(
        open("examples/scenario_a/expected_facts.json", encoding="utf-8").read()
    )

    assert loaded.status == "succeeded"
    assert [event.kind.value for event in loaded.events] == expected_events
    assert facts == expected_facts


@pytest.mark.asyncio
async def test_scenario_a_crash_replay_converges_to_same_final_facts(pg: str) -> None:
    from examples.scenario_a.runner import build_executor, load_final_facts, run_replay

    await _reset_and_migrate(pg)
    run_id = UUID(int=1801)
    executor, backend, run = build_executor(pg, run_id=run_id)
    await executor.seed_run(run, input={"case": "scenario-a"})
    await executor.run_once()

    resumed, _backend, _run = build_executor(pg, run_id=run_id)
    await resumed.run_until_idle()
    loaded = await backend.load(run_id)

    clean_loaded, clean_facts = await run_replay(pg, run_id=UUID(int=1802))

    assert loaded.result == clean_loaded.result
    assert load_final_facts(loaded) == clean_facts


def test_scenario_a_fixture_server_has_three_tools() -> None:
    from examples.scenario_a.fixture_server import fixture_tools

    tools = fixture_tools()

    assert [tool["name"] for tool in tools] == ["ticket.lookup", "kb.search", "ticket.update"]
    assert tools[-1]["annotations"] == {"side_effect": "write", "idempotency": "keyed"}
