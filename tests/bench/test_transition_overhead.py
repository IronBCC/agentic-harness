from __future__ import annotations

import statistics
import time
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


@pytest.mark.bench
@pytest.mark.asyncio
async def test_scenario_a_transition_p95_under_10ms(pg: str) -> None:
    from examples.scenario_a.runner import run_replay

    await _reset_and_migrate(pg)
    timings_ms: list[float] = []
    pool = await asyncpg.create_pool(pg, min_size=1, max_size=4)

    try:
        for index in range(12):
            start = time.perf_counter()
            loaded, _facts = await run_replay(pg, run_id=UUID(int=19_000 + index), pool=pool)
            elapsed_ms = (time.perf_counter() - start) * 1000
            transition_count = max(len(loaded.events), 1)
            timings_ms.append(elapsed_ms / transition_count)
    finally:
        await pool.close()

    assert _p95(timings_ms) < 10.0


def test_transition_gate_would_fail_on_injected_15ms_sleep() -> None:
    timings_ms = [1.0, 1.1, 1.2, 15.0, 15.0]

    assert _p95(timings_ms) > 10.0


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[94]
