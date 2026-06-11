from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from uuid import UUID

import asyncpg
from harness.durability.postgres.eventlog import EventLog
from harness.durability.postgres.queue import Queue
from harness.types import Event, EventKind


async def run_transition_bench(dsn: str, *, transitions: int = 5_000) -> dict[str, float | int]:
    """Measure framework overhead for claim + event append + complete."""
    await _reset_and_migrate(dsn)
    run_id = UUID(int=600)
    await _insert_run_and_tasks(dsn, run_id, transitions)
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
    queue = Queue(dsn, pool=pool)
    log = EventLog(dsn, background_flush=False, pool=pool)
    samples_ms: list[float] = []
    seq = 1

    try:
        for _ in range(transitions):
            start = time.perf_counter()
            task = (await queue.claim("bench-worker", n=1))[0]
            await log.append(
                run_id,
                [
                    Event(
                        seq=seq,
                        node_id=task.node_id,
                        kind=EventKind.node_started,
                        payload={},
                        idempotency_key=f"{task.node_id}:started",
                    )
                ],
            )
            await log.flush()
            await queue.complete(task.task_id, terminal={})
            samples_ms.append((time.perf_counter() - start) * 1000)
            seq += 1
    finally:
        await log.close()
        await pool.close()

    report = {
        "transitions": transitions,
        "p50_ms": _percentile(samples_ms, 50),
        "p95_ms": _percentile(samples_ms, 95),
        "p99_ms": _percentile(samples_ms, 99),
    }
    _write_report("reports/m0/bench.json", report)
    return report


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # Python's inclusive method is stable for small benchmark samples.
    return float(statistics.quantiles(values, n=100, method="inclusive")[percentile - 1])


def _write_report(path: str, report: dict[str, float | int]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


async def _insert_run_and_tasks(dsn: str, run_id: UUID, transitions: int) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO runs (
              run_id, root_run_id, tenant_id, principal, spec_id, spec_version,
              request_class, status, budget
            )
            VALUES ($1, $1, 'tenant-a', '{}', 'm0-bench', 1, 'interactive', 'running', '{}')
            """,
            run_id,
        )
        await conn.executemany(
            """
            INSERT INTO node_tasks (task_id, run_id, node_id, state, attempt, input)
            VALUES ($1, $2, $3, 'pending', 0, '{}')
            """,
            [(UUID(int=700_000 + index), run_id, f"bench-{index}") for index in range(transitions)],
        )
    finally:
        await conn.close()
