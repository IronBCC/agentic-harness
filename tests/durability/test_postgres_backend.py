from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import signature
from uuid import UUID

import asyncpg
import pytest
from harness.types import Event, EventKind, Principal, RunInit


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


async def _insert_task(dsn: str, task_id: UUID, run_id: UUID) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO node_tasks (task_id, run_id, node_id, state, available_at, input)
            VALUES ($1, $2, 'facade-node', 'pending', $3, '{"input": true}')
            """,
            task_id,
            run_id,
            datetime.now(UTC) - timedelta(seconds=1),
        )
    finally:
        await conn.close()


def _run_init(run_id: UUID) -> RunInit:
    return RunInit(
        run_id=run_id,
        root_run_id=run_id,
        tenant_id="tenant-a",
        principal=Principal(user_id="user-a", claims={"role": "tester"}),
        spec_id="spec-a",
        spec_version=1,
        request_class="interactive",
        budget={"tokens": 1000, "usd": 1.0},
    )


@pytest.mark.asyncio
async def test_postgres_backend_conforms_to_durability_protocol(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend
    from harness.protocols import DurabilityBackend

    backend = PostgresBackend(pg, background_flush=False)

    assert isinstance(backend, DurabilityBackend)
    assert list(signature(backend.create_run).parameters) == ["run"]
    assert list(signature(backend.append).parameters) == ["run_id", "events"]
    assert list(signature(backend.load).parameters) == ["run_id"]
    assert list(signature(backend.claim).parameters) == ["worker", "n"]
    assert list(signature(backend.heartbeat).parameters) == ["worker", "task_ids"]
    assert list(signature(backend.complete).parameters) == ["task_id", "terminal"]
    assert list(signature(backend.reschedule).parameters) == ["task_id", "at", "attempt"]


@pytest.mark.asyncio
async def test_postgres_backend_create_load_and_update_run(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend

    run_id = UUID(int=300)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)

    await backend.create_run(_run_init(run_id))
    created = await backend.load(run_id)

    assert created.run_id == run_id
    assert created.status == "running"
    assert created.principal.user_id == "user-a"
    assert created.events == []

    await backend.update_run(
        run_id,
        status="succeeded",
        result={"ok": True},
        budget={"tokens": 900, "usd": 0.9},
    )
    updated = await backend.load(run_id)

    assert updated.status == "succeeded"
    assert updated.result == {"ok": True}
    assert updated.budget == {"tokens": 900, "usd": 0.9}


@pytest.mark.asyncio
async def test_postgres_backend_facade_appends_events_and_manages_tasks(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend

    run_id = UUID(int=301)
    task_id = UUID(int=301_001)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)
    await backend.create_run(_run_init(run_id))
    await _insert_task(pg, task_id, run_id)

    claimed = await backend.claim("worker-a", n=1)
    await backend.append(
        run_id,
        [
            Event(
                seq=1,
                node_id=claimed[0].node_id,
                kind=EventKind.node_started,
                payload={"via": "facade"},
                idempotency_key="facade-start",
                barrier=True,
            )
        ],
    )
    await backend.complete(task_id, terminal={"ok": True})
    loaded = await backend.load(run_id)

    assert [task.task_id for task in claimed] == [task_id]
    assert [event.idempotency_key for event in loaded.events] == ["facade-start"]
    assert await backend.task_state(task_id) == "done"

