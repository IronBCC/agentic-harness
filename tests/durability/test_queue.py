from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from harness.errors import FencingError
from harness.types import Event, EventKind


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


async def _insert_run(conn: asyncpg.Connection, run_id: UUID) -> None:
    await conn.execute(
        """
        INSERT INTO runs (
          run_id, root_run_id, tenant_id, principal, spec_id, spec_version,
          request_class, status, budget
        )
        VALUES ($1, $1, 'tenant-a', '{}', 'spec-a', 1, 'interactive', 'running', '{}')
        """,
        run_id,
    )


async def _insert_task(
    conn: asyncpg.Connection,
    task_id: UUID,
    run_id: UUID,
    *,
    node_id: str | None = None,
    state: str = "pending",
    available_at: datetime | None = None,
    lease_owner: str | None = None,
    lease_expires_at: datetime | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO node_tasks (
          task_id, run_id, node_id, state, available_at, lease_owner, lease_expires_at, input
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, '{}')
        """,
        task_id,
        run_id,
        node_id or f"node-{task_id.int}",
        state,
        available_at or datetime.now(UTC) - timedelta(seconds=1),
        lease_owner,
        lease_expires_at,
    )


@pytest.mark.asyncio
async def test_concurrent_claimers_never_claim_same_task(pg: str) -> None:
    from harness.durability.postgres.queue import Queue

    run_id = UUID(int=200)
    await _reset_and_migrate(pg)
    conn = await asyncpg.connect(pg)
    try:
        await _insert_run(conn, run_id)
        for index in range(100):
            await _insert_task(conn, UUID(int=10_000 + index), run_id)
    finally:
        await conn.close()

    queue = Queue(pg)
    batches = await asyncio.gather(
        *(queue.claim(worker=f"worker-{index}", n=20) for index in range(8))
    )
    claimed = [task.task_id for batch in batches for task in batch]

    assert len(claimed) == 100
    assert len(set(claimed)) == 100


@pytest.mark.asyncio
async def test_expired_lease_is_reaped_and_claimed_by_new_worker(pg: str) -> None:
    from harness.durability.postgres.queue import Queue

    run_id = UUID(int=201)
    task_id = UUID(int=201_001)
    await _reset_and_migrate(pg)
    conn = await asyncpg.connect(pg)
    try:
        await _insert_run(conn, run_id)
        await _insert_task(
            conn,
            task_id,
            run_id,
            state="claimed",
            lease_owner="stale",
            lease_expires_at=datetime.now(UTC) - timedelta(seconds=5),
        )
    finally:
        await conn.close()

    queue = Queue(pg)
    reaped = await queue.reap_expired()
    claimed = await queue.claim(worker="fresh", n=1)

    assert [task.task_id for task in reaped] == [task_id]
    assert [task.task_id for task in claimed] == [task_id]


@pytest.mark.asyncio
async def test_heartbeat_extends_lease_and_complete_marks_done(pg: str) -> None:
    from harness.durability.postgres.queue import Queue

    run_id = UUID(int=202)
    await _reset_and_migrate(pg)
    conn = await asyncpg.connect(pg)
    try:
        await _insert_run(conn, run_id)
        await _insert_task(conn, UUID(int=202_001), run_id)
    finally:
        await conn.close()

    queue = Queue(pg, lease_seconds=1)
    task = (await queue.claim(worker="worker", n=1))[0]
    before = await queue.lease_expires_at(task.task_id)

    await asyncio.sleep(0.01)
    await queue.heartbeat("worker", [task.task_id])
    after = await queue.lease_expires_at(task.task_id)
    await queue.complete(task.task_id, terminal={})

    assert after is not None and before is not None
    assert after > before
    assert await queue.task_state(task.task_id) == "done"


@pytest.mark.asyncio
async def test_reschedule_returns_task_to_pending_and_updates_attempt(pg: str) -> None:
    from harness.durability.postgres.queue import Queue

    run_id = UUID(int=203)
    await _reset_and_migrate(pg)
    conn = await asyncpg.connect(pg)
    try:
        await _insert_run(conn, run_id)
        await _insert_task(conn, UUID(int=203_001), run_id)
    finally:
        await conn.close()

    queue = Queue(pg)
    task = (await queue.claim(worker="worker", n=1))[0]
    available_at = datetime.now(UTC) - timedelta(seconds=1)
    await queue.reschedule(task.task_id, at=available_at, attempt=2)
    claimed = await queue.claim(worker="worker-2", n=1)

    assert len(claimed) == 1
    assert claimed[0].task_id == task.task_id
    assert claimed[0].attempt == 2


@pytest.mark.asyncio
async def test_takeover_makes_stale_owner_event_append_fence(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog
    from harness.durability.postgres.queue import Queue

    run_id = UUID(int=204)
    task_id = UUID(int=204_001)
    await _reset_and_migrate(pg)
    conn = await asyncpg.connect(pg)
    try:
        await _insert_run(conn, run_id)
        await _insert_task(conn, task_id, run_id)
    finally:
        await conn.close()

    queue = Queue(pg, lease_seconds=1)
    stale_task = (await queue.claim(worker="stale", n=1))[0]
    log = EventLog(pg, background_flush=False)
    await log.append(
        run_id,
        [
            Event(
                seq=1,
                node_id=stale_task.node_id,
                kind=EventKind.node_started,
                payload={},
                idempotency_key="stale-start",
            )
        ],
    )
    await log.flush()

    conn = await asyncpg.connect(pg)
    try:
        await conn.execute(
            """
            UPDATE node_tasks
            SET lease_expires_at = now() - interval '1 second'
            WHERE task_id = $1
            """,
            task_id,
        )
    finally:
        await conn.close()
    await queue.reap_expired()
    fresh_task = (await queue.claim(worker="fresh", n=1))[0]

    await log.append(
        run_id,
        [
            Event(
                seq=1,
                node_id=fresh_task.node_id,
                kind=EventKind.node_started,
                payload={},
                idempotency_key="fresh-start",
            )
        ],
    )
    with pytest.raises(FencingError):
        await log.flush()


@pytest.mark.asyncio
async def test_queue_uses_supplied_pool_instead_of_fresh_connect(
    pg: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harness.durability.postgres.queue import Queue

    run_id = UUID(int=205)
    task_id = UUID(int=205_001)
    await _reset_and_migrate(pg)
    conn = await asyncpg.connect(pg)
    try:
        await _insert_run(conn, run_id)
        await _insert_task(conn, task_id, run_id)
    finally:
        await conn.close()
    pool = await asyncpg.create_pool(pg, min_size=1, max_size=2)

    async def fail_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fresh asyncpg.connect should not be used when a pool is supplied")

    monkeypatch.setattr(asyncpg, "connect", fail_connect)
    try:
        queue = Queue(pg, pool=pool)
        claimed = await queue.claim("pool-worker", n=1)
        await queue.complete(task_id, terminal={})

        assert [task.task_id for task in claimed] == [task_id]
        assert await queue.task_state(task_id) == "done"
    finally:
        await pool.close()
