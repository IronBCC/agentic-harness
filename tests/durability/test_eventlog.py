from __future__ import annotations

from uuid import UUID

import asyncpg
import pytest
from harness.errors import FencingError
from harness.types import Event, EventKind
from hypothesis import given
from hypothesis import strategies as st


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


async def _insert_run(dsn: str, run_id: UUID) -> None:
    conn = await asyncpg.connect(dsn)
    try:
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
    finally:
        await conn.close()


def _event(seq: int, kind: EventKind = EventKind.node_started) -> Event:
    return Event(
        seq=seq,
        node_id=f"node-{seq}",
        kind=kind,
        payload={"seq": seq},
        idempotency_key=f"event-{seq}",
    )


@pytest.mark.asyncio
async def test_append_and_load_return_events_in_sequence_order(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog

    run_id = UUID(int=100)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)

    log = EventLog(pg, background_flush=False)
    await log.append(run_id, [_event(2), _event(1)])
    await log.flush()

    loaded = await log.load(run_id)

    assert [event.seq for event in loaded] == [1, 2]
    assert [event.idempotency_key for event in loaded] == ["event-1", "event-2"]


@pytest.mark.asyncio
async def test_barrier_event_is_committed_before_append_returns(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog

    run_id = UUID(int=101)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)

    log = EventLog(pg, background_flush=False)
    await log.append(run_id, [_event(1, EventKind.node_started)])
    await log.append(
        run_id,
        [
            _event(2, EventKind.checkpoint).model_copy(update={"barrier": True}),
        ],
    )

    second_conn = await asyncpg.connect(pg)
    try:
        count = await second_conn.fetchval(
            "SELECT count(*) FROM run_events WHERE run_id = $1",
            run_id,
        )
    finally:
        await second_conn.close()

    assert count == 2


@pytest.mark.asyncio
async def test_duplicate_run_sequence_raises_fencing_error(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog

    run_id = UUID(int=102)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)

    log = EventLog(pg, background_flush=False)
    await log.append(run_id, [_event(1)])
    await log.flush()

    with pytest.raises(FencingError):
        await log.append(
            run_id,
            [
                Event(
                    seq=1,
                    node_id="other",
                    kind=EventKind.node_started,
                    payload={},
                    idempotency_key="different-key",
                )
            ],
        )
        await log.flush()


@pytest.mark.asyncio
async def test_non_barrier_append_is_buffered_until_flush(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog

    run_id = UUID(int=103)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)

    log = EventLog(pg, background_flush=False)
    await log.append(run_id, [_event(1)])

    assert await log.load(run_id) == []

    await log.flush()

    assert [event.seq for event in await log.load(run_id)] == [1]


@pytest.mark.asyncio
async def test_background_flush_commits_buffered_events(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog

    run_id = UUID(int=104)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)

    log = EventLog(pg, flush_interval_s=0.01)
    try:
        await log.append(run_id, [_event(1)])

        for _ in range(50):
            if await log.load(run_id):
                break
    finally:
        await log.close()

    assert [event.seq for event in await log.load(run_id)] == [1]


@pytest.mark.asyncio
async def test_eventlog_uses_supplied_pool_instead_of_fresh_connect(
    pg: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harness.durability.postgres.eventlog import EventLog

    run_id = UUID(int=105)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)
    pool = await asyncpg.create_pool(pg, min_size=1, max_size=2)

    async def fail_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fresh asyncpg.connect should not be used when a pool is supplied")

    monkeypatch.setattr(asyncpg, "connect", fail_connect)
    try:
        log = EventLog(pg, pool=pool, background_flush=False)
        await log.append(run_id, [_event(1).model_copy(update={"barrier": True})])

        loaded = await log.load(run_id)

        assert [event.seq for event in loaded] == [1]
    finally:
        await pool.close()


@given(st.lists(st.sampled_from(list(EventKind)), min_size=0, max_size=25))
def test_replay_is_deterministic_for_same_event_sequence(kinds: list[EventKind]) -> None:
    from harness.durability.replay import replay

    events = [
        Event(
            seq=index,
            node_id=f"node-{index}",
            kind=kind,
            payload={"spent": 1} if kind == EventKind.llm_call else {},
            idempotency_key=f"event-{index}",
        )
        for index, kind in enumerate(kinds, start=1)
    ]

    assert replay(events) == replay(events)
