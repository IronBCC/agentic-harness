from __future__ import annotations

from uuid import UUID

import asyncpg
import pytest
from harness.errors import SchemaViolation
from harness.types import EventKind, YieldStatus


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


def test_node_lifecycle_covers_success_and_failure_transitions() -> None:
    from harness.engine.lifecycle import NodeLifecycle, NodePhase

    lifecycle = NodeLifecycle()

    assert lifecycle.state == NodePhase.pending
    assert lifecycle.claim() == NodePhase.claimed
    assert lifecycle.start_assembling() == NodePhase.assembling
    assert lifecycle.start_executing() == NodePhase.executing
    assert lifecycle.mark_yielded() == NodePhase.yielded
    assert lifecycle.complete() == NodePhase.completed

    assert NodeLifecycle().claim().lease_expired() == NodePhase.pending
    assert NodeLifecycle().claim().start_assembling().assembly_failed() == NodePhase.failed
    assert (
        NodeLifecycle()
        .claim()
        .start_assembling()
        .start_executing()
        .infra_retryable()
        == NodePhase.retry_wait
    )
    assert (
        NodeLifecycle()
        .claim()
        .start_assembling()
        .start_executing()
        .mark_yielded()
        .escalate()
        == NodePhase.escalated
    )
    assert (
        NodeLifecycle()
        .claim()
        .start_assembling()
        .start_executing()
        .mark_yielded()
        .wait()
        == NodePhase.waiting
    )
    assert (
        NodeLifecycle()
        .claim()
        .start_assembling()
        .start_executing()
        .mark_yielded()
        .fail()
        == NodePhase.failed
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"status": "done", "facts": [{"ok": True}], "result_ref": "artifact:1"},
        '{"status":"error","class":"tool_rejected","detail":{"code":400}}',
    ],
)
def test_parse_yield_accepts_strict_dict_or_json(raw: dict[str, object] | str) -> None:
    from harness.engine.lifecycle import parse_yield

    parsed = parse_yield(raw)

    assert parsed.status in {YieldStatus.done, YieldStatus.error}


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        {"status": "not-a-status"},
        {"status": "done", "extra": True},
        ["not", "a", "yield"],
    ],
)
def test_parse_yield_malformed_raises_schema_violation(raw: object) -> None:
    from harness.engine.lifecycle import parse_yield

    with pytest.raises(SchemaViolation):
        parse_yield(raw)


@pytest.mark.parametrize(
    ("raw", "action", "terminal_state", "planner_visible"),
    [
        ({"status": "done"}, "complete", "completed", False),
        ({"status": "error", "class": "infra_retryable"}, "retry", "retry_wait", False),
        ({"status": "error", "class": "schema_violation"}, "retry", "retry_wait", False),
        ({"status": "error", "class": "tool_rejected"}, "emit_fact", "failed", True),
        ({"status": "error", "class": "semantic_failure"}, "emit_fact", "failed", True),
        ({"status": "error", "class": "policy_denied"}, "emit_fact", "failed", True),
    ],
)
def test_route_yield_maps_terminal_statuses(
    raw: dict[str, object],
    action: str,
    terminal_state: str,
    planner_visible: bool,
) -> None:
    from harness.engine.lifecycle import route_yield

    route = route_yield(raw)

    assert route.action == action
    assert route.terminal_state == terminal_state
    assert route.planner_visible == planner_visible


@pytest.mark.parametrize(
    ("raw", "ticket"),
    [
        ({"status": "need_capability", "need": "calendar"}, "M2-09"),
        ({"status": "low_confidence", "reason": "ambiguous"}, "M2-09"),
        ({"status": "need_user_input", "questions": []}, "M3-11"),
        ({"status": "wrap_up_ack", "summary_fact": {}}, "M3-11"),
    ],
)
def test_route_yield_stubs_future_statuses(raw: dict[str, object], ticket: str) -> None:
    from harness.engine.lifecycle import route_yield

    with pytest.raises(NotImplementedError, match=ticket):
        route_yield(raw)


@pytest.mark.asyncio
async def test_write_tool_call_is_preceded_by_barrier_checkpoint_in_event_log(pg: str) -> None:
    from harness.durability.postgres.eventlog import EventLog
    from harness.engine.lifecycle import tool_call_events

    run_id = UUID(int=500)
    await _reset_and_migrate(pg)
    await _insert_run(pg, run_id)
    log = EventLog(pg, background_flush=False)

    events = tool_call_events(
        seq=1,
        node_id="write-node",
        tool_id="crm.update",
        side_effect="write",
        payload={"args": {"status": "closed"}},
        idempotency_key="call-1",
    )
    await log.append(run_id, events)

    loaded = await log.load(run_id)

    assert [(event.seq, event.kind, event.barrier) for event in loaded] == [
        (1, EventKind.checkpoint, True),
        (2, EventKind.tool_call, False),
    ]
    assert loaded[0].payload == {"before": "tool_call", "tool_id": "crm.update"}


def test_read_tool_call_has_no_checkpoint_barrier() -> None:
    from harness.engine.lifecycle import tool_call_events

    events = tool_call_events(
        seq=10,
        node_id="read-node",
        tool_id="crm.lookup",
        side_effect="read",
        payload={"args": {"id": "123"}},
        idempotency_key="call-2",
    )

    assert [(event.seq, event.kind, event.barrier) for event in events] == [
        (10, EventKind.tool_call, False)
    ]
