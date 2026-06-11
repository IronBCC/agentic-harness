from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import asyncpg
from harness.durability.postgres.eventlog import EventLog
from harness.durability.postgres.queue import Queue
from harness.errors import FencingError
from harness.types import Event, EventKind

from tests.m0.probe_service import ProbeService

PHASES = ("after_claim", "mid_execute", "pre_barrier", "post_barrier_pre_complete")


async def run_crash_matrix(dsn: str) -> dict[str, object]:
    """Run deterministic crash/takeover scenarios and write a JSON report."""
    matrix: list[dict[str, object]] = []
    for index, phase in enumerate(PHASES, start=1):
        await _reset_and_migrate(dsn)
        row = await _run_phase(dsn, phase=phase, run_id=UUID(int=800 + index))
        matrix.append(row)
    report: dict[str, object] = {"matrix": matrix}
    _write_report("reports/m0/chaos.json", report)
    return report


async def _run_phase(dsn: str, *, phase: str, run_id: UUID) -> dict[str, object]:
    task_id = UUID(int=900_000 + run_id.int)
    await _insert_run_and_task(dsn, run_id, task_id)
    queue = Queue(dsn)
    log = EventLog(dsn, background_flush=False)
    probe = ProbeService()

    stale_task = (await queue.claim("stale-worker", n=1))[0]
    if phase == "after_claim":
        await _expire_claim(dsn, task_id)
    elif phase == "mid_execute":
        await _append_started(log, run_id, stale_task.node_id, seq=1)
        await _expire_claim(dsn, task_id)
    elif phase == "pre_barrier":
        await _append_started(log, run_id, stale_task.node_id, seq=1)
        await _expire_claim(dsn, task_id)
    elif phase == "post_barrier_pre_complete":
        await _append_started(log, run_id, stale_task.node_id, seq=1)
        await log.append(
            run_id,
            [
                Event(
                    seq=2,
                    node_id=stale_task.node_id,
                    kind=EventKind.checkpoint,
                    payload={"before": "write"},
                    idempotency_key=f"{phase}:checkpoint",
                    barrier=True,
                )
            ],
        )
        await probe.write("reference-write")
        await _expire_claim(dsn, task_id)
    else:
        raise AssertionError(f"unknown phase {phase}")

    await queue.reap_expired()
    fresh_task = (await queue.claim("fresh-worker", n=1))[0]
    if phase in {"after_claim", "mid_execute", "pre_barrier"}:
        await _complete_from_scratch(log, probe, queue, run_id, fresh_task.node_id, task_id)
    else:
        try:
            await log.append(
                run_id,
                [
                    Event(
                        seq=2,
                        node_id=fresh_task.node_id,
                        kind=EventKind.checkpoint,
                        payload={"before": "write"},
                        idempotency_key=f"{phase}:duplicate-checkpoint",
                        barrier=True,
                    )
                ],
            )
        except FencingError:
            pass
        await queue.complete(task_id, terminal={"ok": True})

    stuck_tasks = await _stuck_task_count(dsn)
    if phase != "post_barrier_pre_complete":
        await log.close()
    return {
        "phase": phase,
        "completed": await queue.task_state(task_id) == "done",
        "probe_effect_count": probe.effects_for("reference-write"),
        "stuck_tasks": stuck_tasks,
    }


async def _append_started(log: EventLog, run_id: UUID, node_id: str, *, seq: int) -> None:
    await log.append(
        run_id,
        [
            Event(
                seq=seq,
                node_id=node_id,
                kind=EventKind.node_started,
                payload={},
                idempotency_key=f"{node_id}:started:{seq}",
                barrier=True,
            )
        ],
    )


async def _complete_from_scratch(
    log: EventLog,
    probe: ProbeService,
    queue: Queue,
    run_id: UUID,
    node_id: str,
    task_id: UUID,
) -> None:
    loaded = await log.load(run_id)
    next_seq = max((event.seq for event in loaded), default=0) + 1
    if not loaded:
        await _append_started(log, run_id, node_id, seq=next_seq)
        next_seq += 1
    await log.append(
        run_id,
        [
            Event(
                seq=next_seq,
                node_id=node_id,
                kind=EventKind.checkpoint,
                payload={"before": "write"},
                idempotency_key=f"{node_id}:checkpoint:{next_seq}",
                barrier=True,
            )
        ],
    )
    await probe.write("reference-write")
    await log.append(
        run_id,
        [
            Event(
                seq=next_seq + 1,
                node_id=node_id,
                kind=EventKind.yielded,
                payload={"status": "done"},
                idempotency_key=f"{node_id}:yield:{next_seq + 1}",
                barrier=True,
            )
        ],
    )
    await queue.complete(task_id, terminal={"ok": True})


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


async def _insert_run_and_task(dsn: str, run_id: UUID, task_id: UUID) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO runs (
              run_id, root_run_id, tenant_id, principal, spec_id, spec_version,
              request_class, status, budget
            )
            VALUES ($1, $1, 'tenant-a', '{}', 'm0-chaos', 1, 'interactive', 'running', '{}')
            """,
            run_id,
        )
        await conn.execute(
            """
            INSERT INTO node_tasks (task_id, run_id, node_id, state, attempt, input)
            VALUES ($1, $2, 'chaos-node', 'pending', 0, '{}')
            """,
            task_id,
            run_id,
        )
    finally:
        await conn.close()


async def _expire_claim(dsn: str, task_id: UUID) -> None:
    conn = await asyncpg.connect(dsn)
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


async def _stuck_task_count(dsn: str) -> int:
    conn = await asyncpg.connect(dsn)
    try:
        value = await conn.fetchval(
            "SELECT count(*) FROM node_tasks WHERE state NOT IN ('done', 'failed')"
        )
        return int(value)
    finally:
        await conn.close()


def _write_report(path: str, report: dict[str, object]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
