from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from uuid import UUID

import asyncpg
from harness.durability.postgres.eventlog import EventLog
from harness.durability.postgres.queue import Queue
from harness.types import Event, EventKind

from tests.m0.probe_service import ProbeService
from tests.m0.refgraph import RefGraph, RefNode, build_reference_graph


@dataclass(frozen=True)
class ReferenceRunResult:
    events: list[Event]
    probe: ProbeService


async def run_reference_graph(dsn: str) -> ReferenceRunResult:
    run_id = UUID(int=500)
    graph = build_reference_graph()
    probe = ProbeService()
    log = EventLog(dsn, background_flush=False)
    queue = Queue(dsn)

    await _insert_run(dsn, run_id)
    await _enqueue_ready(dsn, run_id, graph, completed=set(), scheduled=set())

    completed: set[str] = set()
    scheduled: set[str] = {node.node_id for node in graph.ready(completed, scheduled=set())}
    next_seq = 1

    while len(completed) < len(graph.nodes):
        claimed = await queue.claim("m0-runner", n=20)
        if not claimed:
            raise RuntimeError("reference graph stalled with no claimable tasks")
        for task in claimed:
            node = graph.by_id[task.node_id]
            events, next_seq = await _execute_node(run_id, node, probe, next_seq)
            await log.append(run_id, events)
            await log.flush()
            await queue.complete(task.task_id, terminal={"ok": True})
            completed.add(node.node_id)
        await _enqueue_ready(dsn, run_id, graph, completed=completed, scheduled=scheduled)
        scheduled.update(node.node_id for node in graph.ready(completed, scheduled=scheduled))

    await log.append(
        run_id,
        [
            Event(
                seq=next_seq,
                node_id="run",
                kind=EventKind.run_finished,
                payload={"status": "succeeded"},
                idempotency_key="run-finished",
                barrier=True,
            )
        ],
    )
    await log.close()
    return ReferenceRunResult(events=await log.load(run_id), probe=probe)


async def _execute_node(
    run_id: UUID,
    node: RefNode,
    probe: ProbeService,
    next_seq: int,
) -> tuple[list[Event], int]:
    await asyncio.sleep(0.001)
    events = [
        Event(
            seq=next_seq,
            node_id=node.node_id,
            kind=EventKind.node_started,
            payload={},
            idempotency_key=f"{node.node_id}:started",
        ),
        Event(
            seq=next_seq + 1,
            node_id=node.node_id,
            kind=EventKind.llm_call,
            payload={"spent": 1},
            idempotency_key=f"{node.node_id}:llm",
        ),
    ]
    next_seq += 2
    if node.write_key is not None:
        events.append(
            Event(
                seq=next_seq,
                node_id=node.node_id,
                kind=EventKind.checkpoint,
                payload={"before": "write"},
                idempotency_key=f"{node.node_id}:checkpoint",
                barrier=True,
            )
        )
        next_seq += 1
        result = await probe.write(node.write_key)
        events.append(
            Event(
                seq=next_seq,
                node_id=node.node_id,
                kind=EventKind.tool_call,
                payload={"run_id": str(run_id), "result": result},
                idempotency_key=f"{node.node_id}:write",
            )
        )
        next_seq += 1
    events.append(
        Event(
            seq=next_seq,
            node_id=node.node_id,
            kind=EventKind.yielded,
            payload={"status": "done"},
            idempotency_key=f"{node.node_id}:yield",
        )
    )
    next_seq += 1
    return events, next_seq


async def _insert_run(dsn: str, run_id: UUID) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO runs (
              run_id, root_run_id, tenant_id, principal, spec_id, spec_version,
              request_class, status, budget
            )
            VALUES ($1, $1, 'tenant-a', '{}', 'm0-refgraph', 1, 'interactive', 'running', '{}')
            """,
            run_id,
        )
    finally:
        await conn.close()


async def _enqueue_ready(
    dsn: str,
    run_id: UUID,
    graph: RefGraph,
    *,
    completed: set[str],
    scheduled: set[str],
) -> None:
    ready = graph.ready(completed, scheduled)
    if not ready:
        return
    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(
            """
            INSERT INTO node_tasks (task_id, run_id, node_id, state, attempt, input)
            VALUES ($1, $2, $3, 'pending', 0, $4::jsonb)
            """,
            [
                (
                    UUID(int=10_000 + index + len(scheduled)),
                    run_id,
                    node.node_id,
                    json.dumps({}),
                )
                for index, node in enumerate(ready)
            ],
        )
    finally:
        await conn.close()

