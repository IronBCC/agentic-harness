"""Postgres durability backend facade."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

import asyncpg

from harness.durability.postgres.eventlog import EventLog
from harness.durability.postgres.queue import Queue
from harness.errors import ValidationFailed
from harness.types import Event, NodeTask, NodeTaskSnapshot, Principal, RunInit, RunState, RunStatus


class PostgresBackend:
    """Facade implementing the durability protocol on top of M0 primitives."""

    def __init__(
        self,
        dsn: str,
        *,
        pool: asyncpg.Pool | None = None,
        lease_seconds: int = 30,
        flush_interval_s: float = 0.1,
        background_flush: bool = True,
    ) -> None:
        self._dsn = dsn
        self._pool = pool
        self._events = EventLog(
            dsn,
            flush_interval_s=flush_interval_s,
            background_flush=background_flush,
            pool=pool,
        )
        self._queue = Queue(dsn, lease_seconds=lease_seconds, pool=pool)

    async def create_run(self, run: RunInit) -> None:
        """Insert a run row."""
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                  run_id, root_run_id, parent_run_id, tenant_id, principal,
                  spec_id, spec_version, request_class, status, depth, goal_hash,
                  budget, result
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11, $12::jsonb, $13::jsonb)
                """,
                run.run_id,
                run.root_run_id,
                run.parent_run_id,
                run.tenant_id,
                json.dumps(run.principal.model_dump(mode="json")),
                run.spec_id,
                run.spec_version,
                run.request_class,
                run.status,
                run.depth,
                run.goal_hash,
                json.dumps(run.budget),
                json.dumps(run.result) if run.result is not None else None,
            )

    async def append(self, run_id: UUID, events: list[Event]) -> None:
        """Append run events through the event log."""
        await self._events.append(run_id, events)

    async def load(self, run_id: UUID) -> RunState:
        """Load a run row and replay-ordered events."""
        async with self._connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT run_id, root_run_id, parent_run_id, tenant_id, principal,
                       spec_id, spec_version, request_class, status, depth,
                       goal_hash, budget, result, created_at, updated_at
                FROM runs
                WHERE run_id = $1
                """,
                run_id,
            )
        if row is None:
            raise ValidationFailed(f"run not found: {run_id}")

        events = await self._events.load(run_id)
        return RunState(
            run_id=row["run_id"],
            root_run_id=row["root_run_id"],
            parent_run_id=row["parent_run_id"],
            tenant_id=row["tenant_id"],
            principal=Principal.model_validate(_json_value(row["principal"])),
            spec_id=row["spec_id"],
            spec_version=row["spec_version"],
            request_class=row["request_class"],
            status=row["status"],
            depth=row["depth"],
            goal_hash=row["goal_hash"],
            budget=_json_dict(row["budget"]),
            result=_json_optional_dict(row["result"]),
            events=events,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def update_run(
        self,
        run_id: UUID,
        *,
        status: RunStatus | None = None,
        budget: dict[str, object] | None = None,
        result: dict[str, object] | None = None,
    ) -> None:
        """Update mutable run summary fields."""
        async with self._connection() as conn:
            tag = await conn.execute(
                """
                UPDATE runs
                SET status = coalesce($2, status),
                    budget = coalesce($3::jsonb, budget),
                    result = coalesce($4::jsonb, result),
                    updated_at = now()
                WHERE run_id = $1
                """,
                run_id,
                status,
                json.dumps(budget) if budget is not None else None,
                json.dumps(result) if result is not None else None,
            )
        if tag == "UPDATE 0":
            raise ValidationFailed(f"run not found: {run_id}")

    async def claim(self, worker: str, n: int) -> list[NodeTask]:
        """Claim pending tasks."""
        return await self._queue.claim(worker, n)

    async def enqueue_task(
        self,
        *,
        task_id: UUID,
        run_id: UUID,
        node_id: str,
        input: dict[str, object] | None = None,
        priority: int = 0,
        attempt: int = 0,
    ) -> None:
        """Insert a pending task for a run node."""
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO node_tasks (
                  task_id, run_id, node_id, state, attempt, priority, input
                )
                VALUES ($1, $2, $3, 'pending', $4, $5, $6::jsonb)
                """,
                task_id,
                run_id,
                node_id,
                attempt,
                priority,
                json.dumps(input or {}),
            )

    async def list_tasks(self, run_id: UUID) -> list[NodeTaskSnapshot]:
        """Return persisted task snapshots for a run."""
        async with self._connection() as conn:
            rows = await conn.fetch(
                """
                SELECT task_id, run_id, node_id, state, attempt, input,
                       lease_owner, lease_expires_at
                FROM node_tasks
                WHERE run_id = $1
                ORDER BY created_at ASC, task_id ASC
                """,
                run_id,
            )
        return [
            NodeTaskSnapshot(
                task_id=row["task_id"],
                run_id=row["run_id"],
                node_id=row["node_id"],
                state=row["state"],
                attempt=row["attempt"],
                input=_json_dict(row["input"]),
                lease_owner=row["lease_owner"],
                lease_expires_at=row["lease_expires_at"],
            )
            for row in rows
        ]

    async def heartbeat(self, worker: str, task_ids: list[UUID]) -> None:
        """Extend task leases."""
        await self._queue.heartbeat(worker, task_ids)

    async def complete(self, task_id: UUID, terminal: object) -> None:
        """Complete a claimed task."""
        await self._queue.complete(task_id, terminal)

    async def reschedule(
        self,
        task_id: UUID,
        at: datetime,
        attempt: int,
        input: dict[str, object] | None = None,
    ) -> None:
        """Reschedule a task."""
        await self._queue.reschedule(task_id, at, attempt, input)

    async def task_state(self, task_id: UUID) -> str | None:
        """Return a task state for tests and diagnostics."""
        return await self._queue.task_state(task_id)

    async def close(self) -> None:
        """Flush buffered event writes."""
        await self._events.close()

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                yield conn
            return

        conn = await asyncpg.connect(self._dsn)
        try:
            yield conn
        finally:
            await conn.close()


def _json_value(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json_dict(value: object) -> dict[str, object]:
    parsed = _json_value(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _json_optional_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    return _json_dict(value)
